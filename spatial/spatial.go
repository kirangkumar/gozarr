// Package spatial adds geographic point lookups on top of a Zarr array: map a
// latitude/longitude to a grid cell, then read just that cell (or its four
// neighbours for bilinear interpolation) without touching the rest of the grid.
//
// The coordinate arrays are loaded once. After that a lookup is two binary
// searches plus a chunk-cache hit, which is well under a microsecond.
package spatial

import (
	"context"
	"errors"
	"fmt"
	"math"
	"sort"

	zarr "github.com/kirangkumar/gozarr"
)

// ErrOutside is returned when a point lies outside the grid.
var ErrOutside = errors.New("spatial: point outside grid")

// Grid is a rectilinear lat/lon grid described by two monotonic 1-D coordinate
// vectors (ascending or descending), which covers the regular and the
// Gaussian/reduced-regular-in-one-axis grids used by most weather and climate
// products. 2-D curvilinear coordinates are not supported.
type Grid struct {
	Lat, Lon         []float64
	latDesc, lonDesc bool
}

// NewGrid builds a Grid from coordinate vectors, which it keeps (no copy).
func NewGrid(lat, lon []float64) (*Grid, error) {
	g := &Grid{Lat: lat, Lon: lon}
	var err error
	if g.latDesc, err = monotonic(lat); err != nil {
		return nil, fmt.Errorf("spatial: latitude: %w", err)
	}
	if g.lonDesc, err = monotonic(lon); err != nil {
		return nil, fmt.Errorf("spatial: longitude: %w", err)
	}
	return g, nil
}

func monotonic(v []float64) (desc bool, err error) {
	if len(v) < 1 {
		return false, errors.New("empty coordinate")
	}
	if len(v) == 1 {
		return false, nil
	}
	desc = v[1] < v[0]
	for i := 1; i < len(v); i++ {
		if (v[i] < v[i-1]) != desc || v[i] == v[i-1] {
			return false, errors.New("coordinate is not strictly monotonic")
		}
	}
	return desc, nil
}

// LoadGrid reads the coordinate arrays named latName and lonName from a group.
func LoadGrid(ctx context.Context, g *zarr.Group, latName, lonName string) (*Grid, error) {
	read := func(name string) ([]float64, error) {
		a, err := g.Array(ctx, name, zarr.WithCache(zarr.NewCache(0)))
		if err != nil {
			return nil, err
		}
		if len(a.Shape()) != 1 {
			return nil, fmt.Errorf("spatial: %s is not one-dimensional", name)
		}
		d, err := a.Read(ctx)
		if err != nil {
			return nil, err
		}
		return d.Float64(), nil
	}
	lat, err := read(latName)
	if err != nil {
		return nil, err
	}
	lon, err := read(lonName)
	if err != nil {
		return nil, err
	}
	return NewGrid(lat, lon)
}

// normLon maps a query longitude into the numeric range the grid uses, so a
// 0..360 grid accepts -75 and a -180..180 grid accepts 285.
func (g *Grid) normLon(lon float64) float64 {
	lo, hi := g.Lon[0], g.Lon[len(g.Lon)-1]
	if g.lonDesc {
		lo, hi = hi, lo
	}
	for lon < lo-180 && lon+360 <= hi+180 {
		lon += 360
	}
	for lon > hi+180 && lon-360 >= lo-180 {
		lon -= 360
	}
	if lon < lo && lon+360 <= hi+(hi-lo)/float64(len(g.Lon)) {
		lon += 360
	}
	if lon > hi && lon-360 >= lo-(hi-lo)/float64(len(g.Lon)) {
		lon -= 360
	}
	return lon
}

// bracket finds i such that coordinate c lies between v[i] and v[i+1]
// (in value order, whichever way v runs) and the fraction of the way across.
func bracket(v []float64, desc bool, c float64) (i int, frac float64, ok bool) {
	n := len(v)
	if n == 1 {
		return 0, 0, c == v[0]
	}
	var j int
	if desc {
		j = sort.Search(n, func(k int) bool { return v[k] <= c }) // first v[k] <= c
	} else {
		j = sort.Search(n, func(k int) bool { return v[k] >= c }) // first v[k] >= c
	}
	switch {
	case j == 0:
		if v[0] == c {
			return 0, 0, true
		}
		return 0, 0, false
	case j == n:
		return 0, 0, false
	}
	i = j - 1
	return i, (c - v[i]) / (v[j] - v[i]), true
}

// Cell is a grid position.
type Cell struct{ I, J int } // I indexes latitude, J longitude

// Nearest returns the cell whose centre is closest to the point. Points up to
// half a cell beyond the grid edge snap to the edge cell; anything further out
// gives ErrOutside.
func (g *Grid) Nearest(lat, lon float64) (Cell, error) {
	i, err := nearest(g.Lat, g.latDesc, lat)
	if err != nil {
		return Cell{}, err
	}
	j, err := nearest(g.Lon, g.lonDesc, g.normLon(lon))
	if err != nil {
		return Cell{}, err
	}
	return Cell{i, j}, nil
}

func nearest(v []float64, desc bool, c float64) (int, error) {
	n := len(v)
	if n == 1 {
		return 0, nil
	}
	i, f, ok := bracket(v, desc, c)
	if !ok {
		// within half a cell of either end?
		first, second := v[0], v[1]
		last, prev := v[n-1], v[n-2]
		switch {
		case math.Abs(c-first) <= math.Abs(second-first)/2:
			return 0, nil
		case math.Abs(c-last) <= math.Abs(last-prev)/2:
			return n - 1, nil
		}
		return 0, ErrOutside
	}
	if f >= 0.5 {
		return i + 1, nil
	}
	return i, nil
}

// Weights describes a bilinear stencil: the four surrounding cells and the
// fractional position (0..1) between them.
type Weights struct {
	I0, J0 int     // lower cell indices; the stencil covers I0..I0+1, J0..J0+1
	FI, FJ float64 // fraction along latitude / longitude index
}

// Bilinear locates the 2x2 stencil around a point. The point must lie inside
// the grid.
func (g *Grid) Bilinear(lat, lon float64) (Weights, error) {
	i, fi, ok := bracket(g.Lat, g.latDesc, lat)
	if !ok {
		return Weights{}, ErrOutside
	}
	j, fj, ok := bracket(g.Lon, g.lonDesc, g.normLon(lon))
	if !ok {
		return Weights{}, ErrOutside
	}
	return Weights{i, j, fi, fj}, nil
}

// Field is an array laid out over a Grid.
type Field struct {
	Arr              *zarr.Array
	Grid             *Grid
	LatAxis, LonAxis int
}

// NewField binds an array to a grid. latAxis and lonAxis are the array axes
// that hold latitude and longitude.
func NewField(a *zarr.Array, g *Grid, latAxis, lonAxis int) (*Field, error) {
	sh := a.Shape()
	if latAxis < 0 || latAxis >= len(sh) || lonAxis < 0 || lonAxis >= len(sh) || latAxis == lonAxis {
		return nil, errors.New("spatial: bad lat/lon axis")
	}
	if sh[latAxis] != len(g.Lat) || sh[lonAxis] != len(g.Lon) {
		return nil, fmt.Errorf("spatial: array is %dx%d on (lat,lon) but grid is %dx%d",
			sh[latAxis], sh[lonAxis], len(g.Lat), len(g.Lon))
	}
	return &Field{a, g, latAxis, lonAxis}, nil
}

// OpenField opens a named array in a group together with its coordinates.
// Axes are found from dimension names ("lat"/"latitude", "lon"/"longitude").
func OpenField(ctx context.Context, grp *zarr.Group, name, latName, lonName string, opts ...zarr.Option) (*Field, error) {
	a, err := grp.Array(ctx, name, opts...)
	if err != nil {
		return nil, err
	}
	g, err := LoadGrid(ctx, grp, latName, lonName)
	if err != nil {
		return nil, err
	}
	la, lo := -1, -1
	for i, d := range a.DimNames() {
		switch d {
		case latName, "lat", "latitude":
			la = i
		case lonName, "lon", "longitude":
			lo = i
		}
	}
	if la < 0 || lo < 0 {
		return nil, fmt.Errorf("spatial: cannot find lat/lon axes of %s in dimension names %v; use NewField", name, a.DimNames())
	}
	return NewField(a, g, la, lo)
}

// other fills a full index vector from the non-spatial positions in idx.
func (f *Field) full(idx []int, ci, cj int) ([]int, error) {
	nd := len(f.Arr.Shape())
	if len(idx) != nd-2 {
		return nil, fmt.Errorf("spatial: need %d indices for the non-spatial axes, got %d", nd-2, len(idx))
	}
	out := make([]int, nd)
	k := 0
	for a := 0; a < nd; a++ {
		switch a {
		case f.LatAxis:
			out[a] = ci
		case f.LonAxis:
			out[a] = cj
		default:
			out[a] = idx[k]
			k++
		}
	}
	return out, nil
}

// Nearest returns the value of the cell closest to (lat, lon). idx gives the
// positions along the remaining axes in array order, e.g. the time index.
func (f *Field) Nearest(ctx context.Context, lat, lon float64, idx ...int) (float64, error) {
	c, err := f.Grid.Nearest(lat, lon)
	if err != nil {
		return 0, err
	}
	full, err := f.full(idx, c.I, c.J)
	if err != nil {
		return 0, err
	}
	return f.Arr.At(ctx, full...)
}

// Interp bilinearly interpolates between the four surrounding cells. Cells
// holding NaN are left out and the remaining weights renormalised; if all four
// are NaN the result is NaN.
func (f *Field) Interp(ctx context.Context, lat, lon float64, idx ...int) (float64, error) {
	w, err := f.Grid.Bilinear(lat, lon)
	if err != nil {
		return 0, err
	}
	var sum, wsum float64
	for di := 0; di <= 1; di++ {
		for dj := 0; dj <= 1; dj++ {
			i, j := w.I0+di, w.J0+dj
			if i >= len(f.Grid.Lat) || j >= len(f.Grid.Lon) {
				continue // single-row/column grid
			}
			wt := 1.0
			if di == 0 {
				wt *= 1 - w.FI
			} else {
				wt *= w.FI
			}
			if dj == 0 {
				wt *= 1 - w.FJ
			} else {
				wt *= w.FJ
			}
			if wt == 0 {
				continue
			}
			full, err := f.full(idx, i, j)
			if err != nil {
				return 0, err
			}
			v, err := f.Arr.At(ctx, full...)
			if err != nil {
				return 0, err
			}
			if math.IsNaN(v) {
				continue
			}
			sum += v * wt
			wsum += wt
		}
	}
	if wsum == 0 {
		return math.NaN(), nil
	}
	return sum / wsum, nil
}

// Series reads a run [start, stop) along one non-spatial axis at the cell
// nearest to (lat, lon). idx gives positions for the remaining non-spatial
// axes in array order (the series axis is skipped).
func (f *Field) Series(ctx context.Context, lat, lon float64, axis, start, stop int, idx ...int) ([]float64, error) {
	c, err := f.Grid.Nearest(lat, lon)
	if err != nil {
		return nil, err
	}
	nd := len(f.Arr.Shape())
	if axis < 0 || axis >= nd || axis == f.LatAxis || axis == f.LonAxis {
		return nil, errors.New("spatial: series axis must be a non-spatial axis")
	}
	if len(idx) != nd-3 {
		return nil, fmt.Errorf("spatial: need %d indices besides the series axis, got %d", nd-3, len(idx))
	}
	sel := make([]zarr.Slice, nd)
	k := 0
	for a := 0; a < nd; a++ {
		switch a {
		case f.LatAxis:
			sel[a] = zarr.Index(c.I)
		case f.LonAxis:
			sel[a] = zarr.Index(c.J)
		case axis:
			sel[a] = zarr.Range(start, stop)
		default:
			sel[a] = zarr.Index(idx[k])
			k++
		}
	}
	d, err := f.Arr.Read(ctx, sel...)
	if err != nil {
		return nil, err
	}
	return d.Float64(), nil
}
