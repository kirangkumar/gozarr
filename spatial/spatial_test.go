package spatial

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"testing"

	zarr "github.com/kirangkumar/gozarr"
)

func fixtures(t testing.TB) map[string]struct {
	Flat []*float64 `json:"flat"`
} {
	b, err := os.ReadFile("../testdata/expected.json")
	if err != nil {
		t.Skip("fixtures missing; run python internal/gen/gen.py")
	}
	var m map[string]struct {
		Flat []*float64 `json:"flat"`
	}
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	return m
}

func TestNearestAndInterp(t *testing.T) {
	exp := fixtures(t)
	ctx := context.Background()
	for _, grp := range []string{"v2_group.zarr", "v3_group.zarr"} {
		t.Run(grp, func(t *testing.T) {
			g, err := zarr.OpenGroup(ctx, zarr.NewFSStore("../testdata"), grp)
			if err != nil {
				t.Fatal(err)
			}
			f, err := OpenField(ctx, g, "temp", "lat", "lon")
			if err != nil {
				t.Fatal(err)
			}
			if f.LatAxis != 1 || f.LonAxis != 2 {
				t.Fatalf("axes %d %d", f.LatAxis, f.LonAxis)
			}
			temp := exp[grp+"/temp"].Flat
			at := func(tm, i, j int) float64 { return *temp[(tm*40+i)*50+j] }

			// Exactly on a cell centre.
			lat, lon := f.Grid.Lat[12], f.Grid.Lon[30]
			v, err := f.Nearest(ctx, lat, lon, 5)
			if err != nil || v != at(5, 12, 30) {
				t.Fatalf("nearest on-centre: %v %v want %v", v, err, at(5, 12, 30))
			}
			// Slightly off-centre still snaps to it.
			v, _ = f.Nearest(ctx, lat+0.1, lon-0.1, 5)
			if v != at(5, 12, 30) {
				t.Fatalf("nearest off-centre %v", v)
			}
			// Interpolation at the midpoint of 4 cells is their mean (grid lat is descending).
			mlat := (f.Grid.Lat[12] + f.Grid.Lat[13]) / 2
			mlon := (f.Grid.Lon[30] + f.Grid.Lon[31]) / 2
			want := (at(5, 12, 30) + at(5, 12, 31) + at(5, 13, 30) + at(5, 13, 31)) / 4
			got, err := f.Interp(ctx, mlat, mlon, 5)
			if err != nil || math.Abs(got-want) > 1e-4 {
				t.Fatalf("interp %v %v want %v", got, err, want)
			}
			// Interp on a node returns the node.
			got, _ = f.Interp(ctx, lat, lon, 5)
			if math.Abs(got-at(5, 12, 30)) > 1e-4 {
				t.Fatalf("interp at node %v want %v", got, at(5, 12, 30))
			}
			// Time series.
			s, err := f.Series(ctx, lat, lon, 0, 3, 9)
			if err != nil || len(s) != 6 {
				t.Fatalf("series: %v %v", s, err)
			}
			for k, v := range s {
				if v != at(3+k, 12, 30) {
					t.Fatalf("series[%d]=%v want %v", k, v, at(3+k, 12, 30))
				}
			}
			// Outside.
			if _, err := f.Nearest(ctx, 60, 80, 0); err != ErrOutside {
				t.Fatalf("want ErrOutside, got %v", err)
			}
			if _, err := f.Interp(ctx, 20, 120, 0); err != ErrOutside {
				t.Fatalf("want ErrOutside, got %v", err)
			}
		})
	}
}

func TestLongitudeWrap(t *testing.T) {
	g, _ := NewGrid([]float64{0, 1, 2}, []float64{350, 355, 0 + 360})
	c, err := g.Nearest(1, -5) // -5 == 355
	if err != nil || c.J != 1 {
		t.Fatalf("got %+v %v", c, err)
	}
	g2, _ := NewGrid([]float64{2, 1, 0}, []float64{-10, -5, 0, 5})
	if c, err := g2.Nearest(0.9, 357); err != nil || c.I != 1 || c.J != 1 { // 357 == -3 -> nearest -5? no: -3 is closer to -5(2) than 0(3)
		t.Fatalf("got %+v %v", c, err)
	}
}

func TestEdgeSnap(t *testing.T) {
	g, _ := NewGrid([]float64{10, 11, 12}, []float64{70, 71, 72})
	if c, err := g.Nearest(9.6, 72.4); err != nil || c != (Cell{0, 2}) {
		t.Fatalf("%+v %v", c, err)
	}
	if _, err := g.Nearest(9.4, 70); err != ErrOutside {
		t.Fatalf("want outside, got %v", err)
	}
}
