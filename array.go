package zarr

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
)

var nextArrayID atomic.Uint32

// Array is an opened Zarr array. It is safe for concurrent use. Opening reads
// only the metadata; chunk data is fetched lazily and only for the regions you
// ask for.
type Array struct {
	store Store
	path  string
	meta  *arrayMeta
	id    uint32
	cache *Cache
	par   int

	leaf      []int // shape of the unit that is decoded and cached
	ratio     []int // leaf chunks per stored chunk along each axis (sharding)
	grid      []int // number of leaf chunks along each axis
	fillChunk []byte
}

// Option configures Open.
type Option func(*openOpts)

type openOpts struct {
	cache *Cache
	par   int
}

// WithCache uses c for decoded chunks instead of the process-wide default
// (256 MiB). Share one Cache between arrays to bound total memory.
func WithCache(c *Cache) Option { return func(o *openOpts) { o.cache = c } }

// WithConcurrency caps the number of chunks fetched in parallel by one Read.
func WithConcurrency(n int) Option { return func(o *openOpts) { o.par = n } }

// Open opens the array stored at path inside store. The format version (v2 or
// v3) is detected. If a parent group has consolidated metadata, use
// Group.Array instead: it needs no extra requests.
func Open(ctx context.Context, store Store, path string, opts ...Option) (*Array, error) {
	path = strings.Trim(path, "/")
	sub := SubStore(store, path)
	if b, err := sub.Get(ctx, "zarr.json"); err == nil {
		var probe struct {
			NodeType string `json:"node_type"`
		}
		_ = decodeJSON(b, &probe)
		if probe.NodeType == "group" {
			return nil, fmt.Errorf("zarr: %q is a group, not an array", path)
		}
		m, err := parseV3(b)
		if err != nil {
			return nil, err
		}
		return newArray(store, path, m, opts)
	} else if !errors.Is(err, ErrNotFound) {
		return nil, err
	}
	b, err := sub.Get(ctx, ".zarray")
	if err != nil {
		if errors.Is(err, ErrNotFound) {
			return nil, fmt.Errorf("zarr: no array at %q: %w", path, ErrNotFound)
		}
		return nil, err
	}
	var attrs map[string]any
	if ab, err := sub.Get(ctx, ".zattrs"); err == nil {
		_ = decodeJSON(ab, &attrs)
	}
	m, err := parseV2(b, attrs)
	if err != nil {
		return nil, err
	}
	return newArray(store, path, m, opts)
}

func newArray(store Store, path string, m *arrayMeta, opts []Option) (*Array, error) {
	o := openOpts{par: 16}
	for _, f := range opts {
		f(&o)
	}
	if o.cache == nil {
		o.cache = defaultCache()
	}
	nd := len(m.shape)
	if nd > maxDims {
		return nil, fmt.Errorf("zarr: %d dimensions exceeds the supported maximum of %d", nd, maxDims)
	}
	for i := range m.shape {
		if m.shape[i] < 0 || m.chunkShape[i] <= 0 {
			return nil, fmt.Errorf("zarr: invalid shape or chunk shape")
		}
	}
	if m.dimNames == nil {
		if d, ok := m.attrs["_ARRAY_DIMENSIONS"].([]any); ok {
			for _, v := range d {
				s, _ := v.(string)
				m.dimNames = append(m.dimNames, s)
			}
		}
	}
	if o.par < 1 {
		o.par = 1
	}
	a := &Array{store: SubStore(store, path), path: path, meta: m, id: nextArrayID.Add(1), cache: o.cache, par: o.par}
	a.leaf = m.chunkShape
	a.ratio = make([]int, nd)
	for i := range a.ratio {
		a.ratio[i] = 1
	}
	if m.shard != nil {
		a.leaf = m.shard.innerShape
		for i := range a.ratio {
			a.ratio[i] = m.chunkShape[i] / a.leaf[i]
		}
	}
	a.grid = make([]int, nd)
	for i := range a.grid {
		a.grid[i] = (m.shape[i] + a.leaf[i] - 1) / a.leaf[i]
	}
	a.fillChunk = make([]byte, prod(a.leaf)*m.dtype.Size())
	fillBuf(a.fillChunk, m.fill)
	return a, nil
}

// Shape returns the array shape.
func (a *Array) Shape() []int { return atoiSlice(a.meta.shape) }

// ChunkShape returns the shape of one stored chunk (the shard, if sharded).
func (a *Array) ChunkShape() []int { return atoiSlice(a.meta.chunkShape) }

// DType returns the element type.
func (a *Array) DType() DType { return a.meta.dtype }

// Path returns the array's path within its store.
func (a *Array) Path() string { return a.path }

// Attrs returns the user attributes (a shared map; do not modify).
func (a *Array) Attrs() map[string]any { return a.meta.attrs }

// DimNames returns dimension names from v3 metadata or the xarray
// "_ARRAY_DIMENSIONS" attribute; nil if unnamed.
func (a *Array) DimNames() []string { return a.meta.dimNames }

// Version returns the Zarr format version, 2 or 3.
func (a *Array) Version() int { return a.meta.version }

// FillValue returns the fill value as float64.
func (a *Array) FillValue() float64 { return loadFloat64(a.meta.dtype, a.meta.fill, 0) }

// ------------------------------------------------------------------ chunk I/O

// leafBytes returns the decoded leaf chunk at coords (in leaf units).
// The returned slice is shared with the cache: read-only.
func (a *Array) leafBytes(ctx context.Context, coords []int) ([]byte, error) {
	var k cacheKey
	k.arr = a.id
	for i, c := range coords {
		k.c[i] = uint32(c)
	}
	if d, ok := a.cache.peek(k); ok {
		return d, nil
	}
	cs := append([]int(nil), coords...)
	return a.cache.get(k, func() ([]byte, error) { return a.loadLeaf(ctx, cs) })
}

func (a *Array) loadLeaf(ctx context.Context, coords []int) ([]byte, error) {
	m := a.meta
	if m.shard == nil {
		raw, err := a.store.Get(ctx, m.keyFn(coords))
		if errors.Is(err, ErrNotFound) {
			return a.fillChunk, nil
		}
		if err != nil {
			return nil, err
		}
		out, err := m.pipe.decode(raw)
		if err != nil {
			return nil, fmt.Errorf("zarr: decoding chunk %v of %s: %w", coords, a.path, err)
		}
		return a.checkLen(out, coords)
	}
	sp := m.shard
	nd := len(coords)
	var sc, ic [maxDims]int
	pos := 0
	for i := 0; i < nd; i++ {
		sc[i] = coords[i] / a.ratio[i]
		ic[i] = coords[i] % a.ratio[i]
		pos = pos*sp.perShard[i] + ic[i]
	}
	key := m.keyFn(sc[:nd])
	index, err := a.shardIndex(ctx, key, sc[:nd])
	if err != nil {
		return nil, err
	}
	if index == nil { // shard absent
		return a.fillChunk, nil
	}
	off, n, ok := sp.entry(index, pos)
	if !ok {
		return a.fillChunk, nil
	}
	raw, err := a.store.GetRange(ctx, key, int64(off), int64(n))
	if err != nil {
		return nil, err
	}
	if uint64(len(raw)) != n {
		return nil, fmt.Errorf("zarr: short read of shard %s: got %d of %d bytes", key, len(raw), n)
	}
	out, err := sp.decodeInner(raw, a.meta.fill)
	if err != nil {
		return nil, fmt.Errorf("zarr: decoding inner chunk %v of %s: %w", coords, a.path, err)
	}
	return a.checkLen(out, coords)
}

func (a *Array) checkLen(out []byte, coords []int) ([]byte, error) {
	if len(out) != len(a.fillChunk) {
		return nil, fmt.Errorf("zarr: chunk %v of %s decoded to %d bytes, want %d", coords, a.path, len(out), len(a.fillChunk))
	}
	return out, nil
}

// shardIndex returns the validated index entries of a shard, or nil if the
// shard does not exist. Indexes are cached like chunks.
func (a *Array) shardIndex(ctx context.Context, key string, sc []int) ([]byte, error) {
	var k cacheKey
	k.arr, k.index = a.id, true
	for i, c := range sc {
		k.c[i] = uint32(c)
	}
	sp := a.meta.shard
	idx, err := a.cache.get(k, func() ([]byte, error) {
		var raw []byte
		var err error
		if sp.indexAtEnd {
			raw, err = a.store.GetSuffix(ctx, key, int64(sp.indexSize))
		} else {
			raw, err = a.store.GetRange(ctx, key, 0, int64(sp.indexSize))
		}
		if errors.Is(err, ErrNotFound) {
			return []byte{}, nil // cache the absence
		}
		if err != nil {
			return nil, err
		}
		return sp.parseIndex(raw)
	})
	if err != nil {
		return nil, err
	}
	if len(idx) == 0 {
		return nil, nil
	}
	return idx, nil
}

// ---------------------------------------------------------------------- reads

// At reads a single element as float64. It touches exactly one chunk (and, for
// sharded arrays, the shard index), so it costs one cache lookup when the chunk
// is resident.
func (a *Array) At(ctx context.Context, idx ...int) (float64, error) {
	nd := len(a.meta.shape)
	if len(idx) != nd {
		return 0, fmt.Errorf("zarr: got %d indices for a %d-dimensional array", len(idx), nd)
	}
	var cc [maxDims]int
	off := 0
	for i, v := range idx {
		if v < 0 || v >= a.meta.shape[i] {
			return 0, fmt.Errorf("zarr: index %d out of range [0,%d) on axis %d", v, a.meta.shape[i], i)
		}
		cc[i] = v / a.leaf[i]
		off = off*a.leaf[i] + v%a.leaf[i]
	}
	b, err := a.leafBytes(ctx, cc[:nd])
	if err != nil {
		return 0, err
	}
	return loadFloat64(a.meta.dtype, b, off), nil
}

// Slice selects [Start, Stop) along one axis with a positive Step.
// Stop < 0 means "to the end of the axis"; Step 0 means 1.
type Slice struct{ Start, Stop, Step int }

// All selects a whole axis.
var All = Slice{0, -1, 1}

// Range selects [start, stop).
func Range(start, stop int) Slice { return Slice{start, stop, 1} }

// Index selects the single position i (the axis is kept, with length 1).
func Index(i int) Slice { return Slice{i, i + 1, 1} }

// Data is the result of a Read: a dense C-ordered block.
type Data struct {
	DType DType
	Shape []int
	// Bytes holds the elements in C order, little endian.
	Bytes []byte
}

// Len is the number of elements.
func (d *Data) Len() int { return prod(d.Shape) }

// Float64 converts the elements to float64.
func (d *Data) Float64() []float64 {
	out := make([]float64, d.Len())
	for i := range out {
		out[i] = loadFloat64(d.DType, d.Bytes, i)
	}
	return out
}

// Float32 converts the elements to float32.
func (d *Data) Float32() []float32 {
	out := make([]float32, d.Len())
	for i := range out {
		out[i] = float32(loadFloat64(d.DType, d.Bytes, i))
	}
	return out
}

type axisSel struct{ start, stop, step, n int }

func (a *Array) normalize(sel []Slice) ([]axisSel, error) {
	nd := len(a.meta.shape)
	if len(sel) > nd {
		return nil, fmt.Errorf("zarr: %d selectors for a %d-dimensional array", len(sel), nd)
	}
	out := make([]axisSel, nd)
	for i := 0; i < nd; i++ {
		s := All
		if i < len(sel) {
			s = sel[i]
		}
		if s.Step == 0 {
			s.Step = 1
		}
		if s.Stop < 0 || s.Stop > a.meta.shape[i] {
			if s.Stop > a.meta.shape[i] {
				return nil, fmt.Errorf("zarr: stop %d beyond axis %d length %d", s.Stop, i, a.meta.shape[i])
			}
			s.Stop = a.meta.shape[i]
		}
		if s.Start < 0 || s.Step < 0 || s.Start > s.Stop {
			return nil, fmt.Errorf("zarr: invalid slice %+v on axis %d", s, i)
		}
		out[i] = axisSel{s.Start, s.Stop, s.Step, (s.Stop - s.Start + s.Step - 1) / s.Step}
	}
	return out, nil
}

// Read fetches a hyper-rectangle. Axes without a selector are read in full.
// Only the chunks that intersect the selection are fetched, concurrently.
func (a *Array) Read(ctx context.Context, sel ...Slice) (*Data, error) {
	axes, err := a.normalize(sel)
	if err != nil {
		return nil, err
	}
	nd := len(axes)
	es := a.meta.dtype.Size()
	shape := make([]int, nd)
	for i, ax := range axes {
		shape[i] = ax.n
	}
	out := &Data{DType: a.meta.dtype, Shape: shape, Bytes: make([]byte, prod(shape)*es)}
	if len(out.Bytes) == 0 {
		return out, nil
	}
	// chunk range per axis
	lo := make([]int, nd)
	hi := make([]int, nd)
	for i, ax := range axes {
		lo[i] = ax.start / a.leaf[i]
		hi[i] = (ax.start+(ax.n-1)*ax.step)/a.leaf[i] + 1
	}
	var jobs [][]int
	cur := append([]int(nil), lo...)
	for {
		jobs = append(jobs, append([]int(nil), cur...))
		i := nd - 1
		for ; i >= 0; i-- {
			cur[i]++
			if cur[i] < hi[i] {
				break
			}
			cur[i] = lo[i]
		}
		if i < 0 {
			break
		}
	}
	if nd == 0 {
		jobs = [][]int{{}}
	}
	do := func(c []int) error {
		b, err := a.leafBytes(ctx, c)
		if err != nil {
			return err
		}
		a.copyOut(out, axes, c, b)
		return nil
	}
	if len(jobs) == 1 {
		return out, do(jobs[0])
	}
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	var wg sync.WaitGroup
	var once sync.Once
	var firstErr error
	sem := make(chan struct{}, a.par)
	for _, j := range jobs {
		sem <- struct{}{}
		wg.Add(1)
		go func(c []int) {
			defer wg.Done()
			defer func() { <-sem }()
			if ctx.Err() != nil {
				return
			}
			if err := do(c); err != nil {
				once.Do(func() { firstErr = err; cancel() })
			}
		}(j)
	}
	wg.Wait()
	return out, firstErr
}

// copyOut copies the part of leaf chunk c that the selection needs into out.
func (a *Array) copyOut(out *Data, axes []axisSel, c []int, chunk []byte) {
	nd := len(axes)
	es := a.meta.dtype.Size()
	if nd == 0 {
		copy(out.Bytes, chunk[:es])
		return
	}
	// per axis: first/last selected output index inside this chunk
	first := make([]int, nd) // output index range [first, last)
	last := make([]int, nd)
	for i, ax := range axes {
		cs, ce := c[i]*a.leaf[i], (c[i]+1)*a.leaf[i]
		f := 0
		if cs > ax.start {
			f = (cs - ax.start + ax.step - 1) / ax.step
		}
		l := (ce - ax.start + ax.step - 1) / ax.step // first index whose position >= ce
		if l > ax.n {
			l = ax.n
		}
		first[i], last[i] = f, l
		if f >= l {
			return
		}
	}
	cstr := make([]int, nd)
	ostr := make([]int, nd)
	ca, oa := 1, 1
	for i := nd - 1; i >= 0; i-- {
		cstr[i], ostr[i] = ca, oa
		ca *= a.leaf[i]
		oa *= out.Shape[i]
	}
	idx := append([]int(nil), first...)
	lastAx := nd - 1
	ax := axes[lastAx]
	for {
		co, oo := 0, 0
		for i := 0; i < lastAx; i++ {
			co += (axes[i].start + idx[i]*axes[i].step - c[i]*a.leaf[i]) * cstr[i]
			oo += idx[i] * ostr[i]
		}
		n := last[lastAx] - first[lastAx]
		cpos := ax.start + first[lastAx]*ax.step - c[lastAx]*a.leaf[lastAx]
		opos := first[lastAx]
		if ax.step == 1 {
			s := (co + cpos) * es
			d := (oo + opos) * es
			copy(out.Bytes[d:d+n*es], chunk[s:s+n*es])
		} else {
			for k := 0; k < n; k++ {
				s := (co + cpos + k*ax.step) * es
				d := (oo + opos + k) * es
				copy(out.Bytes[d:d+es], chunk[s:s+es])
			}
		}
		i := lastAx - 1
		for ; i >= 0; i-- {
			idx[i]++
			if idx[i] < last[i] {
				break
			}
			idx[i] = first[i]
		}
		if i < 0 {
			break
		}
	}
}

// Numeric is the set of types ReadAs can convert to.
type Numeric interface {
	~int8 | ~int16 | ~int32 | ~int64 | ~int | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~float32 | ~float64
}

// ReadAs reads a region and converts it to []T. Conversion goes through
// float64 for everything except 64-bit integers requested as themselves.
func ReadAs[T Numeric](ctx context.Context, a *Array, sel ...Slice) ([]T, []int, error) {
	d, err := a.Read(ctx, sel...)
	if err != nil {
		return nil, nil, err
	}
	n := d.Len()
	out := make([]T, n)
	switch d.DType {
	case Float32:
		for i := range out {
			out[i] = T(float32frombits(binary.LittleEndian.Uint32(d.Bytes[i*4:])))
		}
	case Float64:
		for i := range out {
			out[i] = T(float64frombits(binary.LittleEndian.Uint64(d.Bytes[i*8:])))
		}
	default:
		for i := range out {
			out[i] = T(loadFloat64(d.DType, d.Bytes, i))
		}
	}
	return out, d.Shape, nil
}
