package zarr

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

type expectation struct {
	Shape []int      `json:"shape"`
	Dtype string     `json:"dtype"`
	Flat  []*float64 `json:"flat"`
}

var (
	expOnce sync.Once
	expData map[string]expectation
	expErr  error
)

func loadExpected(t testing.TB) map[string]expectation {
	expOnce.Do(func() {
		b, err := os.ReadFile("testdata/expected.json")
		if err != nil {
			expErr = err
			return
		}
		expErr = json.Unmarshal(b, &expData)
	})
	if expErr != nil {
		t.Skip("interop fixtures not present (testdata/): ", expErr)
	}
	return expData
}

func same(got float64, want *float64) bool {
	if want == nil {
		return math.IsNaN(got)
	}
	return got == *want || (math.Abs(got-*want) <= 1e-12*math.Max(1, math.Abs(*want)))
}

// Every fixture written by the reference implementation must read back
// exactly, through full reads and through point reads.
func TestInteropFixtures(t *testing.T) {
	exp := loadExpected(t)
	ctx := context.Background()
	store := NewFSStore("testdata")
	for name, e := range exp {
		name, e := name, e
		t.Run(name, func(t *testing.T) {
			path := name
			var a *Array
			var err error
			if i := strings.Index(name, ".zarr/"); i >= 0 {
				var g *Group
				g, err = OpenGroup(ctx, store, name[:i+5])
				if err != nil {
					t.Fatal(err)
				}
				a, err = g.Array(ctx, name[i+6:], WithCache(NewCache(1<<20)))
			} else {
				a, err = Open(ctx, store, path, WithCache(NewCache(1<<20)))
			}
			if err != nil {
				t.Fatal(err)
			}
			if got := a.Shape(); !equalInts(got, e.Shape) {
				t.Fatalf("shape %v want %v", got, e.Shape)
			}
			d, err := a.Read(ctx)
			if err != nil {
				t.Fatal(err)
			}
			vals := d.Float64()
			if len(vals) != len(e.Flat) {
				t.Fatalf("len %d want %d", len(vals), len(e.Flat))
			}
			for i, w := range e.Flat {
				if !same(vals[i], w) {
					t.Fatalf("elem %d: got %v want %v", i, vals[i], deref(w))
				}
			}
			// point reads at the corners and a few interior points
			for _, flat := range []int{0, len(e.Flat) - 1, len(e.Flat) / 2, len(e.Flat) / 3} {
				idx := unravel(flat, e.Shape)
				v, err := a.At(ctx, idx...)
				if err != nil {
					t.Fatal(err)
				}
				if !same(v, e.Flat[flat]) {
					t.Fatalf("At%v = %v want %v", idx, v, deref(e.Flat[flat]))
				}
			}
		})
	}
}

func deref(p *float64) any {
	if p == nil {
		return "NaN"
	}
	return *p
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func unravel(flat int, shape []int) []int {
	idx := make([]int, len(shape))
	for i := len(shape) - 1; i >= 0; i-- {
		idx[i] = flat % shape[i]
		flat /= shape[i]
	}
	return idx
}

var _ = filepath.Join

func TestSlices(t *testing.T) {
	loadExpected(t)
	b, err := os.ReadFile("testdata/cases.json")
	if err != nil {
		t.Skip("no cases.json")
	}
	var cases []struct {
		Name  string     `json:"name"`
		Sel   [][3]int   `json:"sel"`
		Shape []int      `json:"shape"`
		Flat  []*float64 `json:"flat"`
	}
	if err := json.Unmarshal(b, &cases); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	store := NewFSStore("testdata")
	for i, c := range cases {
		a, err := Open(ctx, store, c.Name, WithCache(NewCache(64<<20)))
		if err != nil {
			t.Fatal(err)
		}
		sel := make([]Slice, len(c.Sel))
		for k, s := range c.Sel {
			sel[k] = Slice{s[0], s[1], s[2]}
		}
		d, err := a.Read(ctx, sel...)
		if err != nil {
			t.Fatalf("%d %s: %v", i, c.Name, err)
		}
		if !equalInts(d.Shape, c.Shape) {
			t.Fatalf("%d %s %v: shape %v want %v", i, c.Name, c.Sel, d.Shape, c.Shape)
		}
		for j, v := range d.Float64() {
			if !same(v, c.Flat[j]) {
				t.Fatalf("%d %s %v: elem %d got %v want %v", i, c.Name, c.Sel, j, v, deref(c.Flat[j]))
			}
		}
	}
}

// TestOwnArray is a self-consistency check for data you supply:
//
//	GOZARR_ARRAY=/path/to/store.zarr:path/to/array go test -run TestOwnArray -v .
//
// It reads the array in full (capped at 64 Mi elements), then re-reads random
// points and windows through the point and slice paths and requires identical values.
func TestOwnArray(t *testing.T) {
	spec := os.Getenv("GOZARR_ARRAY")
	if spec == "" {
		t.Skip("set GOZARR_ARRAY=<store>[:<array path>]")
	}
	loc, path, _ := strings.Cut(spec, ":")
	if strings.HasPrefix(loc, "http") { // URLs contain ':'; split on the last one after the scheme
		i := strings.LastIndex(spec, ":")
		if i > strings.Index(spec, "://")+2 {
			loc, path = spec[:i], spec[i+1:]
		} else {
			loc, path = spec, ""
		}
	}
	store, err := NewStore(loc)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	a, err := Open(ctx, store, path, WithCache(NewCache(256<<20)))
	if err != nil {
		t.Fatal(err)
	}
	shape := a.Shape()
	t.Logf("v%d %v %s chunks=%v", a.Version(), shape, a.DType(), a.ChunkShape())
	n := prod(shape)
	if n > 64<<20 {
		t.Skipf("%d elements is too many for a full read", n)
	}
	full, err := a.Read(ctx)
	if err != nil {
		t.Fatal(err)
	}
	vals := full.Float64()
	rng := uint64(88172645463325252)
	next := func(m int) int {
		rng ^= rng << 13
		rng ^= rng >> 7
		rng ^= rng << 17
		return int(rng % uint64(m))
	}
	for k := 0; k < 2000; k++ {
		idx := make([]int, len(shape))
		flat := 0
		for i := range shape {
			idx[i] = next(shape[i])
			flat = flat*shape[i] + idx[i]
		}
		v, err := a.At(ctx, idx...)
		if err != nil {
			t.Fatal(err)
		}
		if v != vals[flat] && !(math.IsNaN(v) && math.IsNaN(vals[flat])) {
			t.Fatalf("At%v = %v but full read has %v", idx, v, vals[flat])
		}
	}
	for k := 0; k < 50; k++ {
		sel := make([]Slice, len(shape))
		for i := range shape {
			lo := next(shape[i])
			hi := lo + 1 + next(shape[i]-lo)
			sel[i] = Slice{lo, hi, 1 + next(3)}
		}
		d, err := a.Read(ctx, sel...)
		if err != nil {
			t.Fatal(err)
		}
		got := d.Float64()
		idx := make([]int, len(shape))
		for j, v := range got {
			rem := j
			flat := 0
			for i := len(shape) - 1; i >= 0; i-- {
				idx[i] = sel[i].Start + (rem%d.Shape[i])*sel[i].Step
				rem /= d.Shape[i]
			}
			for i := range shape {
				flat = flat*shape[i] + idx[i]
			}
			if v != vals[flat] && !(math.IsNaN(v) && math.IsNaN(vals[flat])) {
				t.Fatalf("window %+v element %v = %v, full read has %v", sel, idx, v, vals[flat])
			}
		}
	}
}
