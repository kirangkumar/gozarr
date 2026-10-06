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
		t.Skip("fixtures missing; run: python internal/gen/gen.py (needs zarr, numcodecs, numpy): ", expErr)
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
