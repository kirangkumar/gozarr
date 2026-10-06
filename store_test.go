package zarr

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// A static file server (Go's own) already honours Range, like S3/GCS/CDNs do.
func rangeServer(t testing.TB) (*httptest.Server, *atomic.Int64, *atomic.Int64) {
	var reqs, bytesOut atomic.Int64
	fs := http.FileServer(http.Dir("testdata"))
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reqs.Add(1)
		cw := &countingWriter{ResponseWriter: w, n: &bytesOut}
		fs.ServeHTTP(cw, r)
	}))
	t.Cleanup(srv.Close)
	return srv, &reqs, &bytesOut
}

type countingWriter struct {
	http.ResponseWriter
	n *atomic.Int64
}

func (c *countingWriter) Write(b []byte) (int, error) {
	c.n.Add(int64(len(b)))
	return c.ResponseWriter.Write(b)
}

// The point of the library: one point from a sharded array over HTTP must move
// a small fraction of the shard.
func TestHTTPShardedPointReadFetchesOnlyWhatItNeeds(t *testing.T) {
	loadExpected(t)
	srv, reqs, out := rangeServer(t)
	ctx := context.Background()
	a, err := Open(ctx, NewHTTPStore(srv.URL), "v3_big_sharded.zarr", WithCache(NewCache(0)))
	if err != nil {
		t.Fatal(err)
	}
	reqs.Store(0)
	out.Store(0)
	v, err := a.At(ctx, 50, 70, 20)
	if err != nil {
		t.Fatal(err)
	}
	local, _ := Open(ctx, NewFSStore("testdata"), "v3_big_sharded.zarr")
	want, _ := local.At(ctx, 50, 70, 20)
	if v != want {
		t.Fatalf("got %v want %v", v, want)
	}
	if reqs.Load() != 2 { // index (suffix range) + inner chunk (range)
		t.Errorf("expected 2 requests, got %d", reqs.Load())
	}
	shard, _ := NewFSStore("testdata").Get(ctx, "v3_big_sharded.zarr/c/1/1/0")
	if out.Load() > int64(len(shard))/4 {
		t.Errorf("moved %d bytes for one point from a %d byte shard", out.Load(), len(shard))
	}
	t.Logf("point read moved %d of %d shard bytes in %d requests", out.Load(), len(shard), reqs.Load())

	// With a cache, the second read of a neighbouring point costs nothing.
	b, _ := Open(ctx, NewHTTPStore(srv.URL), "v3_big_sharded.zarr", WithCache(NewCache(32<<20)))
	b.At(ctx, 50, 70, 20)
	reqs.Store(0)
	b.At(ctx, 51, 71, 21)
	if reqs.Load() != 0 {
		t.Errorf("neighbouring point issued %d requests", reqs.Load())
	}
}

func TestHTTPFullRead(t *testing.T) {
	exp := loadExpected(t)
	srv, _, _ := rangeServer(t)
	ctx := context.Background()
	for _, name := range []string{"v2_big_blosc_lz4.zarr", "v2_slash.zarr", "v3_sharded.zarr"} {
		a, err := Open(ctx, NewHTTPStore(srv.URL), name)
		if err != nil {
			t.Fatal(name, err)
		}
		d, err := a.Read(ctx)
		if err != nil {
			t.Fatal(name, err)
		}
		for i, v := range d.Float64() {
			if !same(v, exp[name].Flat[i]) {
				t.Fatalf("%s elem %d", name, i)
			}
		}
	}
	// store from URL string
	if _, err := NewStore("gs://bucket/path"); err != nil {
		t.Fatal(err)
	}
}

func TestMissingArray(t *testing.T) {
	_, err := Open(context.Background(), NewFSStore("testdata"), "nope.zarr")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("got %v", err)
	}
	if _, err := Open(context.Background(), NewFSStore("testdata"), "v3_group.zarr"); err == nil || !strings.Contains(err.Error(), "group") {
		t.Fatalf("opening a group as an array: %v", err)
	}
}

func TestBoundsAndSelectors(t *testing.T) {
	loadExpected(t)
	ctx := context.Background()
	a, _ := Open(ctx, NewFSStore("testdata"), "v2_zstd.zarr")
	if _, err := a.At(ctx, 40, 0, 0); err == nil {
		t.Error("expected out-of-range error")
	}
	if _, err := a.At(ctx, -1, 0, 0); err == nil {
		t.Error("expected negative index error")
	}
	if _, err := a.At(ctx, 1, 2); err == nil {
		t.Error("expected rank error")
	}
	if _, err := a.Read(ctx, Range(0, 41)); err == nil {
		t.Error("expected stop error")
	}
	if _, err := a.Read(ctx, Slice{5, 2, 1}); err == nil {
		t.Error("expected start>stop error")
	}
	d, err := a.Read(ctx, Range(3, 3))
	if err != nil || d.Len() != 0 {
		t.Errorf("empty slice: %v %v", d, err)
	}
}

// Many goroutines asking for the same cold chunk must cause one fetch.
func TestSingleFlight(t *testing.T) {
	loadExpected(t)
	srv, reqs, _ := rangeServer(t)
	ctx := context.Background()
	a, err := Open(ctx, NewHTTPStore(srv.URL), "v2_zstd.zarr", WithCache(NewCache(8<<20)))
	if err != nil {
		t.Fatal(err)
	}
	reqs.Store(0)
	var wg sync.WaitGroup
	for i := 0; i < 64; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := a.At(ctx, 1, 2, 3); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	if reqs.Load() != 1 {
		t.Errorf("64 concurrent reads of one chunk made %d requests", reqs.Load())
	}
}

func TestCacheEvicts(t *testing.T) {
	loadExpected(t)
	ctx := context.Background()
	c := NewCache(64 << 10)
	a, _ := Open(ctx, NewFSStore("testdata"), "v2_big_blosc_lz4.zarr", WithCache(c))
	if _, err := a.Read(ctx); err != nil { // 1.5 MB of chunks through a 64 KiB cache
		t.Fatal(err)
	}
	if c.Len() > 64<<10 {
		t.Errorf("cache holds %d bytes, limit %d", c.Len(), 64<<10)
	}
}

func TestCorruptChunkIsAnErrorNotAPanic(t *testing.T) {
	ctx := context.Background()
	for _, name := range []string{"v2_blosc_lz4.zarr", "v2_big_blosclz.zarr", "v2_zstd.zarr", "v2_gzip.zarr", "v3_zstd_crc.zarr", "v2_odd_blosc.zarr"} {
		raw, err := NewFSStore("testdata").Get(ctx, name+"/.zarray")
		if err != nil {
			raw, err = NewFSStore("testdata").Get(ctx, name+"/zarr.json")
		}
		if err != nil {
			t.Skip("fixtures missing")
		}
		ms := NewMemoryStore()
		ms.Set(".zarray", raw)
		ms.Set("zarr.json", raw)
		if strings.HasPrefix(name, "v3") {
			ms = NewMemoryStore()
			ms.Set("zarr.json", raw)
		} else {
			ms = NewMemoryStore()
			ms.Set(".zarray", raw)
		}
		good, _ := NewFSStore("testdata").Get(ctx, name+"/"+firstChunk(name))
		for trial := 0; trial < 200; trial++ {
			bad := append([]byte(nil), good...)
			switch trial % 3 {
			case 0:
				bad = bad[:trial%len(bad)]
			case 1:
				bad[(trial*7)%len(bad)] ^= byte(trial) | 1
			case 2:
				bad[(trial*13)%len(bad)] = 0xff
				bad[(trial*5)%len(bad)] = 0
			}
			ms.Set(firstChunk(name), bad)
			a, err := Open(ctx, ms, "", WithCache(NewCache(0)))
			if err != nil {
				t.Fatal(err)
			}
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("%s trial %d panicked: %v", name, trial, r)
					}
				}()
				a.At(ctx, 0, 0, 0)
			}()
		}
	}
}

func firstChunk(name string) string {
	if strings.HasPrefix(name, "v3") {
		return "c/0/0/0"
	}
	if strings.Contains(name, "odd") {
		return "0.0"
	}
	return "0.0.0"
}
