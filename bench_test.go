package zarr

import (
	"context"
	"testing"
)

func benchArray(b *testing.B, name string, cacheBytes int64) *Array {
	if _, err := NewFSStore("testdata").Get(context.Background(), name+"/zarr.json"); err != nil {
		if _, err2 := NewFSStore("testdata").Get(context.Background(), name+"/.zarray"); err2 != nil {
			b.Skip("fixtures missing")
		}
	}
	a, err := Open(context.Background(), NewFSStore("testdata"), name, WithCache(NewCache(cacheBytes)))
	if err != nil {
		b.Fatal(err)
	}
	return a
}

// Point read when the chunk is already decoded in memory.
func BenchmarkAtHot(b *testing.B) {
	ctx := context.Background()
	for _, name := range []string{"v2_big_blosc_lz4.zarr", "v3_big_sharded.zarr"} {
		b.Run(name, func(b *testing.B) {
			a := benchArray(b, name, 64<<20)
			a.At(ctx, 50, 70, 20)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				a.At(ctx, 50, 70, 20)
			}
		})
	}
}

func BenchmarkAtHotParallel(b *testing.B) {
	ctx := context.Background()
	a := benchArray(b, "v3_big_sharded.zarr", 64<<20)
	a.Read(ctx)
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			a.At(ctx, i%96, (i*7)%128, (i*3)%40)
			i++
		}
	})
}

// Point read with no cache: fetch from the local file system and decode every time.
func BenchmarkAtCold(b *testing.B) {
	ctx := context.Background()
	for _, name := range []string{"v2_big_blosc_lz4.zarr", "v2_big_blosc_bit.zarr", "v2_zstd.zarr", "v3_big_sharded.zarr", "v3_big_blosc.zarr"} {
		b.Run(name, func(b *testing.B) {
			a := benchArray(b, name, 0)
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				a.At(ctx, 5+i%3, 9, 3)
			}
		})
	}
}

func BenchmarkReadWindow(b *testing.B) {
	ctx := context.Background()
	a := benchArray(b, "v3_big_sharded.zarr", 64<<20)
	a.Read(ctx, Range(0, 10), Range(0, 10), Range(0, 40))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		a.Read(ctx, Range(0, 10), Range(0, 10), Range(0, 40))
	}
}
