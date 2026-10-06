package spatial

import (
	"context"
	"testing"

	zarr "github.com/kirangkumar/gozarr"
)

func BenchmarkNearestHot(b *testing.B) {
	ctx := context.Background()
	g, err := zarr.OpenGroup(ctx, zarr.NewFSStore("../testdata"), "v3_group.zarr")
	if err != nil {
		b.Skip("fixtures missing")
	}
	f, err := OpenField(ctx, g, "temp", "lat", "lon", zarr.WithCache(zarr.NewCache(64<<20)))
	if err != nil {
		b.Fatal(err)
	}
	f.Nearest(ctx, 20.5, 80.2, 3)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Nearest(ctx, 20.5, 80.2, 3)
	}
}

func BenchmarkInterpHot(b *testing.B) {
	ctx := context.Background()
	g, err := zarr.OpenGroup(ctx, zarr.NewFSStore("../testdata"), "v3_group.zarr")
	if err != nil {
		b.Skip("fixtures missing")
	}
	f, _ := OpenField(ctx, g, "temp", "lat", "lon", zarr.WithCache(zarr.NewCache(64<<20)))
	f.Interp(ctx, 20.5, 80.2, 3)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Interp(ctx, 20.5, 80.2, 3)
	}
}
