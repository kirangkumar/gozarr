// Package zarr reads Zarr v2 and v3 arrays without loading them into memory.
//
// Zarr stores an array as many small compressed chunks. This package opens an
// array by reading its metadata only, and fetches, decodes and caches exactly
// the chunks a read touches. For sharded v3 arrays it goes one level further:
// it reads the shard's index, then a byte range for the one inner chunk it
// needs, so a single point costs a few KiB of I/O however large the shard is.
//
//	store := zarr.NewHTTPStore("https://storage.googleapis.com/my-bucket/forecast.zarr")
//	a, _ := zarr.Open(ctx, store, "temperature_2m")
//	v, _ := a.At(ctx, 12, 340, 218)                       // one element
//	d, _ := a.Read(ctx, zarr.Range(0, 24), zarr.Index(340), zarr.Index(218)) // a time series
//
// Reading only; there is no writer. Decoding is pure Go (no cgo), including
// Blosc, so the library cross-compiles and has two small dependencies.
package zarr
