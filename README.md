# gozarr

A pure-Go **reader** for [Zarr](https://zarr.dev) v2 and v3 that never loads a whole array.
Ask for one grid point and it fetches one chunk. For sharded arrays it fetches one *byte range*.

```go
store := zarr.NewHTTPStore("https://storage.googleapis.com/my-bucket/forecast.zarr")
a, _ := zarr.Open(ctx, store, "temperature_2m")

v, _ := a.At(ctx, 12, 340, 218)                                          // one element
d, _ := a.Read(ctx, zarr.Range(0, 24), zarr.Index(340), zarr.Index(218)) // a time series
```

## Why

Most Zarr tooling is Python. Services written in Go (APIs, workers, bots) that need a value at
a latitude/longitude end up shelling out to a sidecar. gozarr reads the format directly, with a
decoded-chunk cache sized for the "same few locations, again and again" access pattern.

## What it supports

| | |
|---|---|
| Formats | Zarr v2 and v3 arrays and groups, consolidated metadata (v2 `.zmetadata`, v3 inline) |
| Compression | Blosc (blosclz, lz4, lz4hc, zlib, zstd; byte and bit shuffle), zstd, gzip, zlib, bz2, lz4 |
| v3 codecs | `bytes` (both endians), `transpose`, `crc32c`, `sharding_indexed` (incl. nested) |
| v2 filters | shuffle, delta, fixedscaleoffset, astype, bitround, quantize |
| dtypes | bool, int8–64, uint8–64, float16/32/64. Strings, complex and structured types are not supported |
| Memory order | C and F, any endianness. Output is always C order, little endian |
| Stores | local directory, HTTP(S) with Range (GCS, S3, Azure, CDNs), in-memory. Implement `Store` (3 methods) for others |
| Chunk keys | v2 `.`/`/` separators; v3 `default` and `v2` encodings |
| Not supported | writing, irregular chunk grids, variable-length types, complex numbers |

During development every codec path was checked against 73 files written by the reference
`zarr-python`, including edge cases: partial edge chunks, missing chunks (fill value), NaN fills,
odd blosc block sizes, nested shards, big-endian data. Those files are not in this repository; the
interop tests skip when `testdata/` is absent. To check your own data, set `GOZARR_ARRAY`
(see Development).

## Speed

Apple M4, Go 1.24, local SSD, 96×128×40 float32 arrays. `go test -bench . -benchmem`

| Operation | Time | Allocs |
|---|---|---|
| `At` — chunk already decoded in the cache | **~80 ns** | 0 |
| `At` — same, 10 goroutines in parallel | ~63 ns | 0 |
| `At` — cold, sharded v3, zstd inner chunks (≈1 KiB read) | ~62 µs | 19 |
| `At` — cold, v2 zstd | ~34 µs | 11 |
| `At` — cold, 245 KiB Blosc-lz4 chunk | ~260 µs | 12 |
| `At` — cold, 245 KiB Blosc bit-shuffled chunk | ~475 µs | 12 |
| 10×10×40 window, hot | ~11 µs | 59 |
| `spatial.Nearest` (lat/lon → value), hot | ~150 ns | 2 |
| `spatial.Interp` (bilinear), hot | ~500 ns | 8 |

**What "sub-millisecond" means here.** Local and cached reads are well under a millisecond.
A cold read over the network is dominated by round-trip time (typically 10–50 ms to a cloud
bucket); no reader can beat that. What the library controls is *how many* round trips and *how
many bytes*: a point from a sharded array over HTTP is two requests (the shard index via a
suffix range, then the one inner chunk) and moved 8 KiB of a 433 KiB shard in the test suite
(`TestHTTPShardedPointReadFetchesOnlyWhatItNeeds`). After that, neighbouring points are cache hits.

Chunk layout is the other lever, and it's decided when the data is written: chunk along the axis
you read. A time series per location wants chunks long in time and short in space.

## Lat/lon lookups

```go
g, _ := zarr.OpenGroup(ctx, store, "")                    // consolidated metadata: no extra requests
f, _ := spatial.OpenField(ctx, g, "temperature_2m", "latitude", "longitude")

v, _ := f.Nearest(ctx, 17.38, 78.48, 12)                  // value at the nearest cell, time index 12
v, _ = f.Interp(ctx, 17.38, 78.48, 12)                    // bilinear (NaN cells are skipped)
s, _ := f.Series(ctx, 17.38, 78.48, 0, 0, 72)             // steps 0..72 along axis 0 at that point
```

Grids are rectilinear lat/lon (ascending or descending, 0–360 or −180–180 longitudes are both
accepted). Points up to half a cell outside the grid snap to the edge; further out gives
`spatial.ErrOutside`.

## Caching and concurrency

* Decoded chunks live in an LRU bounded by bytes (`zarr.NewCache(n)`; default 256 MiB, shared by
  all arrays unless you pass `WithCache`). Share one cache across arrays to cap total memory.
* Concurrent requests for the same missing chunk are coalesced into one fetch.
* Everything is safe for concurrent use. `Read` fetches the chunks it needs in parallel
  (`WithConcurrency`, default 16).
* Cached slices are shared; `At`/`Read` never hand them out, so callers can't corrupt the cache.

## Private buckets

`HTTPStore` takes any `*http.Client`, so request signing (SigV4, Google OAuth) is a `RoundTripper`.
A native GCS/S3 `Store` fits in about 40 lines and can live in your own package; the interface is
`Get`, `GetRange`, `GetSuffix`.

## Limits worth knowing

* Reads are exact; there is no resampling beyond bilinear in `spatial`.
* A single decoded chunk is capped at 1 GiB (guards against corrupt headers and decompression bombs).
* Arrays above 16 dimensions are rejected.
* Chunk and shard indices must fit in 32 bits.

## Development

```sh
make test race bench
# Check an array of your own: reads it whole and point by point and compares the two.
GOZARR_ARRAY=/path/to/store.zarr:array/path go test -run TestOwnArray -v .
```

## Licence

MIT.
