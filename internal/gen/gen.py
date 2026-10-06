"""Generates interop fixtures with the reference zarr-python. Run: python internal/gen/gen.py"""
import json, shutil, os, numpy as np, zarr, numcodecs
from zarr.codecs import BytesCodec, GzipCodec, ZstdCodec, BloscCodec, TransposeCodec, Crc32cCodec

root = os.path.join(os.path.dirname(__file__), "..", "..", "testdata")
shutil.rmtree(root, ignore_errors=True); os.makedirs(root)
expect = {}
rng = np.random.default_rng(7)

def rec(name, arr):
    expect[name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "flat": arr.astype("float64").ravel().tolist()}

# ---- v2 ----
def v2(name, data, chunks, compressor, order="C", filters=None, fill=0, dtype=None):
    p = os.path.join(root, name)
    z = zarr.create_array(p, shape=data.shape, chunks=chunks, dtype=data.dtype, zarr_format=2,
                          compressors=compressor, order=order, filters=filters, fill_value=fill, overwrite=True)
    z[...] = data
    rec(name, data)

f32 = (rng.random((40, 50, 30)) * 100).astype("float32")
v2("v2_blosc_lz4.zarr", f32, (10, 16, 8), numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE))
v2("v2_blosc_zstd_bit.zarr", f32, (10, 16, 8), numcodecs.Blosc(cname="zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE))
v2("v2_blosc_noshuf.zarr", f32, (10, 16, 8), numcodecs.Blosc(cname="lz4hc", clevel=5, shuffle=numcodecs.Blosc.NOSHUFFLE))
v2("v2_blosc_zlib.zarr", f32, (10, 16, 8), numcodecs.Blosc(cname="zlib", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE))
v2("v2_zstd.zarr", f32, (10, 16, 8), numcodecs.Zstd(level=3))
v2("v2_gzip.zarr", f32, (10, 16, 8), numcodecs.GZip(level=4))
v2("v2_zlib.zarr", f32, (10, 16, 8), numcodecs.Zlib(level=4))
v2("v2_none.zarr", f32, (10, 16, 8), None)
v2("v2_forder.zarr", f32, (10, 16, 8), numcodecs.Zstd(level=3), order="F")
v2("v2_shuffle_filter.zarr", f32, (10, 16, 8), numcodecs.Zlib(level=1), filters=[numcodecs.Shuffle(elementsize=4)])
v2("v2_delta_filter.zarr", f32.astype("int32"), (10, 16, 8), numcodecs.Zlib(level=1), filters=[numcodecs.Delta(dtype="<i4")])
for dt in ["int8","int16","int32","int64","uint8","uint16","uint32","uint64","float16","float64","bool"]:
    d = (rng.random((17, 23)) * 100).astype(dt)
    v2(f"v2_dtype_{dt}.zarr", d, (5, 7), numcodecs.Blosc(cname="lz4"))
v2("v2_be.zarr", (rng.random((9,9))*1000).astype(">f8"), (4,4), numcodecs.Zstd())
v2("v2_i2_be.zarr", (rng.random((9,9))*1000).astype(">i2"), (4,4), numcodecs.Zstd())
sparse = np.full((30, 30), -9999.0, dtype="float32"); sparse[:10, :10] = rng.random((10,10)).astype("float32")
v2("v2_fill.zarr", sparse, (10, 10), numcodecs.Zstd(), fill=-9999.0)   # empty chunks are never written
v2("v2_scalar_1d.zarr", np.arange(1000, dtype="float64"), (128,), numcodecs.Blosc())
v2("v2_nan_fill.zarr", np.where(rng.random((12,12))>.5, np.nan, 1.5).astype("float32"), (5,5), numcodecs.Zstd(), fill=float("nan"))

# v2 hierarchy + consolidated metadata with attributes
g = zarr.open_group(os.path.join(root, "v2_group.zarr"), mode="w", zarr_format=2)
g.attrs["title"] = "grid"
lat = g.create_array("lat", shape=(40,), chunks=(40,), dtype="float64"); lat[:] = np.linspace(37.5, 6.5, 40); lat.attrs["_ARRAY_DIMENSIONS"] = ["lat"]
lon = g.create_array("lon", shape=(50,), chunks=(50,), dtype="float64"); lon[:] = np.linspace(68.0, 97.5, 50); lon.attrs["_ARRAY_DIMENSIONS"] = ["lon"]
t = g.create_array("temp", shape=(30, 40, 50), chunks=(10, 20, 25), dtype="float32", fill_value=float("nan"), compressors=numcodecs.Blosc(cname="zstd"))
temp = (rng.random((30, 40, 50)) * 40).astype("float32"); t[:] = temp
t.attrs["_ARRAY_DIMENSIONS"] = ["time", "lat", "lon"]; t.attrs["scale_factor"] = 1.0
zarr.consolidate_metadata(os.path.join(root, "v2_group.zarr"))
rec("v2_group.zarr/temp", temp); rec("v2_group.zarr/lat", np.linspace(37.5, 6.5, 40)); rec("v2_group.zarr/lon", np.linspace(68.0, 97.5, 50))

# ---- v3 ----
def v3(name, data, chunks, codecs=None, shards=None, fill=0, dims=None, compressors="default", serializer="auto", filters=None):
    p = os.path.join(root, name)
    kw = dict(shape=data.shape, chunks=chunks, dtype=data.dtype, zarr_format=3, fill_value=fill, overwrite=True)
    if shards: kw["shards"] = shards
    if compressors != "default": kw["compressors"] = compressors
    if filters is not None: kw["filters"] = filters
    if serializer != "auto": kw["serializer"] = serializer
    if dims: kw["dimension_names"] = dims
    z = zarr.create_array(p, **kw); z[...] = data
    rec(name, data)

v3("v3_default.zarr", f32, (10, 16, 8))
v3("v3_zstd_crc.zarr", f32, (10, 16, 8), compressors=[ZstdCodec(level=3, checksum=True)])
v3("v3_gzip.zarr", f32, (10, 16, 8), compressors=[GzipCodec(level=4)])
v3("v3_blosc.zarr", f32, (10, 16, 8), compressors=[BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")])
v3("v3_blosc_zstd.zarr", f32, (10, 16, 8), compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="shuffle")])
v3("v3_none.zarr", f32, (10, 16, 8), compressors=None)
v3("v3_transpose.zarr", f32, (10, 16, 8), filters=[TransposeCodec(order=(2, 0, 1))], compressors=[ZstdCodec()])
v3("v3_big_endian.zarr", (rng.random((9,9))*1000).astype("float64"), (4,4), serializer=BytesCodec(endian="big"), compressors=[ZstdCodec()])
v3("v3_sharded.zarr", f32, (5, 8, 4), shards=(10, 16, 8))
v3("v3_sharded_zstd.zarr", f32, (5, 8, 4), shards=(20, 32, 16), compressors=[ZstdCodec()])
v3("v3_sharded_nocomp.zarr", f32, (5, 8, 4), shards=(10, 16, 8), compressors=None)
v3("v3_sharded_blosc.zarr", f32, (5, 8, 4), shards=(10, 16, 8), compressors=[BloscCodec()])
v3("v3_fill.zarr", sparse, (10, 10), fill=-9999.0)
v3("v3_sharded_fill.zarr", sparse, (5, 5), shards=(10, 10), fill=-9999.0)
for dt in ["int8","int16","int32","int64","uint8","uint16","uint32","uint64","float16","float32","float64","bool"]:
    d = (rng.random((17, 23)) * 100).astype(dt)
    v3(f"v3_dtype_{dt}.zarr", d, (5, 7))
v3("v3_dims.zarr", f32, (10, 16, 8), dims=["time", "lat", "lon"])

g = zarr.open_group(os.path.join(root, "v3_group.zarr"), mode="w", zarr_format=3)
g.attrs["title"] = "grid3"
lat = g.create_array("lat", shape=(40,), chunks=(40,), dtype="float64", dimension_names=["lat"]); lat[:] = np.linspace(37.5, 6.5, 40)
lon = g.create_array("lon", shape=(50,), chunks=(50,), dtype="float64", dimension_names=["lon"]); lon[:] = np.linspace(68.0, 97.5, 50)
t = g.create_array("temp", shape=(30, 40, 50), chunks=(1, 10, 10), shards=(30, 40, 50), dtype="float32",
                   dimension_names=["time", "lat", "lon"], fill_value=float("nan")); t[:] = temp
zarr.consolidate_metadata(os.path.join(root, "v3_group.zarr"))
rec("v3_group.zarr/temp", temp); rec("v3_group.zarr/lat", np.linspace(37.5, 6.5, 40)); rec("v3_group.zarr/lon", np.linspace(68.0, 97.5, 50))


# ---- larger / multi-block blosc, blosclz, '/' separator, 0-d, slicing cases ----
big = (rng.random((96, 128, 40)) * 1000).astype("float32")
v2("v2_big_blosc_lz4.zarr", big, (48, 64, 20), numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE))
v2("v2_big_blosclz.zarr", big, (48, 64, 20), numcodecs.Blosc(cname="blosclz", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE))
v2("v2_big_blosclz_noshuf.zarr", big, (48, 64, 20), numcodecs.Blosc(cname="blosclz", clevel=9, shuffle=numcodecs.Blosc.NOSHUFFLE))
v2("v2_big_blosc_bit.zarr", big, (48, 64, 20), numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE))
v2("v2_big_blosc_f64.zarr", big.astype("float64"), (48, 64, 20), numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE))
v2("v2_big_blosc_i16.zarr", (big).astype("int16"), (48, 64, 20), numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE))
smooth = np.cumsum(np.cumsum(rng.random((60, 70)), 0), 1).astype("float32")  # compressible
v2("v2_smooth_blosclz.zarr", smooth, (30, 35), numcodecs.Blosc(cname="blosclz", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE))
v2("v2_odd_blosc.zarr", (rng.random((37, 53)) * 10).astype("float32"), (13, 17), numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE))
v2("v2_blosc_bytes.zarr", (rng.random((101,)) * 255).astype("uint8"), (33,), numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE))
# '/' dimension separator
p = os.path.join(root, "v2_slash.zarr")
z = zarr.create_array(p, shape=big.shape, chunks=(48, 64, 20), dtype="float32", zarr_format=2, compressors=numcodecs.Zstd(), chunk_key_encoding={"name": "v2", "separator": "/"}, overwrite=True)
z[...] = big; rec("v2_slash.zarr", big)
# 0-d and 1-d
v3("v3_zero_d.zarr", np.array(3.25, dtype="float64"), (), compressors=None) if False else None
v3("v3_big_sharded.zarr", big, (16, 16, 8), shards=(48, 64, 40), compressors=[ZstdCodec()])
v3("v3_big_blosc.zarr", big, (48, 64, 20), compressors=[BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")])

# nested sharding: shard -> 16x16 chunks -> each holding 4x4 sub-chunks
from zarr.codecs import ShardingCodec
nested = (rng.random((32, 32)) * 100).astype("float32")
pn = os.path.join(root, "v3_nested_shard.zarr")
zn = zarr.create_array(pn, shape=nested.shape, dtype="float32", chunks=(32, 32), zarr_format=3, compressors=None,
    serializer=ShardingCodec(chunk_shape=(16, 16), codecs=[ShardingCodec(chunk_shape=(4, 4), codecs=[BytesCodec(), ZstdCodec()])]), overwrite=True)
zn[...] = nested; rec("v3_nested_shard.zarr", nested)

cases = []
def case(name, arr, sel):
    sl = tuple(slice(a, b, c) for a, b, c in sel)
    sub = arr[sl]
    cases.append({"name": name, "sel": sel, "shape": list(sub.shape), "flat": sub.astype("float64").ravel().tolist()})
for nm in ["v2_big_blosc_lz4.zarr", "v3_big_sharded.zarr", "v3_sharded.zarr", "v2_forder.zarr"]:
    arr = big if "big" in nm else f32
    shp = arr.shape
    case(nm, arr, [[0, shp[0], 1], [0, shp[1], 1], [0, shp[2], 1]])
    case(nm, arr, [[3, 17, 1], [5, 50, 1], [2, 29, 1]])
    case(nm, arr, [[0, shp[0], 3], [1, shp[1], 5], [0, shp[2], 2]])
    case(nm, arr, [[7, 8, 1], [0, shp[1], 1], [0, shp[2], 1]])
    case(nm, arr, [[11, 40, 7], [13, 14, 1], [3, shp[2], 4]])
    case(nm, arr, [[0, min(shp[0], 33), 2], [0, min(shp[1], 50), 1], [1, shp[2], 1]])
json.dump(cases, open(os.path.join(root, "cases.json"), "w"))

def clean(x):
    return [None if (isinstance(v, float) and v != v) else v for v in x]
for k in expect: expect[k]["flat"] = clean(expect[k]["flat"])
json.dump(expect, open(os.path.join(root, "expected.json"), "w"))
print("fixtures:", len(expect))
