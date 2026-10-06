package zarr

import (
	"bytes"
	"compress/bzip2"
	"compress/gzip"
	"compress/zlib"
	"encoding/binary"
	"fmt"
	"hash/crc32"
	"io"
	"math"
)

// step is one stage of a decode pipeline: bytes in, bytes out. It may return
// its input unchanged, and must not modify it unless it owns it.
type step func([]byte) ([]byte, error)

var crcTable = crc32.MakeTable(crc32.Castagnoli)

func gzipStep(in []byte) ([]byte, error) {
	r, err := gzip.NewReader(bytes.NewReader(in))
	if err != nil {
		return nil, err
	}
	defer r.Close()
	return readLimited(r)
}

func zlibStep(in []byte) ([]byte, error) {
	r, err := zlib.NewReader(bytes.NewReader(in))
	if err != nil {
		return nil, err
	}
	defer r.Close()
	return readLimited(r)
}

func bz2Step(in []byte) ([]byte, error) { return readLimited(bzip2.NewReader(bytes.NewReader(in))) }

func readLimited(r io.Reader) ([]byte, error) {
	out, err := io.ReadAll(io.LimitReader(r, maxDecoded+1))
	if err == nil && len(out) > maxDecoded {
		return nil, fmt.Errorf("zarr: decoded chunk exceeds %d bytes", maxDecoded)
	}
	return out, err
}

func zstdStep(in []byte) ([]byte, error) { return zstdDec.DecodeAll(in, nil) }

func bloscStep(in []byte) ([]byte, error) { return bloscDecode(in) }

// lz4Step is numcodecs' LZ4: a 4 byte little endian size followed by a block.
func lz4Step(in []byte) ([]byte, error) {
	if len(in) < 4 {
		return nil, fmt.Errorf("zarr: short lz4 chunk")
	}
	n := int(binary.LittleEndian.Uint32(in))
	if n > maxDecoded {
		return nil, fmt.Errorf("zarr: lz4 chunk too large")
	}
	out := make([]byte, n)
	if err := bloscDecompress(1, in[4:], out); err != nil {
		return nil, err
	}
	return out, nil
}

func crc32cStep(in []byte) ([]byte, error) {
	if len(in) < 4 {
		return nil, fmt.Errorf("zarr: short crc32c chunk")
	}
	body := in[:len(in)-4]
	if crc32.Checksum(body, crcTable) != binary.LittleEndian.Uint32(in[len(in)-4:]) {
		return nil, fmt.Errorf("zarr: crc32c mismatch")
	}
	return body, nil
}

func endianStep(size int) step {
	return func(in []byte) ([]byte, error) {
		out := append([]byte(nil), in...)
		byteswap(out, size)
		return out, nil
	}
}

func shuffleStep(elemsize int) step {
	return func(in []byte) ([]byte, error) {
		if elemsize <= 1 {
			return in, nil
		}
		out := make([]byte, len(in))
		byteUnshuffle(out, in, elemsize)
		return out, nil
	}
}

// deltaStep undoes numcodecs.Delta: a running sum over the elements.
func deltaStep(dt DType, be bool) step {
	return func(in []byte) ([]byte, error) {
		out := append([]byte(nil), in...)
		sz := dt.Size()
		if be {
			byteswap(out, sz)
		}
		n := len(out) / sz
		switch dt {
		case Float32:
			var acc float32
			for i := 0; i < n; i++ {
				acc += math.Float32frombits(binary.LittleEndian.Uint32(out[i*4:]))
				binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(acc))
			}
		case Float64:
			var acc float64
			for i := 0; i < n; i++ {
				acc += math.Float64frombits(binary.LittleEndian.Uint64(out[i*8:]))
				binary.LittleEndian.PutUint64(out[i*8:], math.Float64bits(acc))
			}
		default:
			var acc uint64
			for i := 0; i < n; i++ {
				var v uint64
				switch sz {
				case 1:
					v = uint64(out[i])
				case 2:
					v = uint64(binary.LittleEndian.Uint16(out[i*2:]))
				case 4:
					v = uint64(binary.LittleEndian.Uint32(out[i*4:]))
				default:
					v = binary.LittleEndian.Uint64(out[i*8:])
				}
				acc += v
				switch sz {
				case 1:
					out[i] = byte(acc)
				case 2:
					binary.LittleEndian.PutUint16(out[i*2:], uint16(acc))
				case 4:
					binary.LittleEndian.PutUint32(out[i*4:], uint32(acc))
				default:
					binary.LittleEndian.PutUint64(out[i*8:], acc)
				}
			}
		}
		if be {
			byteswap(out, sz)
		}
		return out, nil
	}
}

// convertStep converts every element between two dtypes (numcodecs AsType and
// the decode half of FixedScaleOffset). Byte order of the source/target is
// given by the *Big flags.
func convertStep(from DType, fromBig bool, to DType, toBig bool, f func(float64) float64) step {
	return func(in []byte) ([]byte, error) {
		src := in
		if fromBig {
			src = append([]byte(nil), in...)
			byteswap(src, from.Size())
		}
		n := len(src) / from.Size()
		out := make([]byte, n*to.Size())
		for i := 0; i < n; i++ {
			v := loadFloat64(from, src, i)
			if f != nil {
				v = f(v)
			}
			storeFloat64(to, out, i, v)
		}
		if toBig {
			byteswap(out, to.Size())
		}
		return out, nil
	}
}

// transposeStep rearranges a C-ordered array of shape `shape` so that
// out = transpose(in, perm), i.e. out.shape[i] == shape[perm[i]].
func transposeStep(shape, perm []int, elem int) step {
	return func(in []byte) ([]byte, error) { return transposeBytes(in, shape, perm, elem), nil }
}

func identityPerm(perm []int) bool {
	for i, p := range perm {
		if i != p {
			return false
		}
	}
	return true
}

func transposeBytes(src []byte, shape, perm []int, elem int) []byte {
	nd := len(shape)
	if identityPerm(perm) {
		return src
	}
	sstr := make([]int, nd)
	acc := 1
	for i := nd - 1; i >= 0; i-- {
		sstr[i] = acc
		acc *= shape[i]
	}
	dshape := make([]int, nd)
	dstr := make([]int, nd) // source stride for each destination axis
	for i, p := range perm {
		dshape[i] = shape[p]
		dstr[i] = sstr[p]
	}
	dst := make([]byte, len(src))
	if acc == 0 {
		return dst
	}
	idx := make([]int, nd)
	inner := dshape[nd-1]
	istr := dstr[nd-1]
	o := 0
	for {
		so := 0
		for i := 0; i < nd-1; i++ {
			so += idx[i] * dstr[i]
		}
		switch elem {
		case 4:
			for k := 0; k < inner; k++ {
				s := (so + k*istr) * 4
				copy(dst[o:o+4], src[s:s+4])
				o += 4
			}
		case 8:
			for k := 0; k < inner; k++ {
				s := (so + k*istr) * 8
				copy(dst[o:o+8], src[s:s+8])
				o += 8
			}
		default:
			for k := 0; k < inner; k++ {
				s := (so + k*istr) * elem
				copy(dst[o:o+elem], src[s:s+elem])
				o += elem
			}
		}
		i := nd - 2
		for ; i >= 0; i-- {
			idx[i]++
			if idx[i] < dshape[i] {
				break
			}
			idx[i] = 0
		}
		if i < 0 {
			break
		}
	}
	return dst
}

func inversePerm(p []int) []int {
	inv := make([]int, len(p))
	for i, v := range p {
		inv[v] = i
	}
	return inv
}

func permute(shape, perm []int) []int {
	out := make([]int, len(shape))
	for i, p := range perm {
		out[i] = shape[p]
	}
	return out
}
