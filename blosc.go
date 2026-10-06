package zarr

import (
	"bytes"
	"compress/zlib"
	"encoding/binary"
	"errors"
	"fmt"
	"io"

	"github.com/klauspost/compress/zstd"
	"github.com/pierrec/lz4/v4"
)

// A pure-Go decoder for the Blosc 1 frame format (what numcodecs and zarr v3
// write): blosclz, lz4/lz4hc, zlib and zstd inside, with byte- and bit-shuffle.

const (
	bloscHeader   = 16
	bloscMemcpyed = 0x2
	bloscDoShuf   = 0x1
	bloscDoBit    = 0x4
	bloscNoSplit  = 0x10
)

var errBlosc = errors.New("zarr: corrupt blosc data")

// maxDecoded bounds any single decoded chunk, so a corrupt header or a
// compression bomb cannot make us allocate without limit.
const maxDecoded = 1 << 30

func bloscDecode(src []byte) ([]byte, error) {
	if len(src) < bloscHeader {
		return nil, errBlosc
	}
	flags := src[2]
	typesize := int(src[3])
	nbytes := int(binary.LittleEndian.Uint32(src[4:]))
	blocksize := int(binary.LittleEndian.Uint32(src[8:]))
	cbytes := int(binary.LittleEndian.Uint32(src[12:]))
	if cbytes > len(src) || nbytes < 0 || nbytes > maxDecoded {
		return nil, errBlosc
	}
	dst := make([]byte, nbytes)
	if nbytes == 0 {
		return dst, nil
	}
	if flags&bloscMemcpyed != 0 {
		if len(src) < bloscHeader+nbytes {
			return nil, errBlosc
		}
		copy(dst, src[bloscHeader:])
		return dst, nil
	}
	if blocksize <= 0 || typesize <= 0 {
		return nil, errBlosc
	}
	nblocks := nbytes / blocksize
	leftover := nbytes % blocksize
	if leftover != 0 {
		nblocks++
	}
	if len(src) < bloscHeader+4*nblocks {
		return nil, errBlosc
	}
	compFormat := (flags >> 5) & 0x7
	shuffled := flags&bloscDoShuf != 0 && typesize > 1
	bitshuffled := flags&bloscDoBit != 0 && typesize > 1
	var tmp []byte
	if shuffled || bitshuffled {
		tmp = make([]byte, blocksize)
	}
	for b := 0; b < nblocks; b++ {
		bsize := blocksize
		isLeft := false
		if b == nblocks-1 && leftover != 0 {
			bsize = leftover
			isLeft = true
		}
		start := int(binary.LittleEndian.Uint32(src[bloscHeader+4*b:]))
		if start < 0 || start > len(src) {
			return nil, errBlosc
		}
		out := dst[b*blocksize : b*blocksize+bsize]
		target := out
		if shuffled || bitshuffled {
			target = tmp[:bsize]
		}
		nsplits := 1
		if flags&bloscNoSplit == 0 && typesize <= 16 && bsize/typesize >= 128 && !isLeft {
			nsplits = typesize
		}
		neb := bsize / nsplits
		p := start
		o := 0
		for s := 0; s < nsplits; s++ {
			if p+4 > len(src) {
				return nil, errBlosc
			}
			cs := int(binary.LittleEndian.Uint32(src[p:]))
			p += 4
			if cs < 0 || p+cs > len(src) {
				return nil, errBlosc
			}
			chunk := src[p : p+cs]
			p += cs
			if cs == neb {
				copy(target[o:o+neb], chunk)
			} else if err := bloscDecompress(compFormat, chunk, target[o:o+neb]); err != nil {
				return nil, err
			}
			o += neb
		}
		switch {
		case bitshuffled:
			// c-blosc only bit-shuffles blocks holding a multiple of 8 elements;
			// any other block is stored as is.
			if (bsize/typesize)%8 == 0 {
				bitUnshuffle(out, target, typesize)
			} else {
				copy(out, target)
			}
		case shuffled:
			byteUnshuffle(out, target, typesize)
		}
	}
	return dst, nil
}

var zstdDec, _ = zstd.NewReader(nil, zstd.WithDecoderConcurrency(0), zstd.WithDecoderMaxMemory(maxDecoded))

func bloscDecompress(format byte, src, dst []byte) error {
	switch format {
	case 0:
		return blosclzDecode(src, dst)
	case 1:
		n, err := lz4.UncompressBlock(src, dst)
		if err != nil || n != len(dst) {
			return fmt.Errorf("zarr: blosc lz4: %v (got %d want %d)", err, n, len(dst))
		}
		return nil
	case 3:
		r, err := zlib.NewReader(bytes.NewReader(src))
		if err != nil {
			return err
		}
		defer r.Close()
		_, err = io.ReadFull(r, dst)
		return err
	case 4:
		out, err := zstdDec.DecodeAll(src, dst[:0])
		if err != nil {
			return err
		}
		if len(out) != len(dst) {
			return errBlosc
		}
		if len(out) > 0 && &out[0] != &dst[0] {
			copy(dst, out)
		}
		return nil
	}
	return fmt.Errorf("zarr: blosc compressor %d not supported", format)
}

// byteUnshuffle inverts blosc's byte shuffle. The trailing bytes that do not
// make a whole element are stored as is.
func byteUnshuffle(dst, src []byte, ts int) {
	n := len(src) / ts
	for j := 0; j < ts; j++ {
		plane := src[j*n : (j+1)*n]
		for i, v := range plane {
			dst[i*ts+j] = v
		}
	}
	copy(dst[n*ts:], src[n*ts:])
}

// bitUnshuffle inverts bitshuffle: elements are processed in groups of 8, the
// leftover elements and trailing bytes are stored as is.
func bitUnshuffle(dst, src []byte, ts int) {
	n := len(src) / ts
	m := n - n%8
	rows := m / 8
	for j := 0; j < ts; j++ {
		for g := 0; g < rows; g++ {
			// byte b of x is bit-row b of byte-plane j for element group g
			var x uint64
			for b := 0; b < 8; b++ {
				x |= uint64(src[(j*8+b)*rows+g]) << (8 * b)
			}
			x = transpose8x8(x)
			o := g*8*ts + j
			for k := 0; k < 8; k++ {
				dst[o+k*ts] = byte(x >> (8 * k))
			}
		}
	}
	copy(dst[m*ts:], src[m*ts:])
}

// transpose8x8 transposes an 8x8 bit matrix held in a uint64 (row = byte).
func transpose8x8(x uint64) uint64 {
	t := (x ^ (x >> 7)) & 0x00AA00AA00AA00AA
	x ^= t ^ (t << 7)
	t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCC
	x ^= t ^ (t << 14)
	t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0
	x ^= t ^ (t << 28)
	return x
}

// blosclzDecode decodes the FastLZ-derived format used by blosc's own codec.
func blosclzDecode(src, dst []byte) error {
	ip, op := 0, 0
	if len(src) == 0 {
		return errBlosc
	}
	ctrl := int(src[ip] & 31)
	ip++
	for {
		if ctrl >= 32 {
			length := (ctrl >> 5) - 1
			ofs := (ctrl & 31) << 8
			if length == 7-1 {
				for {
					if ip >= len(src) {
						return errBlosc
					}
					c := int(src[ip])
					ip++
					length += c
					if c != 255 {
						break
					}
				}
			}
			if ip >= len(src) {
				return errBlosc
			}
			code := int(src[ip])
			ip++
			length += 3
			ref := op - ofs - code
			if code == 255 && ofs == 31<<8 {
				if ip+2 > len(src) {
					return errBlosc
				}
				ofs = int(src[ip])<<8 | int(src[ip+1])
				ip += 2
				ref = op - ofs - 8191
			}
			ref--
			if ref < 0 || op+length > len(dst) {
				return errBlosc
			}
			for i := 0; i < length; i++ { // overlapping copies are intentional
				dst[op+i] = dst[ref+i]
			}
			op += length
		} else {
			ctrl++
			if ip+ctrl > len(src) || op+ctrl > len(dst) {
				return errBlosc
			}
			copy(dst[op:], src[ip:ip+ctrl])
			ip += ctrl
			op += ctrl
		}
		if ip >= len(src) {
			break
		}
		ctrl = int(src[ip])
		ip++
	}
	if op != len(dst) {
		return errBlosc
	}
	return nil
}
