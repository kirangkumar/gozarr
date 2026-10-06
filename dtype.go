package zarr

import (
	"encoding/binary"
	"fmt"
	"math"
)

// DType is an element type of an array. Only fixed-size numeric types and
// booleans are supported; strings, complex and structured types are not.
type DType uint8

const (
	Bool DType = iota + 1
	Int8
	Int16
	Int32
	Int64
	Uint8
	Uint16
	Uint32
	Uint64
	Float16
	Float32
	Float64
)

var dtypeNames = map[DType]string{
	Bool: "bool", Int8: "int8", Int16: "int16", Int32: "int32", Int64: "int64",
	Uint8: "uint8", Uint16: "uint16", Uint32: "uint32", Uint64: "uint64",
	Float16: "float16", Float32: "float32", Float64: "float64",
}

func (d DType) String() string {
	if s, ok := dtypeNames[d]; ok {
		return s
	}
	return fmt.Sprintf("dtype(%d)", d)
}

// Size is the element size in bytes.
func (d DType) Size() int {
	switch d {
	case Bool, Int8, Uint8:
		return 1
	case Int16, Uint16, Float16:
		return 2
	case Int32, Uint32, Float32:
		return 4
	}
	return 8
}

func dtypeFromV3(name string) (DType, error) {
	for d, n := range dtypeNames {
		if n == name {
			return d, nil
		}
	}
	return 0, fmt.Errorf("zarr: unsupported data_type %q", name)
}

// dtypeFromV2 parses numpy typestrings like "<f4", ">i2", "|u1", "|b1".
// It returns the dtype and whether the stored bytes are big endian.
func dtypeFromV2(s string) (DType, bool, error) {
	if len(s) < 3 {
		return 0, false, fmt.Errorf("zarr: unsupported dtype %q", s)
	}
	be := s[0] == '>'
	kind, size := s[1], s[2:]
	var d DType
	switch {
	case kind == 'b' && size == "1":
		d = Bool
	case kind == 'i':
		d = map[string]DType{"1": Int8, "2": Int16, "4": Int32, "8": Int64}[size]
	case kind == 'u':
		d = map[string]DType{"1": Uint8, "2": Uint16, "4": Uint32, "8": Uint64}[size]
	case kind == 'f':
		d = map[string]DType{"2": Float16, "4": Float32, "8": Float64}[size]
	}
	if d == 0 {
		return 0, false, fmt.Errorf("zarr: unsupported dtype %q", s)
	}
	return d, be, nil
}

// byteswap reverses the byte order of every element in place.
func byteswap(b []byte, size int) {
	switch size {
	case 2:
		for i := 0; i+1 < len(b); i += 2 {
			b[i], b[i+1] = b[i+1], b[i]
		}
	case 4:
		for i := 0; i+3 < len(b); i += 4 {
			b[i], b[i+1], b[i+2], b[i+3] = b[i+3], b[i+2], b[i+1], b[i]
		}
	case 8:
		for i := 0; i+7 < len(b); i += 8 {
			b[i], b[i+1], b[i+2], b[i+3], b[i+4], b[i+5], b[i+6], b[i+7] =
				b[i+7], b[i+6], b[i+5], b[i+4], b[i+3], b[i+2], b[i+1], b[i]
		}
	}
}

// float16 -> float32 (IEEE 754 half precision).
func f16to32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	frac := uint32(h) & 0x3ff
	switch {
	case exp == 0:
		if frac == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 15 + 1)
		for frac&0x400 == 0 {
			frac <<= 1
			e--
		}
		frac &= 0x3ff
		return math.Float32frombits(sign | e<<23 | frac<<13)
	case exp == 0x1f:
		return math.Float32frombits(sign | 0x7f800000 | frac<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | frac<<13)
}

func f32to16(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	exp := int32(b>>23&0xff) - 127 + 15
	frac := b & 0x7fffff
	switch {
	case b>>23&0xff == 0xff:
		if frac != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case exp >= 0x1f:
		return sign | 0x7c00
	case exp <= 0:
		if exp < -10 {
			return sign
		}
		frac |= 0x800000
		shift := uint(14 - exp)
		h := uint16(frac >> shift)
		if frac>>(shift-1)&1 == 1 {
			h++
		}
		return sign | h
	}
	h := sign | uint16(exp)<<10 | uint16(frac>>13)
	if frac>>12&1 == 1 {
		h++
	}
	return h
}

// loadFloat64 reads element i (little endian, native layout) as float64.
func loadFloat64(d DType, b []byte, i int) float64 {
	switch d {
	case Bool, Uint8:
		return float64(b[i])
	case Int8:
		return float64(int8(b[i]))
	case Int16:
		return float64(int16(binary.LittleEndian.Uint16(b[i*2:])))
	case Uint16:
		return float64(binary.LittleEndian.Uint16(b[i*2:]))
	case Int32:
		return float64(int32(binary.LittleEndian.Uint32(b[i*4:])))
	case Uint32:
		return float64(binary.LittleEndian.Uint32(b[i*4:]))
	case Int64:
		return float64(int64(binary.LittleEndian.Uint64(b[i*8:])))
	case Uint64:
		return float64(binary.LittleEndian.Uint64(b[i*8:]))
	case Float16:
		return float64(f16to32(binary.LittleEndian.Uint16(b[i*2:])))
	case Float32:
		return float64(math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:])))
	}
	return math.Float64frombits(binary.LittleEndian.Uint64(b[i*8:]))
}

// storeFloat64 writes v as element i (little endian).
func storeFloat64(d DType, b []byte, i int, v float64) {
	switch d {
	case Bool:
		if v != 0 {
			b[i] = 1
		} else {
			b[i] = 0
		}
	case Int8:
		b[i] = byte(int8(v))
	case Uint8:
		b[i] = byte(uint8(v))
	case Int16:
		binary.LittleEndian.PutUint16(b[i*2:], uint16(int16(v)))
	case Uint16:
		binary.LittleEndian.PutUint16(b[i*2:], uint16(v))
	case Int32:
		binary.LittleEndian.PutUint32(b[i*4:], uint32(int32(v)))
	case Uint32:
		binary.LittleEndian.PutUint32(b[i*4:], uint32(v))
	case Int64:
		binary.LittleEndian.PutUint64(b[i*8:], uint64(int64(v)))
	case Uint64:
		binary.LittleEndian.PutUint64(b[i*8:], uint64(v))
	case Float16:
		binary.LittleEndian.PutUint16(b[i*2:], f32to16(float32(v)))
	case Float32:
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(float32(v)))
	default:
		binary.LittleEndian.PutUint64(b[i*8:], math.Float64bits(v))
	}
}

func float32frombits(b uint32) float32 { return math.Float32frombits(b) }
func float64frombits(b uint64) float64 { return math.Float64frombits(b) }
