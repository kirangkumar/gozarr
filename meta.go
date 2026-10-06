package zarr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// arrayMeta is the format-independent description of an array.
type arrayMeta struct {
	version    int
	shape      []int
	chunkShape []int // chunk (or shard) shape: the unit stored under one key
	dtype      DType
	fill       []byte // one element, little endian
	dimNames   []string
	attrs      map[string]any
	keyFn      func(coords []int) string

	pipe  *pipeline  // decodes one stored chunk (non-sharded arrays)
	shard *shardSpec // set when the array is sharded
}

type rawCodec struct {
	Name   string          `json:"name"`
	Config json.RawMessage `json:"configuration"`
}

func decodeJSON(b []byte, v any) error {
	d := json.NewDecoder(bytes.NewReader(b))
	d.UseNumber()
	return d.Decode(v)
}

func atoiSlice(in []int) []int { return append([]int(nil), in...) }

func prod(s []int) int {
	p := 1
	for _, v := range s {
		p *= v
	}
	return p
}

// ----------------------------------------------------------------- fill value

func encodeFill(dt DType, raw json.RawMessage) ([]byte, error) {
	out := make([]byte, dt.Size())
	s := strings.TrimSpace(string(raw))
	if s == "" || s == "null" {
		return out, nil
	}
	var v float64
	switch {
	case s == "true":
		v = 1
	case s == "false":
		v = 0
	case s[0] == '"':
		var str string
		if err := json.Unmarshal(raw, &str); err != nil {
			return nil, err
		}
		switch str {
		case "NaN":
			v = math.NaN()
		case "Infinity":
			v = math.Inf(1)
		case "-Infinity":
			v = math.Inf(-1)
		default:
			if strings.HasPrefix(str, "0x") { // raw IEEE bits, zarr v3
				bits, err := strconv.ParseUint(str[2:], 16, 64)
				if err != nil {
					return nil, fmt.Errorf("zarr: bad fill_value %q", str)
				}
				for i := 0; i < dt.Size(); i++ {
					out[i] = byte(bits >> (8 * i))
				}
				return out, nil
			}
			f, err := strconv.ParseFloat(str, 64)
			if err != nil {
				return nil, fmt.Errorf("zarr: bad fill_value %q", str)
			}
			v = f
		}
	default:
		switch dt {
		case Int64:
			if n, err := strconv.ParseInt(s, 10, 64); err == nil {
				storeInt64(out, uint64(n))
				return out, nil
			}
		case Uint64:
			if n, err := strconv.ParseUint(s, 10, 64); err == nil {
				storeInt64(out, n)
				return out, nil
			}
		}
		f, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return nil, fmt.Errorf("zarr: bad fill_value %s", s)
		}
		v = f
	}
	storeFloat64(dt, out, 0, v)
	return out, nil
}

func storeInt64(b []byte, v uint64) {
	for i := 0; i < 8; i++ {
		b[i] = byte(v >> (8 * i))
	}
}

// ------------------------------------------------------------------------ v2

type v2Meta struct {
	Shape      []int            `json:"shape"`
	Chunks     []int            `json:"chunks"`
	DType      json.RawMessage  `json:"dtype"`
	Fill       json.RawMessage  `json:"fill_value"`
	Order      string           `json:"order"`
	Filters    []map[string]any `json:"filters"`
	Compressor map[string]any   `json:"compressor"`
	Sep        string           `json:"dimension_separator"`
}

func num(m map[string]any, k string) float64 {
	switch v := m[k].(type) {
	case json.Number:
		f, _ := v.Float64()
		return f
	case float64:
		return v
	}
	return 0
}

func str(m map[string]any, k string) string { s, _ := m[k].(string); return s }

func parseV2(zarray []byte, attrs map[string]any) (*arrayMeta, error) {
	var m v2Meta
	if err := decodeJSON(zarray, &m); err != nil {
		return nil, fmt.Errorf("zarr: bad .zarray: %w", err)
	}
	var dts string
	if err := json.Unmarshal(m.DType, &dts); err != nil {
		return nil, fmt.Errorf("zarr: unsupported (structured) dtype %s", m.DType)
	}
	dt, be, err := dtypeFromV2(dts)
	if err != nil {
		return nil, err
	}
	if len(m.Shape) != len(m.Chunks) {
		return nil, fmt.Errorf("zarr: shape/chunks rank mismatch")
	}
	fill, err := encodeFill(dt, m.Fill)
	if err != nil {
		return nil, err
	}
	sep := m.Sep
	if sep == "" {
		sep = "."
	}
	meta := &arrayMeta{
		version: 2, shape: m.Shape, chunkShape: m.Chunks, dtype: dt, fill: fill, attrs: attrs,
		keyFn: func(c []int) string {
			if len(c) == 0 {
				return "0"
			}
			var sb strings.Builder
			for i, v := range c {
				if i > 0 {
					sb.WriteString(sep)
				}
				sb.WriteString(strconv.Itoa(v))
			}
			return sb.String()
		},
	}
	// Decode order is the reverse of encode order: compressor, then filters
	// (last first), then endianness, then undo the memory order.
	var steps []step
	if m.Compressor != nil {
		s, err := v2Compressor(m.Compressor)
		if err != nil {
			return nil, err
		}
		steps = append(steps, s)
	}
	cur, curBig := dt, be // dtype of the bytes as the filters see them
	for i := len(m.Filters) - 1; i >= 0; i-- {
		f := m.Filters[i]
		switch id := str(f, "id"); id {
		case "shuffle":
			es := int(num(f, "elementsize"))
			if es == 0 {
				es = dt.Size()
			}
			steps = append(steps, shuffleStep(es))
		case "delta":
			fdt, fbe, err := dtypeFromV2(str(f, "dtype"))
			if err != nil {
				return nil, err
			}
			steps = append(steps, deltaStep(fdt, fbe))
			if a := str(f, "astype"); a != "" && a != str(f, "dtype") {
				adt, abe, err := dtypeFromV2(a)
				if err != nil {
					return nil, err
				}
				steps = append(steps, convertStep(adt, abe, fdt, fbe, nil))
			}
		case "fixedscaleoffset":
			fdt, fbe, err := dtypeFromV2(str(f, "dtype"))
			if err != nil {
				return nil, err
			}
			adt, abe, err := dtypeFromV2(str(f, "astype"))
			if err != nil {
				return nil, err
			}
			scale, offset := num(f, "scale"), num(f, "offset")
			steps = append(steps, convertStep(adt, abe, fdt, fbe, func(v float64) float64 { return v/scale + offset }))
		case "astype":
			fdt, fbe, err := dtypeFromV2(str(f, "decode_dtype"))
			if err != nil {
				return nil, err
			}
			edt, ebe, err := dtypeFromV2(str(f, "encode_dtype"))
			if err != nil {
				return nil, err
			}
			steps = append(steps, convertStep(edt, ebe, fdt, fbe, nil))
		case "bitround", "quantize": // lossy on encode, identity on decode
		default:
			return nil, fmt.Errorf("zarr: unsupported v2 filter %q", id)
		}
	}
	_ = cur
	_ = curBig
	if be && dt.Size() > 1 {
		steps = append(steps, endianStep(dt.Size()))
	}
	nd := len(m.Shape)
	if m.Order == "F" && nd > 1 {
		perm := make([]int, nd)
		for i := range perm {
			perm[i] = nd - 1 - i
		}
		// bytes are the C-order flattening of the axis-reversed array
		steps = append(steps, transposeStep(permute(m.Chunks, perm), perm, dt.Size()))
	}
	meta.pipe = &pipeline{steps: steps}
	return meta, nil
}

func v2Compressor(c map[string]any) (step, error) {
	switch id := str(c, "id"); id {
	case "blosc":
		return bloscStep, nil
	case "zstd":
		return zstdStep, nil
	case "gzip":
		return gzipStep, nil
	case "zlib":
		return zlibStep, nil
	case "bz2":
		return bz2Step, nil
	case "lz4":
		return lz4Step, nil
	default:
		return nil, fmt.Errorf("zarr: unsupported v2 compressor %q", id)
	}
}

// ------------------------------------------------------------------------ v3

type v3Meta struct {
	Shape     []int           `json:"shape"`
	DataType  json.RawMessage `json:"data_type"`
	ChunkGrid struct {
		Name   string `json:"name"`
		Config struct {
			ChunkShape []int `json:"chunk_shape"`
		} `json:"configuration"`
	} `json:"chunk_grid"`
	KeyEnc struct {
		Name   string `json:"name"`
		Config struct {
			Sep string `json:"separator"`
		} `json:"configuration"`
	} `json:"chunk_key_encoding"`
	Fill     json.RawMessage `json:"fill_value"`
	Codecs   []rawCodec      `json:"codecs"`
	Dims     []string        `json:"dimension_names"`
	Attrs    map[string]any  `json:"attributes"`
	NodeType string          `json:"node_type"`
}

func parseV3(b []byte) (*arrayMeta, error) {
	var m v3Meta
	if err := decodeJSON(b, &m); err != nil {
		return nil, fmt.Errorf("zarr: bad zarr.json: %w", err)
	}
	return buildV3(&m)
}

func buildV3(m *v3Meta) (*arrayMeta, error) {
	var dts string
	if err := json.Unmarshal(m.DataType, &dts); err != nil {
		return nil, fmt.Errorf("zarr: unsupported data_type %s", m.DataType)
	}
	dt, err := dtypeFromV3(dts)
	if err != nil {
		return nil, err
	}
	if m.ChunkGrid.Name != "regular" {
		return nil, fmt.Errorf("zarr: unsupported chunk_grid %q", m.ChunkGrid.Name)
	}
	cs := m.ChunkGrid.Config.ChunkShape
	if len(cs) != len(m.Shape) {
		return nil, fmt.Errorf("zarr: shape/chunk_shape rank mismatch")
	}
	fill, err := encodeFill(dt, m.Fill)
	if err != nil {
		return nil, err
	}
	meta := &arrayMeta{version: 3, shape: m.Shape, chunkShape: cs, dtype: dt, fill: fill,
		dimNames: m.Dims, attrs: m.Attrs}
	switch m.KeyEnc.Name {
	case "default", "":
		sep := m.KeyEnc.Config.Sep
		if sep == "" {
			sep = "/"
		}
		meta.keyFn = func(c []int) string {
			var sb strings.Builder
			sb.WriteByte('c')
			for _, v := range c {
				sb.WriteString(sep)
				sb.WriteString(strconv.Itoa(v))
			}
			return sb.String()
		}
	case "v2":
		sep := m.KeyEnc.Config.Sep
		if sep == "" {
			sep = "."
		}
		meta.keyFn = func(c []int) string {
			if len(c) == 0 {
				return "0"
			}
			parts := make([]string, len(c))
			for i, v := range c {
				parts[i] = strconv.Itoa(v)
			}
			return strings.Join(parts, sep)
		}
	default:
		return nil, fmt.Errorf("zarr: unsupported chunk_key_encoding %q", m.KeyEnc.Name)
	}
	cc, err := buildCodecChain(m.Codecs, dt, cs, true)
	if err != nil {
		return nil, err
	}
	if cc.shard != nil {
		meta.shard = cc.shard
	} else {
		meta.pipe = cc.pipe
	}
	return meta, nil
}

// codecChain is the result of interpreting a v3 "codecs" list.
type codecChain struct {
	pipe  *pipeline
	shard *shardSpec
}

// buildCodecChain interprets a v3 codec list for chunks of the given shape.
// allowShard is false inside an index codec chain.
func buildCodecChain(codecs []rawCodec, dt DType, shape []int, allowShard bool) (*codecChain, error) {
	// Find the array->bytes codec; everything before it is array->array,
	// everything after is bytes->bytes.
	abAt := -1
	for i, c := range codecs {
		if c.Name == "bytes" || c.Name == "sharding_indexed" {
			abAt = i
			break
		}
	}
	if abAt < 0 {
		return nil, fmt.Errorf("zarr: codec list has no array->bytes codec")
	}
	// array->array codecs in encode order, tracking the shape after each.
	shapes := [][]int{shape}
	var aa []rawCodec
	for _, c := range codecs[:abAt] {
		switch c.Name {
		case "transpose":
			var cfg struct {
				Order []int `json:"order"`
			}
			if err := decodeJSON(c.Config, &cfg); err != nil || len(cfg.Order) != len(shape) {
				return nil, fmt.Errorf("zarr: bad transpose codec")
			}
			shapes = append(shapes, permute(shapes[len(shapes)-1], cfg.Order))
			aa = append(aa, c)
		default:
			return nil, fmt.Errorf("zarr: unsupported array->array codec %q", c.Name)
		}
	}
	var steps []step
	// bytes->bytes in reverse (decode) order
	for i := len(codecs) - 1; i > abAt; i-- {
		c := codecs[i]
		switch c.Name {
		case "gzip":
			steps = append(steps, gzipStep)
		case "zstd":
			steps = append(steps, zstdStep)
		case "blosc":
			steps = append(steps, bloscStep)
		case "crc32c":
			steps = append(steps, crc32cStep)
		case "bz2":
			steps = append(steps, bz2Step)
		default:
			return nil, fmt.Errorf("zarr: unsupported bytes->bytes codec %q", c.Name)
		}
	}
	ab := codecs[abAt]
	if ab.Name == "sharding_indexed" {
		if !allowShard || len(aa) > 0 || len(steps) > 0 {
			return nil, fmt.Errorf("zarr: unsupported codec layout around sharding_indexed")
		}
		sp, err := buildShard(ab.Config, dt, shape)
		if err != nil {
			return nil, err
		}
		return &codecChain{shard: sp}, nil
	}
	var cfg struct {
		Endian string `json:"endian"`
	}
	if len(ab.Config) > 0 {
		_ = decodeJSON(ab.Config, &cfg)
	}
	if cfg.Endian == "big" && dt.Size() > 1 {
		steps = append(steps, endianStep(dt.Size()))
	}
	for i := len(aa) - 1; i >= 0; i-- {
		var cfg struct {
			Order []int `json:"order"`
		}
		_ = decodeJSON(aa[i].Config, &cfg)
		// encoded shape is shapes[i+1]; undoing it yields shapes[i]
		steps = append(steps, transposeStep(shapes[i+1], inversePerm(cfg.Order), dt.Size()))
	}
	return &codecChain{pipe: &pipeline{steps: steps}}, nil
}

// pipeline turns the bytes stored under a chunk key into the chunk's elements:
// C order, little endian, in the array's dtype.
type pipeline struct{ steps []step }

func (p *pipeline) decode(in []byte) ([]byte, error) {
	var err error
	for _, s := range p.steps {
		if in, err = s(in); err != nil {
			return nil, err
		}
	}
	return in, nil
}
