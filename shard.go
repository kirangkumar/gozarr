package zarr

import (
	"encoding/binary"
	"fmt"
)

// shardSpec describes a sharding_indexed codec: one stored object (a shard)
// holds many independently compressed inner chunks plus an index of their
// byte ranges. That index is what lets us read a single point with one small
// range request instead of fetching the shard.
type shardSpec struct {
	shardShape []int
	innerShape []int
	perShard   []int // inner chunks per shard along each axis
	inner      *codecChain
	indexAtEnd bool
	indexCRC   bool
	indexBig   bool
	indexSize  int // bytes, including the checksum
	dt         DType
}

const emptyEntry = ^uint64(0)

func buildShard(cfg []byte, dt DType, shape []int) (*shardSpec, error) {
	var c struct {
		ChunkShape  []int      `json:"chunk_shape"`
		Codecs      []rawCodec `json:"codecs"`
		IndexCodecs []rawCodec `json:"index_codecs"`
		IndexLoc    string     `json:"index_location"`
	}
	if err := decodeJSON(cfg, &c); err != nil {
		return nil, fmt.Errorf("zarr: bad sharding config: %w", err)
	}
	if len(c.ChunkShape) != len(shape) {
		return nil, fmt.Errorf("zarr: sharding chunk_shape rank mismatch")
	}
	sp := &shardSpec{shardShape: shape, innerShape: c.ChunkShape, dt: dt, indexAtEnd: c.IndexLoc != "start"}
	sp.perShard = make([]int, len(shape))
	for i := range shape {
		if c.ChunkShape[i] <= 0 || shape[i]%c.ChunkShape[i] != 0 {
			return nil, fmt.Errorf("zarr: inner chunk shape does not divide shard shape")
		}
		sp.perShard[i] = shape[i] / c.ChunkShape[i]
	}
	inner, err := buildCodecChain(c.Codecs, dt, c.ChunkShape, true)
	if err != nil {
		return nil, err
	}
	sp.inner = inner
	sp.indexSize = prod(sp.perShard) * 16
	for _, ic := range c.IndexCodecs {
		switch ic.Name {
		case "bytes":
			var e struct {
				Endian string `json:"endian"`
			}
			if len(ic.Config) > 0 {
				_ = decodeJSON(ic.Config, &e)
			}
			sp.indexBig = e.Endian == "big"
		case "crc32c":
			sp.indexCRC = true
			sp.indexSize += 4
		default:
			return nil, fmt.Errorf("zarr: unsupported index codec %q", ic.Name)
		}
	}
	return sp, nil
}

// parseIndex validates the raw index bytes and returns the entries as
// little-endian uint64 pairs.
func (s *shardSpec) parseIndex(raw []byte) ([]byte, error) {
	if len(raw) != s.indexSize {
		return nil, fmt.Errorf("zarr: shard index is %d bytes, want %d", len(raw), s.indexSize)
	}
	if s.indexCRC {
		var err error
		if raw, err = crc32cStep(raw); err != nil {
			return nil, fmt.Errorf("zarr: shard index: %w", err)
		}
	}
	if s.indexBig {
		raw = append([]byte(nil), raw...)
		byteswap(raw, 8)
	}
	return raw, nil
}

func (s *shardSpec) entry(index []byte, pos int) (off, n uint64, present bool) {
	off = binary.LittleEndian.Uint64(index[pos*16:])
	n = binary.LittleEndian.Uint64(index[pos*16+8:])
	return off, n, !(off == emptyEntry && n == emptyEntry)
}

// decodeInner decodes the stored bytes of one inner chunk.
func (s *shardSpec) decodeInner(raw []byte, fill []byte) ([]byte, error) {
	if s.inner.pipe != nil {
		return s.inner.pipe.decode(raw)
	}
	return s.inner.shard.decodeAll(raw, fill) // nested sharding
}

// decodeAll decodes a whole shard (all inner chunks) into one C-ordered buffer.
// Used for nested sharding; top-level shards are read chunk by chunk instead.
func (s *shardSpec) decodeAll(raw []byte, fill []byte) ([]byte, error) {
	if len(raw) < s.indexSize {
		return nil, fmt.Errorf("zarr: shard shorter than its index")
	}
	var idxRaw []byte
	if s.indexAtEnd {
		idxRaw = raw[len(raw)-s.indexSize:]
	} else {
		idxRaw = raw[:s.indexSize]
	}
	index, err := s.parseIndex(idxRaw)
	if err != nil {
		return nil, err
	}
	es := s.dt.Size()
	out := make([]byte, prod(s.shardShape)*es)
	fillBuf(out, fill)
	nd := len(s.shardShape)
	pos := make([]int, nd)
	for k := 0; k < prod(s.perShard); k++ {
		off, n, ok := s.entry(index, k)
		if ok {
			if off+n > uint64(len(raw)) {
				return nil, fmt.Errorf("zarr: shard entry out of range")
			}
			chunk, err := s.decodeInner(raw[off:off+n], fill)
			if err != nil {
				return nil, err
			}
			copyBlock(out, s.shardShape, chunk, s.innerShape, pos, s.innerShape, es)
		}
		for i := nd - 1; i >= 0; i-- { // advance pos in C order
			pos[i]++
			if pos[i] < s.perShard[i] {
				break
			}
			pos[i] = 0
		}
	}
	return out, nil
}

func fillBuf(b, fill []byte) {
	allZero := true
	for _, v := range fill {
		if v != 0 {
			allZero = false
		}
	}
	if allZero {
		return
	}
	for i := 0; i+len(fill) <= len(b); i += len(fill) {
		copy(b[i:], fill)
	}
}

// copyBlock copies `src` (shape ss) into `dst` (shape ds) at block position
// pos (in units of blk), where blk == ss.
func copyBlock(dst []byte, ds []int, src []byte, ss []int, pos, blk []int, es int) {
	nd := len(ds)
	dstr := make([]int, nd)
	acc := 1
	for i := nd - 1; i >= 0; i-- {
		dstr[i] = acc
		acc *= ds[i]
	}
	base := 0
	for i := 0; i < nd; i++ {
		base += pos[i] * blk[i] * dstr[i]
	}
	idx := make([]int, nd)
	row := ss[nd-1] * es
	so := 0
	for {
		d := base
		for i := 0; i < nd-1; i++ {
			d += idx[i] * dstr[i]
		}
		copy(dst[d*es:d*es+row], src[so:so+row])
		so += row
		i := nd - 2
		for ; i >= 0; i-- {
			idx[i]++
			if idx[i] < ss[i] {
				break
			}
			idx[i] = 0
		}
		if i < 0 {
			break
		}
	}
}
