package zarr

import (
	"container/list"
	"sync"
)

const maxDims = 16

// cacheKey identifies a decoded chunk (or a shard index) of one array.
// It is a plain value so lookups allocate nothing.
type cacheKey struct {
	arr   uint32
	index bool
	c     [maxDims]uint32
}

func (k *cacheKey) hash() uint32 {
	h := uint32(2166136261) ^ k.arr
	for _, v := range k.c {
		h = (h ^ v) * 16777619
	}
	return h
}

type cacheEntry struct {
	key  cacheKey
	data []byte
	el   *list.Element
}

type flight struct {
	wg   sync.WaitGroup
	data []byte
	err  error
}

type cacheShard struct {
	mu     sync.Mutex
	m      map[cacheKey]*cacheEntry
	lru    *list.List
	flying map[cacheKey]*flight
	bytes  int64
	max    int64
}

// Cache is a size-bounded LRU of decoded chunks, shared by any number of
// arrays. Concurrent requests for the same missing chunk are coalesced into a
// single fetch. The zero value is not usable; call NewCache.
type Cache struct {
	shards [32]cacheShard
	on     bool
}

// NewCache returns a cache holding at most maxBytes of decoded data. With
// maxBytes <= 0 nothing is retained, but concurrent fetches are still coalesced.
func NewCache(maxBytes int64) *Cache {
	c := &Cache{on: maxBytes > 0}
	for i := range c.shards {
		c.shards[i] = cacheShard{
			m: map[cacheKey]*cacheEntry{}, lru: list.New(),
			flying: map[cacheKey]*flight{}, max: maxBytes / int64(len(c.shards)),
		}
	}
	return c
}

func (c *Cache) shard(k *cacheKey) *cacheShard { return &c.shards[k.hash()%uint32(len(c.shards))] }

// get returns the cached value, or loads it with fn (once, however many
// goroutines ask). Cached slices are shared and must be treated as read-only.
func (c *Cache) get(k cacheKey, fn func() ([]byte, error)) ([]byte, error) {
	s := c.shard(&k)
	s.mu.Lock()
	if e, ok := s.m[k]; ok {
		s.lru.MoveToFront(e.el)
		d := e.data
		s.mu.Unlock()
		return d, nil
	}
	if f, ok := s.flying[k]; ok {
		s.mu.Unlock()
		f.wg.Wait()
		return f.data, f.err
	}
	f := &flight{}
	f.wg.Add(1)
	s.flying[k] = f
	s.mu.Unlock()

	f.data, f.err = fn()

	s.mu.Lock()
	delete(s.flying, k)
	if f.err == nil && c.on && int64(len(f.data)) <= s.max {
		e := &cacheEntry{key: k, data: f.data}
		e.el = s.lru.PushFront(e)
		s.m[k] = e
		s.bytes += int64(len(f.data))
		for s.bytes > s.max {
			old := s.lru.Back()
			oe := old.Value.(*cacheEntry)
			s.lru.Remove(old)
			delete(s.m, oe.key)
			s.bytes -= int64(len(oe.data))
		}
	}
	s.mu.Unlock()
	f.wg.Done()
	return f.data, f.err
}

// peek returns a cached value without loading.
func (c *Cache) peek(k cacheKey) ([]byte, bool) {
	s := c.shard(&k)
	s.mu.Lock()
	e, ok := s.m[k]
	if ok {
		s.lru.MoveToFront(e.el)
	}
	s.mu.Unlock()
	if !ok {
		return nil, false
	}
	return e.data, true
}

// Len reports how many bytes are currently cached.
func (c *Cache) Len() int64 {
	var n int64
	for i := range c.shards {
		s := &c.shards[i]
		s.mu.Lock()
		n += s.bytes
		s.mu.Unlock()
	}
	return n
}

var defaultCache = sync.OnceValue(func() *Cache { return NewCache(256 << 20) })
