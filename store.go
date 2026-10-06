package zarr

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// ErrNotFound is returned by a Store when a key does not exist. For chunk keys
// it is not an error at the array level: a missing chunk reads as the fill value.
var ErrNotFound = errors.New("zarr: key not found")

// Store is a read-only key/value view of a Zarr hierarchy. Keys use "/" as the
// separator and never start with one, e.g. "temp/.zarray" or "temp/c/0/1/2".
//
// Implementations must be safe for concurrent use.
type Store interface {
	// Get returns the whole value stored at key.
	Get(ctx context.Context, key string) ([]byte, error)
	// GetRange returns length bytes starting at offset. A shorter result is
	// returned when the value ends first.
	GetRange(ctx context.Context, key string, offset, length int64) ([]byte, error)
	// GetSuffix returns the last n bytes of the value (all of it if shorter).
	GetSuffix(ctx context.Context, key string, n int64) ([]byte, error)
}

// ---------------------------------------------------------------- MemoryStore

// MemoryStore keeps every value in memory. It is mainly useful for tests and as
// an overlay for already-downloaded data.
type MemoryStore struct {
	mu sync.RWMutex
	m  map[string][]byte
}

func NewMemoryStore() *MemoryStore { return &MemoryStore{m: map[string][]byte{}} }

// Set stores a value. The slice is retained, not copied.
func (s *MemoryStore) Set(key string, v []byte) {
	s.mu.Lock()
	s.m[key] = v
	s.mu.Unlock()
}

func (s *MemoryStore) get(key string) ([]byte, error) {
	s.mu.RLock()
	v, ok := s.m[key]
	s.mu.RUnlock()
	if !ok {
		return nil, ErrNotFound
	}
	return v, nil
}

func (s *MemoryStore) Get(_ context.Context, key string) ([]byte, error) { return s.get(key) }

func (s *MemoryStore) GetRange(_ context.Context, key string, off, n int64) ([]byte, error) {
	v, err := s.get(key)
	if err != nil {
		return nil, err
	}
	return sliceRange(v, off, n), nil
}

func (s *MemoryStore) GetSuffix(_ context.Context, key string, n int64) ([]byte, error) {
	v, err := s.get(key)
	if err != nil {
		return nil, err
	}
	if n > int64(len(v)) {
		n = int64(len(v))
	}
	return v[int64(len(v))-n:], nil
}

func sliceRange(v []byte, off, n int64) []byte {
	if off >= int64(len(v)) || off < 0 {
		return nil
	}
	end := off + n
	if end > int64(len(v)) || end < off {
		end = int64(len(v))
	}
	return v[off:end]
}

// -------------------------------------------------------------------- FSStore

// FSStore reads from a directory on the local file system.
type FSStore struct{ root string }

func NewFSStore(root string) *FSStore { return &FSStore{root: root} }

func (s *FSStore) path(key string) (string, error) {
	if strings.Contains(key, "..") {
		return "", fmt.Errorf("zarr: invalid key %q", key)
	}
	return filepath.Join(s.root, filepath.FromSlash(key)), nil
}

func (s *FSStore) Get(_ context.Context, key string) ([]byte, error) {
	p, err := s.path(key)
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(p)
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	return b, err
}

func (s *FSStore) open(key string) (*os.File, error) {
	p, err := s.path(key)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(p)
	if errors.Is(err, os.ErrNotExist) {
		return nil, ErrNotFound
	}
	return f, err
}

func (s *FSStore) GetRange(_ context.Context, key string, off, n int64) ([]byte, error) {
	f, err := s.open(key)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf := make([]byte, n)
	m, err := f.ReadAt(buf, off)
	if err != nil && err != io.EOF {
		return nil, err
	}
	return buf[:m], nil
}

func (s *FSStore) GetSuffix(_ context.Context, key string, n int64) ([]byte, error) {
	f, err := s.open(key)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	if n > st.Size() {
		n = st.Size()
	}
	buf := make([]byte, n)
	m, err := f.ReadAt(buf, st.Size()-n)
	if err != nil && err != io.EOF {
		return nil, err
	}
	return buf[:m], nil
}

// ------------------------------------------------------------------ HTTPStore

// HTTPStore reads from any HTTP(S) server that supports Range requests, which
// covers public GCS / S3 / Azure buckets, CDNs and plain static servers.
// For private buckets, pass a Client whose Transport signs the requests.
type HTTPStore struct {
	base   string
	client *http.Client
	header http.Header
}

// HTTPOption customises an HTTPStore.
type HTTPOption func(*HTTPStore)

// WithHTTPClient sets the client (timeouts, connection pool, request signing).
func WithHTTPClient(c *http.Client) HTTPOption { return func(s *HTTPStore) { s.client = c } }

// WithHeader adds a header (for example Authorization) to every request.
func WithHeader(k, v string) HTTPOption { return func(s *HTTPStore) { s.header.Set(k, v) } }

func NewHTTPStore(baseURL string, opts ...HTTPOption) *HTTPStore {
	s := &HTTPStore{
		base:   strings.TrimRight(baseURL, "/"),
		header: http.Header{},
		client: &http.Client{Transport: &http.Transport{
			MaxIdleConns: 256, MaxIdleConnsPerHost: 64, ForceAttemptHTTP2: true,
		}},
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

func (s *HTTPStore) do(ctx context.Context, key, rng string) ([]byte, error) {
	u := s.base + "/" + (&url.URL{Path: key}).EscapedPath()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return nil, err
	}
	for k, v := range s.header {
		req.Header[k] = v
	}
	if rng != "" {
		req.Header.Set("Range", rng)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK, http.StatusPartialContent:
	case http.StatusNotFound, http.StatusForbidden: // S3 answers 403 for missing keys without list rights
		io.Copy(io.Discard, resp.Body)
		return nil, ErrNotFound
	case http.StatusRequestedRangeNotSatisfiable:
		io.Copy(io.Discard, resp.Body)
		return nil, nil
	default:
		io.Copy(io.Discard, resp.Body)
		return nil, fmt.Errorf("zarr: GET %s: %s", u, resp.Status)
	}
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	// A server that ignores Range answers 200 with the whole body.
	if rng != "" && resp.StatusCode == http.StatusOK {
		return nil, errNoRange
	}
	return b, nil
}

var errNoRange = errors.New("zarr: server does not support HTTP range requests")

func (s *HTTPStore) Get(ctx context.Context, key string) ([]byte, error) { return s.do(ctx, key, "") }

func (s *HTTPStore) GetRange(ctx context.Context, key string, off, n int64) ([]byte, error) {
	return s.do(ctx, key, fmt.Sprintf("bytes=%d-%d", off, off+n-1))
}

func (s *HTTPStore) GetSuffix(ctx context.Context, key string, n int64) ([]byte, error) {
	return s.do(ctx, key, fmt.Sprintf("bytes=-%d", n))
}

// ---------------------------------------------------------------- PrefixStore

// SubStore returns a Store rooted at prefix inside parent.
func SubStore(parent Store, prefix string) Store {
	prefix = strings.Trim(prefix, "/")
	if prefix == "" {
		return parent
	}
	return &prefixStore{parent, prefix + "/"}
}

type prefixStore struct {
	Store
	p string
}

func (s *prefixStore) Get(ctx context.Context, k string) ([]byte, error) {
	return s.Store.Get(ctx, s.p+k)
}
func (s *prefixStore) GetRange(ctx context.Context, k string, o, n int64) ([]byte, error) {
	return s.Store.GetRange(ctx, s.p+k, o, n)
}
func (s *prefixStore) GetSuffix(ctx context.Context, k string, n int64) ([]byte, error) {
	return s.Store.GetSuffix(ctx, s.p+k, n)
}

// NewStore picks a Store from a location: an http(s) URL, a file:// URL or a
// local path.
func NewStore(loc string) (Store, error) {
	switch {
	case strings.HasPrefix(loc, "http://"), strings.HasPrefix(loc, "https://"):
		return NewHTTPStore(loc), nil
	case strings.HasPrefix(loc, "gs://"):
		return NewHTTPStore("https://storage.googleapis.com/" + strings.TrimPrefix(loc, "gs://")), nil
	case strings.HasPrefix(loc, "file://"):
		return NewFSStore(strings.TrimPrefix(loc, "file://")), nil
	case strings.Contains(loc, "://"):
		return nil, fmt.Errorf("zarr: unsupported store scheme in %q", loc)
	}
	return NewFSStore(loc), nil
}
