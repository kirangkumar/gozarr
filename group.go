package zarr

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
)

// Group is a node that contains arrays and other groups. If the group carries
// consolidated metadata, every array below it opens without a single request.
type Group struct {
	store   Store
	path    string
	version int
	attrs   map[string]any
	// consolidated documents, keyed by node path relative to the group:
	// v2: contents of the ".zarray" file; v3: the node's zarr.json.
	arrays map[string]json.RawMessage
	v2attr map[string]map[string]any
}

// OpenGroup opens the group at path ("" for the store root).
func OpenGroup(ctx context.Context, store Store, path string) (*Group, error) {
	path = strings.Trim(path, "/")
	sub := SubStore(store, path)
	g := &Group{store: store, path: path}
	if b, err := sub.Get(ctx, "zarr.json"); err == nil {
		var doc struct {
			NodeType string         `json:"node_type"`
			Attrs    map[string]any `json:"attributes"`
			Consol   *struct {
				Metadata map[string]json.RawMessage `json:"metadata"`
			} `json:"consolidated_metadata"`
		}
		if err := decodeJSON(b, &doc); err != nil {
			return nil, err
		}
		if doc.NodeType != "group" {
			return nil, fmt.Errorf("zarr: %q is not a group", path)
		}
		g.version, g.attrs = 3, doc.Attrs
		if doc.Consol != nil {
			g.arrays = map[string]json.RawMessage{}
			for k, v := range doc.Consol.Metadata {
				var probe struct {
					NodeType string `json:"node_type"`
				}
				_ = json.Unmarshal(v, &probe)
				if probe.NodeType == "array" {
					g.arrays[k] = v
				}
			}
		}
		return g, nil
	} else if !errors.Is(err, ErrNotFound) {
		return nil, err
	}
	if _, err := sub.Get(ctx, ".zgroup"); err != nil {
		if errors.Is(err, ErrNotFound) {
			return nil, fmt.Errorf("zarr: no group at %q: %w", path, ErrNotFound)
		}
		return nil, err
	}
	g.version = 2
	if b, err := sub.Get(ctx, ".zattrs"); err == nil {
		_ = decodeJSON(b, &g.attrs)
	}
	if b, err := sub.Get(ctx, ".zmetadata"); err == nil {
		var doc struct {
			Metadata map[string]json.RawMessage `json:"metadata"`
		}
		if err := decodeJSON(b, &doc); err == nil {
			g.arrays = map[string]json.RawMessage{}
			g.v2attr = map[string]map[string]any{}
			for k, v := range doc.Metadata {
				switch {
				case strings.HasSuffix(k, "/.zarray"):
					g.arrays[strings.TrimSuffix(k, "/.zarray")] = v
				case strings.HasSuffix(k, "/.zattrs"):
					var a map[string]any
					if decodeJSON(v, &a) == nil {
						g.v2attr[strings.TrimSuffix(k, "/.zattrs")] = a
					}
				}
			}
		}
	}
	return g, nil
}

// Attrs returns the group's attributes.
func (g *Group) Attrs() map[string]any { return g.attrs }

// Version returns the Zarr format version, 2 or 3.
func (g *Group) Version() int { return g.version }

// Arrays lists array paths known from consolidated metadata, sorted. It is
// empty when the group has none: Zarr stores cannot be listed over plain HTTP.
func (g *Group) Arrays() []string {
	out := make([]string, 0, len(g.arrays))
	for k := range g.arrays {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// Array opens the named array (a path relative to the group).
func (g *Group) Array(ctx context.Context, name string, opts ...Option) (*Array, error) {
	name = strings.Trim(name, "/")
	full := strings.Trim(g.path+"/"+name, "/")
	if doc, ok := g.arrays[name]; ok {
		var m *arrayMeta
		var err error
		if g.version == 3 {
			m, err = parseV3(doc)
		} else {
			m, err = parseV2(doc, g.v2attr[name])
		}
		if err != nil {
			return nil, err
		}
		return newArray(g.store, full, m, opts)
	}
	return Open(ctx, g.store, full, opts...)
}
