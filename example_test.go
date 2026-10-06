package zarr_test

import (
	"context"
	"fmt"

	zarr "github.com/kirangkumar/gozarr"
)

func Example() {
	ctx := context.Background()
	a, err := zarr.Open(ctx, zarr.NewFSStore("testdata"), "v2_zstd.zarr")
	if err != nil {
		fmt.Println("skipped:", err)
		return
	}
	fmt.Println(a.Shape(), a.DType())
	// Output: [40 50 30] float32
}
