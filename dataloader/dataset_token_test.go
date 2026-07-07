package dataloader_test

import (
	"fmt"

	"github.com/itsubaki/gpt/dataloader"
)

func ExampleTokenDataset() {
	dataset := dataloader.TokenDataset{
		Tokens:     []int{0, 1, 2, 3, 4, 5},
		ContextLen: 3,
	}

	for i := range dataset.Len() {
		x, y := dataset.GetItem(i)
		fmt.Println(x, y)
	}

	// Output:
	// [0 1 2] [1 2 3]
	// [1 2 3] [2 3 4]
	// [2 3 4] [3 4 5]
}
