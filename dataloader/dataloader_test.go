package dataloader_test

import (
	"fmt"

	"github.com/itsubaki/gpt/dataloader"
)

func ExampleDataLoader() {
	loader := dataloader.DataLoader{
		BatchSize: 2,
		Cycle:     true,
		Shuffle:   false,
		Dataset: &dataloader.TokenDataset{
			Tokens:     []int{0, 1, 2, 3, 4, 5},
			ContextLen: 2,
		},
	}

	for range 10 {
		x, y := loader.Batch()
		fmt.Println(x, y)
	}

	// Output:
	// variable[2 1]([0 1]) variable[2 1]([1 2])
	// variable[2 1]([1 2]) variable[2 1]([2 3])
	// variable[2 1]([2 3]) variable[2 1]([3 4])
	// variable[2 1]([3 4]) variable[2 1]([4 5])
	// variable[2 1]([0 1]) variable[2 1]([1 2])
	// variable[2 1]([1 2]) variable[2 1]([2 3])
	// variable[2 1]([2 3]) variable[2 1]([3 4])
	// variable[2 1]([3 4]) variable[2 1]([4 5])
	// variable[2 1]([0 1]) variable[2 1]([1 2])
	// variable[2 1]([1 2]) variable[2 1]([2 3])
}
