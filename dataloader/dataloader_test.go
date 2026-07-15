package dataloader_test

import (
	"fmt"

	"github.com/itsubaki/gpt/dataloader"
)

func ExampleDataLoader() {
	loader := dataloader.DataLoader{
		BatchSize: 1,
		Shuffle:   false,
		Dataset: dataloader.NewTokenDataset(
			[]int{0, 1, 2, 3, 4, 5}, // tokens
			2,                       // contextLen
		),
	}

	for range 10 {
		x, y := loader.Batch()
		fmt.Println(x, y)
	}

	// Output:
	// variable[1 2]([0 1]) variable[1 2]([1 2])
	// variable[1 2]([1 2]) variable[1 2]([2 3])
	// variable[1 2]([2 3]) variable[1 2]([3 4])
	// variable[1 2]([3 4]) variable[1 2]([4 5])
	// variable[1 2]([0 1]) variable[1 2]([1 2])
	// variable[1 2]([1 2]) variable[1 2]([2 3])
	// variable[1 2]([2 3]) variable[1 2]([3 4])
	// variable[1 2]([3 4]) variable[1 2]([4 5])
	// variable[1 2]([0 1]) variable[1 2]([1 2])
	// variable[1 2]([1 2]) variable[1 2]([2 3])
}

func ExampleDataLoader_batch2() {
	loader := dataloader.DataLoader{
		BatchSize: 2,
		Shuffle:   false,
		Dataset: dataloader.NewTokenDataset(
			[]int{0, 1, 2, 3, 4, 5}, // tokens
			2,                       // contextLen
		),
	}

	for range 10 {
		x, y := loader.Batch()
		fmt.Println(x, y)
	}

	// Output:
	// variable[2 2]([0 1 1 2]) variable[2 2]([1 2 2 3])
	// variable[2 2]([2 3 3 4]) variable[2 2]([3 4 4 5])
	// variable[2 2]([0 1 1 2]) variable[2 2]([1 2 2 3])
	// variable[2 2]([2 3 3 4]) variable[2 2]([3 4 4 5])
	// variable[2 2]([0 1 1 2]) variable[2 2]([1 2 2 3])
	// variable[2 2]([2 3 3 4]) variable[2 2]([3 4 4 5])
	// variable[2 2]([0 1 1 2]) variable[2 2]([1 2 2 3])
	// variable[2 2]([2 3 3 4]) variable[2 2]([3 4 4 5])
	// variable[2 2]([0 1 1 2]) variable[2 2]([1 2 2 3])
	// variable[2 2]([2 3 3 4]) variable[2 2]([3 4 4 5])
}
