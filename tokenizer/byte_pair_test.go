package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExampleCount() {
	// p14
	ids := []int{1, 2, 3, 1, 2}
	counts := tokenizer.Count(ids)
	for pair, count := range counts {
		fmt.Println(pair, count)
	}

	// Unordered output:
	// [1 2] 2
	// [2 3] 1
	// [3 1] 1
}

func ExampleMerge() {
	// p14
	ids := []int{1, 2, 3, 1, 2}
	merged := tokenizer.Merge(ids, tokenizer.Pair{1, 2}, 4)
	fmt.Println(merged)

	// Output:
	// [4 3 4]
}
