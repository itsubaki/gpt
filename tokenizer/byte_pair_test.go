package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExampleCount() {
	// p14
	ids := []int{1, 2, 3, 1, 2}
	counts, _ := tokenizer.Count(ids)
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

func ExampleTrainBPE() {
	// p17
	text := "Hello world! This is BPE training."

	mergeRules := tokenizer.TrainBPE(text, 260)
	for pair, newID := range mergeRules.Seq2() {
		fmt.Println(pair, newID)
	}

	// Unordered output:
	// [105 115] 256
	// [256 32] 257
	// [105 110] 258
	// [72 101] 259
}
