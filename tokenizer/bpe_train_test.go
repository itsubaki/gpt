package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExampleCountPairs() {
	// p14
	ids := []int{1, 2, 3, 1, 2}
	counts := tokenizer.CountPairs(ids, 1)
	for pair, count := range counts.Seq2() {
		fmt.Println(pair, count)
	}

	// Output:
	// [1 2] 2
	// [2 3] 1
	// [3 1] 1
}

func ExampleTrainBPE() {
	// p17
	sample := "Hello world!<|endoftext|>This is BPE training."
	mergeRules := tokenizer.TrainBPE(sample, 260)
	for pair, newID := range mergeRules.Seq2() {
		fmt.Println(pair, newID)
	}

	// Output:
	// [105 115] 256
	// [105 110] 257
	// [72 101] 258
}
