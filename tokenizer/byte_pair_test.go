package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExampleCountPairs() {
	// p14
	ids := []int{1, 2, 3, 1, 2}
	counts, _ := tokenizer.CountPairs(ids)
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

	// Output:
	// [105 115] 256
	// [256 32] 257
	// [105 110] 258
	// [72 101] 259
}

func ExampleBPETokenizer_Encode() {
	// p21
	train := "Hello world! This is BPE training."
	mergeRules := tokenizer.TrainBPE(train, 260)
	tknizer := tokenizer.NewBPETokenizer(mergeRules)

	text := "Hello世界😁"
	ids := tknizer.Encode(text)
	decoded := tknizer.Decode(ids)
	fmt.Println(ids)
	fmt.Println(decoded)

	// Output:
	// [259 108 108 111 228 184 150 231 149 140 240 159 152 129]
	// Hello世界😁
}
