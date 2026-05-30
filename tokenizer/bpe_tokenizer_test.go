package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExampleBPETokenizer_Encode() {
	// p27
	train := "Hello world!<|endoftext|>This is BPE training."
	mergeRules := tokenizer.TrainBPE(train, 260)
	tknizer := tokenizer.NewBPETokenizer(mergeRules)

	text := "Hello world!<|endoftext|>"
	ids := tknizer.Encode(text)
	decoded := tknizer.Decode(ids)
	fmt.Println(ids)
	fmt.Println(decoded)

	// Output:
	// [72 101 108 108 111 32 119 111 114 108 100 33 259]
	// Hello world!<|endoftext|>
}

func ExampleMerge() {
	// p14
	ids := []int{1, 2, 3, 1, 2}
	merged := tokenizer.Merge(ids, tokenizer.Pair{1, 2}, 4)
	fmt.Println(merged)

	// Output:
	// [4 3 4]
}

func ExampleReSplit() {
	text := "Hello world!<|endoftext|>This is BPE training."
	pattern := "<|endoftext|>"
	split := tokenizer.ReSplit(text, pattern)
	for _, s := range split {
		fmt.Printf("[%s]", s)
	}

	// Output:
	// [Hello world!][<|endoftext|>][This is BPE training.]
}
