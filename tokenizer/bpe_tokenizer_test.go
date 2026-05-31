package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExampleBPETokenizer_Encode() {
	// p27
	mergeRules := tokenizer.NewDefaultDict[tokenizer.Pair, int]()
	mergeRules.Set(tokenizer.Pair{105, 115}, 256)
	mergeRules.Set(tokenizer.Pair{256, 32}, 257)
	mergeRules.Set(tokenizer.Pair{105, 110}, 258)

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

func ExampleBPETokenizer_Encode_preTokenize() {
	// p33
	sample := "Say hello! Why hello? Just hello.<|endoftext|>Good morning!"
	mergeRules := tokenizer.TrainBPE(sample, 270)
	tknizer := tokenizer.NewBPETokenizer(mergeRules)

	text := "Say hello!"
	ids := tknizer.Encode(text)
	decoded := tknizer.Decode(ids)

	fmt.Println(ids)
	fmt.Println(decoded)
	for _, id := range ids {
		fmt.Printf("%3d -> %q\n", id, tknizer.Decode([]int{id}))
	}

	// Output:
	// [266 260 33]
	// Say hello!
	// 266 -> "Say"
	// 260 -> " hello"
	//  33 -> "!"
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
	sample := "Hello world!<|endoftext|>This is BPE training."
	pattern := "<|endoftext|>"
	split := tokenizer.ReSplit(sample, pattern)
	for _, s := range split {
		fmt.Printf("[%s]", s)
	}

	// Output:
	// [Hello world!][<|endoftext|>][This is BPE training.]
}
