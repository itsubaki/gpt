package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExampleByteTokenizer() {
	// p10
	tokenizer := tokenizer.NewByteTokenizer()
	text := "hello世界😁"

	ids := tokenizer.Encode(text)
	decoded := tokenizer.Decode(ids)

	fmt.Printf("%v\n", ids)
	fmt.Println(decoded)

	// Output:
	// [104 101 108 108 111 228 184 150 231 149 140 240 159 152 129]
	// hello世界😁
}
