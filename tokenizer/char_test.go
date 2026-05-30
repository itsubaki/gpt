package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExampleCharTokenizer() {
	// p7
	tokenizer := tokenizer.NewCharTokenizer()
	text := "hello世界😁"

	tokens := tokenizer.Encode(text)
	decoded := tokenizer.Decode(tokens)

	fmt.Println(tokens)
	fmt.Println(decoded)

	// Output:
	// [104 101 108 108 111 19990 30028 128513]
	// hello世界😁
}
