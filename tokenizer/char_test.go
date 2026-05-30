package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExampleCharTokenizer() {
	// p7
	tokenizer := tokenizer.NewCharTokenizer()
	text := "hello世界😁"

	ids := tokenizer.Encode(text)
	decoded := tokenizer.Decode(ids)

	fmt.Println(ids)
	fmt.Println(decoded)

	// Output:
	// [104 101 108 108 111 19990 30028 128513]
	// hello世界😁
}
