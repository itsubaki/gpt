package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExamplePreTokenize() {
	text := "Hello! I'm fine."
	preTokens := tokenizer.PreTokenize(text)
	for _, token := range preTokens {
		fmt.Printf("[%s]", token)
	}

	// Output:
	// [Hello][!][ I]['m][ fine][.]
}
