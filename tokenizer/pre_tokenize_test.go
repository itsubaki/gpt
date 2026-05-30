package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExamplePreTokenize() {
	// p30
	sample := "Hello! I'm fine."
	preTokens := tokenizer.PreTokenize(sample)
	for _, token := range preTokens {
		fmt.Printf("[%s]", token)
	}

	// Output:
	// [Hello][!][ I]['m][ fine][.]
}

func ExamplePreTokenize_hello() {
	// p28
	sample := "Say hello! Why hello? Just hello."
	preTokens := tokenizer.PreTokenize(sample)
	for _, token := range preTokens {
		fmt.Printf("[%s]", token)
	}

	// Output:
	// [Say][ hello][!][ Why][ hello][?][ Just][ hello][.]
}
