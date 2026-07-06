package grpo_test

import (
	"fmt"

	"github.com/itsubaki/gpt/grpo"
)

var _ grpo.Tokenizer = (*MockTokenizer)(nil)

type MockTokenizer struct{}

func (t *MockTokenizer) Encode(text string) []int {
	var encoded []int
	for _, r := range text {
		encoded = append(encoded, int(r))
	}

	return encoded
}

func ExampleDataset() {
	dataset := grpo.NewDataset(&MockTokenizer{})
	fmt.Println(dataset.Len())
	fmt.Println("---")

	prompt, gt := dataset.GetItem(4)
	fmt.Print(prompt)
	fmt.Println(gt)

	// Output:
	// 81
	// ---
	// ### Instruction:
	// 1+5=
	//
	// ### Response:
	// 6
}

func ExampleDataset_GetBatch() {
	dataset := grpo.NewDataset(&MockTokenizer{})
	ids, masks := dataset.GetBatch(
		[]string{
			"### Instruction:\n1+1=\n\n### Response:\n2",
			"### Instruction:\n9+9=\n\n### Response:\n18",
		},
		[]string{
			"2",
			"18",
		},
	)

	fmt.Println(ids.Shape())
	fmt.Println(masks.Shape())

	// Output:
	// [2 41]
	// [2 41]
}
