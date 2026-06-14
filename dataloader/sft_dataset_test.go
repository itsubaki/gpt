package dataloader_test

import (
	"fmt"

	"github.com/itsubaki/gpt/dataloader"
)

type MockTokenizer struct{}

func (t *MockTokenizer) Encode(text string) []int {
	var encoded []int
	for _, r := range text {
		encoded = append(encoded, int(r))
	}

	return encoded
}

func ExampleSFTDataset() {
	alpaca := []dataloader.Alpaca{
		{
			Instruction: "Hello",
			Response:    "Hello, how can I help you?",
		},
		{
			Instruction: "Hey",
			Response:    "Hey, what's up?",
		},
	}

	mockTokenizer := &MockTokenizer{}
	dataset := dataloader.NewSFTDataset(alpaca, mockTokenizer, 256)

	for i := range dataset.Len() {
		ids, labels := dataset.GetItem(i)
		fmt.Println(len(ids), len(labels))
	}

	// Output:
	// 256 256
	// 256 256
}
