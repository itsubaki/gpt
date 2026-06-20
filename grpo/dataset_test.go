package grpo_test

import (
	"fmt"

	"github.com/itsubaki/gpt/grpo"
)

func ExampleDataset() {
	dataset := grpo.NewDataset()
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
