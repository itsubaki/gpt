package grpo_test

import (
	"fmt"

	"github.com/itsubaki/gpt/cmd/grpo/grpo"
)

func ExampleDataLoader() {
	loader := grpo.DataLoader{
		BatchSize: 1,
		Shuffle:   false,
		Dataset:   grpo.NewDataset(&MockTokenizer{}),
	}

	for range 3 {
		prompts, gts := loader.Batch()
		for i := range prompts {
			fmt.Print(prompts[i])
			fmt.Println(gts[i])
			fmt.Println("--------")
		}
	}

	// Output:
	// ### Instruction:
	// 1+1=
	//
	// ### Response:
	// 2
	// --------
	// ### Instruction:
	// 1+2=
	//
	// ### Response:
	// 3
	// --------
	// ### Instruction:
	// 1+3=
	//
	// ### Response:
	// 4
	// --------
}
