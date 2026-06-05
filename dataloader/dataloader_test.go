package dataloader_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/dataloader"
)

func ExampleDataLoader() {
	data := []*variable.Variable{
		variable.New(1),
		variable.New(2),
		variable.New(3),
		variable.New(4),
		variable.New(5),
	}

	label := []*variable.Variable{
		variable.New(10),
		variable.New(20),
		variable.New(30),
		variable.New(40),
		variable.New(50),
	}

	loader := &dataloader.DataLoader{
		BatchSize: 2,
		N:         5,
		Data:      data,
		Label:     label,
		Shuffle:   false,
	}

	for range 4 {
		x, y := loader.Batch()
		fmt.Println(x, y)
	}

	// Output:
	// [variable(1) variable(2)] [variable(10) variable(20)]
	// [variable(3) variable(4)] [variable(30) variable(40)]
	// [variable(5)] [variable(50)]
	// [variable(1) variable(2)] [variable(10) variable(20)]
}
