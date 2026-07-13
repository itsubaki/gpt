package function_test

import (
	"fmt"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
	F "github.com/itsubaki/gpt/function"
)

func ExamplePick() {
	probs := variable.New(
		0.1, 0.2, 0.3,
		0.4, 0.5, 0.6,

		0.7, 0.8, 0.9,
		1.0, 1.1, 1.2,

		1.3, 1.4, 1.5,
		1.6, 1.7, 1.8,
	).Reshape(3, 2, 3)

	labels := tensor.New([]int{3, 2}, []int{
		2, 0,
		1, 2,
		0, 1,
	})

	y := F.Pick(labels)(probs)
	fmt.Println(y)

	y.Backward()
	fmt.Println(probs.Grad)

	// Output:
	// variable[3 2]([0.3 0.4 0.8 1.2 1.3 1.7])
	// variable[3 2 3]([0 0 1 1 0 0 0 1 0 0 0 1 1 0 0 0 1 0])
}
