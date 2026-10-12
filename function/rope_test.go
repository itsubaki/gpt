package function_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	F "github.com/itsubaki/gpt/function"
)

func ExampleRoPE() {
	x := variable.New(
		1, 2, 3, 4,
		1, 2, 3, 4,
	).Reshape(1, 1, 2, 4)

	var offset int
	y := F.RoPE(10000, 4, 4)(offset)(x)
	fmt.Println(y)

	// Output:
	// variable[1 1 2 4]([1 2 3 4 -1.1426396 1.9220755 2.9598508 4.0297995])
}

func ExampleRoPE_backward() {
	x := variable.New(
		1, 2, 3, 4,
		1, 2, 3, 4,
	).Reshape(1, 1, 2, 4)

	var offset int
	y := F.RoPE(10000, 4, 4)(offset)(x)
	y.Backward()
	fmt.Println(x.Grad)

	// Output:
	// variable[1 1 2 4]([1 1 1 1 1.3817732 -0.30116868 1.0099498 0.9899502])
}
