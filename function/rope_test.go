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

	y := F.RoPE(10000, 4, 4)(x)
	fmt.Println(y)

	// Output:
	// variable[1 1 2 4]([1 2 3 4 -1.1426396637476532 1.922075596544176 2.9598506679133294 4.029799501669161])
}
