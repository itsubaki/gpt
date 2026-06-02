package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/layer"
)

func ExampleRoPE() {
	theta := 0.5
	keydim := 4
	maxContextLen := 100

	x := variable.Randn([]int{10, 20, 30, keydim})
	rope := layer.RoPE(theta, keydim, maxContextLen)

	out := rope.First(x)
	fmt.Println(x.Shape())
	fmt.Println(out.Shape())

	out.Backward()
	fmt.Println(x.Grad.Shape())

	// Output:
	// [10 20 30 4]
	// [10 20 30 4]
	// [10 20 30 4]
}
