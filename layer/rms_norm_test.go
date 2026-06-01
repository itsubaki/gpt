package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
)

func ExampleRMSNorm() {
	embeddim := 64

	x := variable.Randn([]int{10, 20, embeddim})
	norm := L.RMSNorm(embeddim)
	output := norm.First(x)

	fmt.Println(x.Shape())
	fmt.Println(output.Shape())

	output.Backward()
	fmt.Println(x.Grad.Shape())

	// Output:
	// [10 20 64]
	// [10 20 64]
	// [10 20 64]
}
