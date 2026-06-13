package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
)

func ExampleLayerNorm() {
	embedDim := 64
	x := variable.Randn([]int{10, 20, embedDim})
	norm := L.LayerNorm(embedDim)

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
