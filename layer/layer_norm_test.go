package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
)

func ExampleLayerNorm() {
	embeddim := 64
	norm := L.LayerNorm(embeddim)

	x := variable.Randn([]int{10, 20, embeddim})
	output := norm.First(x)
	fmt.Println(x.Shape())
	fmt.Println(output.Shape())

	// Output:
	// [10 20 64]
	// [10 20 64]
}
