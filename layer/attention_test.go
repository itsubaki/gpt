package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
)

func ExampleMultiHeadAttention() {
	// p100
	embeddim := 64
	numOfhead := 8
	headDim := 64
	batchSize := 2
	contextLen := 10

	x := variable.Randn([]int{batchSize, contextLen, embeddim})
	mha := L.MultiHeadAttention(embeddim, numOfhead, headDim, nil)
	output := mha.First(x)

	fmt.Println(x.Shape())
	fmt.Println(output.Shape())

	output.Backward()
	fmt.Println(x.Grad.Shape())

	// Output:
	// [2 10 64]
	// [2 10 64]
	// [2 10 64]
}
