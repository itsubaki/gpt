package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
)

func ExampleMultiHeadAttention() {
	// p100
	embeddim := 512
	numOfhead := 8
	headDim := 64
	mha := L.MultiHeadAttention(embeddim, numOfhead, headDim, 0.1)

	batchSize := 2
	contextLen := 10
	x := variable.Randn([]int{batchSize, contextLen, embeddim})

	output := mha.First(x)
	fmt.Println(x.Shape())
	fmt.Println(output.Shape())

	// Output:
	// [2 10 512]
	// [2 10 512]
}
