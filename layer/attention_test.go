package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
)

func ExampleMultiHeadAttention() {
	// p100
	embedDim := 512
	numOfhead := 8
	headDim := 64
	batchSize := 2
	contextLen := 10

	x := variable.Randn([]int{batchSize, contextLen, embedDim})
	mha := L.MultiHeadAttention(embedDim, numOfhead, headDim)

	output := mha.First(x)
	fmt.Println(x.Shape())
	fmt.Println(output.Shape())

	output.Backward()
	fmt.Println(x.Grad.Shape())

	// Output:
	// [2 10 512]
	// [2 10 512]
	// [2 10 512]
}

func ExampleMultiHeadAttention_rope() {
	// p215
	embedDim := 512
	numOfhead := 8
	headDim := 64
	batchSize := 2
	contextLen := 10

	x := variable.Randn([]int{batchSize, contextLen, embedDim})
	mha := L.MultiHeadAttention(embedDim, numOfhead, headDim)

	output := mha.First(x)
	fmt.Println(x.Shape())
	fmt.Println(output.Shape())

	output.Backward()
	fmt.Println(x.Grad.Shape())

	// Output:
	// [2 10 512]
	// [2 10 512]
	// [2 10 512]
}
