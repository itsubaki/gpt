package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/function"
	L "github.com/itsubaki/gpt/layer"
)

func ExampleMultiHeadAttention() {
	// p100
	embedDim := 512
	numOfhead := 8
	headDim := 64
	batchSize := 2
	contextLen := 10
	theta := 1000.0

	rope := function.RoPE(theta, embedDim, contextLen)
	mha := L.MultiHeadAttention(embedDim, numOfhead, headDim, rope)

	x := variable.Randn([]int{batchSize, contextLen, embedDim})
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
	theta := 1000.0

	rope := function.RoPE(theta, embedDim, contextLen)
	mha := L.MultiHeadAttention(embedDim, numOfhead, headDim, rope)

	x := variable.Randn([]int{batchSize, contextLen, embedDim})
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
