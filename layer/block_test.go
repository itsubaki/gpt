package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/function"
	L "github.com/itsubaki/gpt/layer"
)

func ExampleBlock() {
	embedDim := 64
	numOfhead := 8
	batchSize := 2
	contextLen := 30
	theta := 1000.0

	rope := function.RoPE(theta, embedDim, contextLen)
	block := L.Block(embedDim, numOfhead, rope)

	x := variable.Randn([]int{batchSize, contextLen, embedDim})
	output := block.First(x)
	fmt.Println(x.Shape())
	fmt.Println(output.Shape())

	output.Backward()
	fmt.Println(x.Grad.Shape())

	// Output:
	// [2 30 64]
	// [2 30 64]
	// [2 30 64]
}
