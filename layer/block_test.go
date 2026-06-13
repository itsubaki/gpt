package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
)

func ExampleBlock() {
	embedDim := 64
	numOfhead := 8
	batchSize := 2
	contextLen := 30

	x := variable.Randn([]int{batchSize, contextLen, embedDim})
	block := L.Block(embedDim, numOfhead)

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
