package layer_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
)

func ExampleBlock() {
	embeddim := 128
	numOfhead := 8
	ffdim := 4 * embeddim
	block := L.Block(embeddim, numOfhead, ffdim, 0.1)

	batchSize := 2
	contextLen := 30
	x := variable.Randn([]int{batchSize, contextLen, embeddim})

	output := block.First(x)
	fmt.Println(x.Shape())
	fmt.Println(output.Shape())

	// Output:
	// [2 30 128]
	// [2 30 128]
}
