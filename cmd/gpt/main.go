package main

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/model"
)

func main() {
	vocabSize := 1000
	maxContextLen := 256
	embeddim := 384
	numOfHeads := 6
	numOfBlocks := 6
	ffdim := 4 * embeddim

	m := model.NewGPT(
		vocabSize,
		maxContextLen,
		embeddim,
		numOfHeads,
		numOfBlocks,
		ffdim,
	)

	tokens := make([]float64, maxContextLen)
	for i := range tokens {
		tokens[i] = float64(rand.Intn(vocabSize))
	}

	x := variable.New(tokens...).Reshape(1, maxContextLen)
	logits := m.Forward(x)

	// [1 256 1000]
	fmt.Println(logits.Shape())

	logits.Backward()

	var total int
	for name, param := range m.Params().Seq2() {
		fmt.Println(name, param.Shape(), param.Grad.Shape())
		total += param.Size()
	}
	fmt.Println("total:", total)
}
