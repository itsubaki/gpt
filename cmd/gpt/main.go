package main

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/model"
)

func sample(vocabSize, maxContextLen int) *variable.Variable {
	tokens := make([]float64, maxContextLen)
	for i := range tokens {
		tokens[i] = float64(rand.Intn(vocabSize))
	}

	return variable.New(tokens...).Reshape(1, maxContextLen)
}

func main() {
	vocabSize := 1000
	maxContextLen := 256
	embeddim := 384
	numOfHeads := 6
	numOfBlocks := 6
	ffdim := 4 * embeddim
	theta := 10000.0

	m := model.NewGPT(
		vocabSize,
		maxContextLen,
		embeddim,
		numOfHeads,
		numOfBlocks,
		ffdim,
		theta,
	)

	x := sample(vocabSize, maxContextLen)
	logits := m.Forward(x)
	fmt.Println(logits.Shape()) // [1 256 1000]

	logits.Backward()

	var total int
	for name, param := range m.Params().Seq2() {
		fmt.Println(name, param.Shape(), param.Grad.Shape())
		total += param.Size()
	}

	fmt.Println("total:", total)
}
