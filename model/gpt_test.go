package model_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/model"
)

func ExampleGPT() {
	vocabSize := 1000
	maxContextLen := 256
	embeddim := 384
	numOfHead := 6
	numOfBlock := 6
	ffdim := 4 * embeddim
	dropoutRate := 0.1

	m := model.NewGPT(
		vocabSize,
		maxContextLen,
		embeddim,
		numOfHead,
		numOfBlock,
		ffdim,
		dropoutRate,
	)

	tokens := make([]float64, maxContextLen)
	for i := range tokens {
		tokens[i] = float64(rand.Intn(vocabSize))
	}

	x := variable.New(tokens...).Reshape(1, maxContextLen)
	logits := m.Forward(x)
	fmt.Println(logits.Shape())
}
