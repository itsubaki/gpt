package main

import (
	"fmt"
	"math/rand"

	F "github.com/itsubaki/autograd/function"
	O "github.com/itsubaki/autograd/optimizer"
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

func batchY(B, C, vocabSize int) *variable.Variable {
	tokens := make([]float64, B*C)
	for i := range tokens {
		tokens[i] = float64(rand.Intn(vocabSize))
	}

	return variable.New(tokens...).Reshape(B, C)
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

	o := O.SGD{
		LearningRate: 0.01,
	}

	// batch
	x := sample(vocabSize, maxContextLen)
	y := F.Reshape(-1)(batchY(1, maxContextLen, vocabSize)) // (B, C) -> (B*C)

	// forward
	logits := m.Forward(x)
	loss := F.CrossEntropy(F.Reshape(-1, vocabSize)(logits), y) // (B, C, V) -> (B*C, V)
	fmt.Println(logits.Shape())
	fmt.Println(loss.At())

	// backward and update
	m.Cleargrads()
	loss.Backward()
	o.Update(m)

	// print param shapes and total param count
	var total int
	for _, param := range m.Params().Seq2() {
		_, _ = param.Shape(), param.Grad.Shape()
		total += param.Size()
	}

	fmt.Println("total:", total)
}
