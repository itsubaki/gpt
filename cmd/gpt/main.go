package main

import (
	"fmt"
	"math/rand"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/hook"
	O "github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/model"
)

func batch(B, C, V int) *variable.Variable {
	tokens := make([]float64, B*C)
	for i := range tokens {
		tokens[i] = float64(rand.Intn(V))
	}

	return variable.New(tokens...).Reshape(B, C)
}

// D2Z
func getLearningRate(it int, maxLR float64, warmupIters, maxIters int) float64 {
	if it < warmupIters {
		return maxLR * float64(it) / float64(warmupIters)
	}

	if it < maxIters {
		progress := float64(it-warmupIters) / float64(maxIters-warmupIters)
		return maxLR * (1.0 - progress)
	}

	return 0.0
}

func main() {
	vocabSize := 1000
	maxContextLen := 256
	embeddim := 384
	numOfHeads := 6
	numOfBlocks := 6
	ffdim := 4 * embeddim
	theta := 10000.0
	batchSize := 1
	maxLR := 6e-4
	warmupIters := 200
	maxIters := 40000

	m := model.NewGPT(
		vocabSize,
		maxContextLen,
		embeddim,
		numOfHeads,
		numOfBlocks,
		ffdim,
		theta,
	)

	o := O.AdamW{
		Adam: O.Adam{
			Alpha: maxLR,
			Beta1: 0.9,
			Beta2: 0.999,
			Hook: []O.Hook{
				hook.ClipGrad(1.0),
			},
		},
		WeightDecay: 0.01,
	}

	for i := range 3 {
		// learning rate scheduling
		lr := getLearningRate(i, maxLR, warmupIters, maxIters)
		o.Alpha = lr

		// batch
		x := batch(batchSize, maxContextLen, vocabSize)
		y := batch(batchSize, maxContextLen, vocabSize)

		// forward
		logits := m.Forward(x)
		loss := F.CrossEntropy(
			F.Reshape(-1, logits.Size(-1))(logits), // (B, C, V) -> (B*C, V)
			F.Reshape(-1)(y),                       // (B, C) -> (B*C)
		)
		fmt.Println(logits.Shape())
		fmt.Println(loss.At())

		// backward and update
		m.Cleargrads()
		loss.Backward()
		o.Update(m)
	}

	// print param shapes and total param count
	var total int
	for _, param := range m.Params().Seq2() {
		_, _ = param.Shape(), param.Grad.Shape()
		total += param.Size()
	}

	fmt.Println("total:", total)
}
