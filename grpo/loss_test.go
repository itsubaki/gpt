package grpo_test

import (
	"fmt"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/grpo"
)

var _ grpo.Model = (*MockModel)(nil)

type MockModel struct {
	logits  []float64
	B, C, V int
}

func (m *MockModel) Forward(ids *variable.Variable) *variable.Variable {
	return variable.New(m.logits...).Reshape(m.B, m.C, m.V)
}

func ExampleComputeProbs() {
	m := &MockModel{
		logits: []float64{
			1, 0, 0,
			0, 1, 0,

			0, 0, 1,
			1, 0, 0,
		},
		B: 2,
		C: 2,
		V: 3,
	}

	ids := variable.New(
		0, 1,
		1, 2,
	).Reshape(2, 2)

	probs := grpo.ComputeProbs(m, ids)
	probs.Backward()
	fmt.Println(probs)

	// Output:
	// variable[2 1]([0.21194155761708544 0.5761168847658291])
}

func ExampleLoss() {
	model := &MockModel{
		logits: []float64{
			2, 1, 0,
			0, 2, 1,
			1, 0, 2,

			2, 0, 1,
			1, 2, 0,
			0, 1, 2,
		},
		B: 2,
		C: 3,
		V: 3,
	}

	oldModel := &MockModel{
		logits: []float64{
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,

			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		},
		B: 2,
		C: 3,
		V: 3,
	}

	ids := variable.New(
		0, 1, 2,
		1, 2, 0,
	).Reshape(2, 3)

	mask := variable.New(
		1, 1,
		1, 1,
	).Reshape(2, 2)

	advantages := []float64{
		1.0,
		0.7,
	}

	loss := grpo.Loss(
		model,
		oldModel,
		ids,
		mask,
		advantages,
		0.2,
	)

	fmt.Println(loss)

	// Output:
	// variable(-1.9629863337842814)
}
