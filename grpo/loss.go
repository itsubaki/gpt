package grpo

import (
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/function"
)

type Model interface {
	Forward(x *variable.Variable) *variable.Variable
}

func ComputeProbs(model Model, ids *variable.Variable) *variable.Variable {
	logits := model.Forward(ids)                      // (B, C, V)
	logits = slice(logits, 1, 0, logits.Shape()[1]-1) // (B, C-1, V)
	probs := F.Softmax(-1)(logits)                    // (B, C-1, V)
	labels := slice(ids, 1, 1, ids.Shape()[1])        // (B, C-1)
	return function.PickLast(probs, labels)           // (B, C-1)
}

func slice(x *variable.Variable, axis, start, end int) *variable.Variable {
	indices := make([]int, end-start)
	for i := range indices {
		indices[i] = start + i
	}

	return variable.GetItem(axis, indices)(x)
}
