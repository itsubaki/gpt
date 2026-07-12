package grpo

import (
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/function"
)

type Model interface {
	Forward(x *variable.Variable) *variable.Variable
	ClearCache()
}

func Loss(
	model Model,
	oldModel Model,
	ids *variable.Variable,
	mask *variable.Variable,
	advantages []float64,
	epsilon float64,
) *variable.Variable {
	probs := ComputeProbs(model, ids)

	var oldProbs *variable.Variable
	func() {
		defer variable.Nograd().End()
		oldProbs = ComputeProbs(oldModel, ids)
	}()

	ratio := F.Div(probs, F.AddC(1e-8, oldProbs))              // props / (oldProps + 1e-8)
	adv := F.Unsqueeze(-1)(variable.New(advantages...))        //
	unclipped := F.Mul(ratio, adv)                             // ratio * adv
	clipped := F.Mul(F.Clip(1-epsilon, 1+epsilon)(ratio), adv) // clip(ratio, 1-epsilon, 1+epsilon) * adv

	masks := slice(mask, 1, 1, mask.Shape()[1])                      // (B, C-1)
	tokenObjective := F.Mul(masks, function.Min(unclipped, clipped)) // masks * min(unclipped, clipped)
	sum := F.Sum()(tokenObjective)

	samples := float64(ids.Shape()[0])              //
	return F.Neg(F.Div(sum, variable.New(samples))) // -1 * sum / samples
}

func ComputeProbs(model Model, ids *variable.Variable) *variable.Variable {
	logits := model.Forward(ids)                         // (B, C, V)
	logits = slice(logits, 1, 0, logits.Shape()[1]-1)    // (B, C-1, V)
	probs := F.Softmax(-1)(logits)                       // (B, C-1, V)
	labels := slice(ids, 1, 1, ids.Shape()[1])           // (B, C-1)
	return function.Pick(tensor.Int(labels.Data))(probs) // (B, C-1)
}

func slice(x *variable.Variable, axis, start, end int) *variable.Variable {
	indices := make([]int, end-start)
	for i := range indices {
		indices[i] = start + i
	}

	return variable.GetItem(axis, indices)(x)
}
