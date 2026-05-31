package layer

import (
	"math"

	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*MultiHeadAttentionT)(nil)

func MultiHeadAttention(embeddim, numOfHead, headdim int, dropoutRate float64) *MultiHeadAttentionT {
	E, H, D, bias := embeddim, numOfHead, headdim, false
	return &MultiHeadAttentionT{
		numOfHead:   numOfHead,
		headdim:     headdim,
		dropoutRate: dropoutRate,
		Layers: L.Layers{
			"Wq": Linear(E, H*D, bias),
			"Wk": Linear(E, H*D, bias),
			"Wv": Linear(E, H*D, bias),
			"Wo": Linear(H*D, E, bias),
		},
	}
}

type MultiHeadAttentionT struct {
	headdim     int
	numOfHead   int
	dropoutRate float64
	L.Layers
}

func (l *MultiHeadAttentionT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *MultiHeadAttentionT) Forward(x ...*variable.Variable) []*variable.Variable {
	v, shape := x[0], x[0].Shape()
	B, C, H, D := shape[0], shape[1], l.numOfHead, l.headdim

	Q := l.Layers["Wq"].First(v)
	K := l.Layers["Wk"].First(v)
	V := l.Layers["Wv"].First(v)

	Q = F.Transpose(0, 2, 1, 3)(F.Reshape(B, C, H, D)(Q)) // (B, H, C, D)
	K = F.Transpose(0, 2, 1, 3)(F.Reshape(B, C, H, D)(K)) // (B, H, C, D)
	V = F.Transpose(0, 2, 1, 3)(F.Reshape(B, C, H, D)(V)) // (B, H, C, D)

	Kt := F.Transpose(0, 1, 3, 2)(K)                   // (B, H, D, C)
	scores := F.MatMul(Q, Kt)                          // (B, H, C, D) @ (B, H, D, C) -> (B, H, C, C)
	scores = F.MulC(1.0/math.Sqrt(float64(D)), scores) // (B, H, C, C)

	// attention mask
	mask := tensor.Tril(tensor.Ones[float64](C, C))
	scores = F.MaskFill(mask, math.Inf(-1))(scores)

	weights := F.Softmax(-1)(scores)                  // (B, H, C, C)
	weights = F.DropoutSimple(l.dropoutRate)(weights) // (B, H, C, C)
	hidden := F.MatMul(weights, V)                    // (B, H, C, C) @ (B, H, C, D) -> (B, H, C, D)
	hidden = F.Transpose(0, 2, 1, 3)(hidden)          // (B, H, C, D) -> (B, C, H, D)
	hidden = F.Reshape(B, C, H*D)(hidden)             // (B, C, H*D)
	output := l.Layers["Wo"].First(hidden)            // (B, C, E)
	output = F.DropoutSimple(l.dropoutRate)(output)   // (B, C, E)

	return []*variable.Variable{
		output,
	}
}
