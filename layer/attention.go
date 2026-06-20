package layer

import (
	"math"

	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/function"
)

var _ L.Layer = (*MultiHeadAttentionT)(nil)

func MultiHeadAttention(embedDim, numOfHeads, headDim int, rope function.RoPEFunc, useCache ...bool) *MultiHeadAttentionT {
	E, H, D := embedDim, numOfHeads, headDim
	return &MultiHeadAttentionT{
		numOfHeads: numOfHeads,
		headDim:    headDim,
		rope:       rope,
		useCache:   len(useCache) > 0 && useCache[0],
		Layers: L.Layers{
			"Wq": Linear(E, H*D),
			"Wk": Linear(E, H*D),
			"Wv": Linear(E, H*D),
			"Wo": Linear(H*D, E),
		},
	}
}

type MultiHeadAttentionT struct {
	numOfHeads int
	headDim    int
	rope       function.RoPEFunc
	offset     int
	useCache   bool
	kCache     *variable.Variable
	vCache     *variable.Variable
	L.Layers
}

func (l *MultiHeadAttentionT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *MultiHeadAttentionT) Forward(x ...*variable.Variable) []*variable.Variable {
	v, shape := x[0], x[0].Shape()
	B, C, H, D := shape[0], shape[1], l.numOfHeads, l.headDim

	Q := l.Layers["Wq"].First(v)
	K := l.Layers["Wk"].First(v)
	V := l.Layers["Wv"].First(v)

	Q = F.Transpose(0, 2, 1, 3)(F.Reshape(B, C, H, D)(Q)) // (B, H, C, D)
	K = F.Transpose(0, 2, 1, 3)(F.Reshape(B, C, H, D)(K)) // (B, H, C, D)
	V = F.Transpose(0, 2, 1, 3)(F.Reshape(B, C, H, D)(V)) // (B, H, C, D)

	// RoPE
	Q = l.rope(l.offset)(Q)
	K = l.rope(l.offset)(K)

	// cache
	isFirstCall := l.kCache == nil
	if l.useCache {
		if isFirstCall {
			// first call, initialize cache
			l.kCache = K
			l.vCache = V
		} else {
			// subsequent calls, append to cache
			l.kCache = F.Concat(2)(l.kCache, K) // (B, H, C+cache, D)
			l.vCache = F.Concat(2)(l.vCache, V) // (B, H, C+cache, D)
		}

		// use cache
		K = l.kCache
		V = l.vCache

		// update offset
		l.offset += C
	}

	// QK^t/sqrt(d)
	Kt := F.Transpose(0, 1, 3, 2)(K)                   // (B, H, D, C)
	scores := F.MatMul(Q, Kt)                          // (B, H, C, D) @ (B, H, D, C) -> (B, H, C, C)
	scores = F.MulC(1.0/math.Sqrt(float64(D)), scores) // (B, H, C, C)

	// attention mask
	if !l.useCache || isFirstCall {
		mask := tensor.Tril(tensor.Ones[float64](C, C))
		cond := func(m float64) bool { return m == 0 }
		scores = F.MaskFill(mask, cond, math.Inf(-1))(scores) // (B, H, C, C)
	}

	// (softmax(QK^t/sqrt(d))V)Wo
	weights := F.Softmax(-1)(scores)         // (B, H, C, C)
	hidden := F.MatMul(weights, V)           // (B, H, C, C) @ (B, H, C, D) -> (B, H, C, D)
	hidden = F.Transpose(0, 2, 1, 3)(hidden) // (B, H, C, D) -> (B, C, H, D)
	hidden = F.Reshape(B, C, H*D)(hidden)    // (B, C, H*D)
	output := l.Layers["Wo"].First(hidden)   // (B, C, E)

	return []*variable.Variable{
		output,
	}
}

func (l *MultiHeadAttentionT) ClearCache() {
	l.kCache = nil
	l.vCache = nil
	l.offset = 0
}
