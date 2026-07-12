package layer

import (
	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/function"
)

var _ L.Layer = (*BlockT)(nil)

func Block(embedDim, numOfHeads int, rope function.RoPEFunc) *BlockT {
	headDim := int(embedDim / numOfHeads)
	return &BlockT{
		Layers: L.Layers{
			"norm1": RMSNorm(embedDim),                                       // instead of LayerNorm(embedDim)
			"norm2": RMSNorm(embedDim),                                       // instead of LayerNorm(embedDim)
			"attn":  MultiHeadAttention(embedDim, numOfHeads, headDim, rope), //
			"ffn":   SwiGLU(embedDim),                                        // instead of FFN(ffDim, embedDim)
		},
	}
}

type BlockT struct {
	L.Layers
}

func (l *BlockT) Params() L.Parameters {
	params := make(L.Parameters)
	for name, layer := range l.Layers {
		for k, p := range layer.Params() {
			params[name+"."+k] = p
		}
	}

	return params
}

func (l *BlockT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *BlockT) Forward(x ...*variable.Variable) []*variable.Variable {
	x0 := l.Layers["norm1"].First(x...)
	x1 := l.Layers["attn"].First(x0)
	x2 := F.Add(x[0], x1)
	x3 := l.Layers["norm2"].First(x2)
	x4 := l.Layers["ffn"].First(x3)
	x5 := F.Add(x2, x4)
	return []*variable.Variable{x5}
}

func (l *BlockT) ClearCache() {
	l.Layers["attn"].(*MultiHeadAttentionT).ClearCache()
}

func (l *BlockT) Eval() {
	l.Layers["attn"].(*MultiHeadAttentionT).Eval()
}

func (l *BlockT) Train() {
	l.Layers["attn"].(*MultiHeadAttentionT).Train()
}
