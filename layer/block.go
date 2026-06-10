package layer

import (
	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*BlockT)(nil)

func Block(embeddim, numOfHeads int, useCache ...bool) *BlockT {
	headdim := int(embeddim / numOfHeads)
	return &BlockT{
		Layers: L.Layers{
			"norm1": RMSNorm(embeddim),                                              // instead of LayerNorm(embeddim)
			"norm2": RMSNorm(embeddim),                                              // instead of LayerNorm(embeddim)
			"attn":  MultiHeadAttention(embeddim, numOfHeads, headdim, useCache...), //
			"ffn":   SwiGLU(embeddim),                                               // instead of FFN(ffdim, embeddim)
		},
	}
}

type BlockT struct {
	L.Layers
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
