package layer

import (
	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*BlockT)(nil)

func Block(embeddim, numOfHead, ffdim int, dropoutRate float64) *BlockT {
	headdim := int(embeddim / numOfHead)
	return &BlockT{
		norm1: LayerNorm(embeddim),
		norm2: LayerNorm(embeddim),
		attn:  MultiHeadAttention(embeddim, numOfHead, headdim, dropoutRate),
		ffn:   FFN(embeddim, ffdim, dropoutRate),
	}
}

type BlockT struct {
	norm1 *LayerNormT
	norm2 *LayerNormT
	attn  *MultiHeadAttentionT
	ffn   *FFNT
	L.Parameters
}

func (l *BlockT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *BlockT) Forward(x ...*variable.Variable) []*variable.Variable {
	x0 := l.norm1.First(x...)
	x1 := l.attn.First(x0)
	x2 := F.Add(x[0], x1)
	x3 := l.norm2.First(x2)
	x4 := l.ffn.First(x3)
	x5 := F.Add(x[0], x4)
	return []*variable.Variable{x5}
}
