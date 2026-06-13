package layer

import (
	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*EmbeddingsT)(nil)

func Embeddings(xDim, embedDim int) *EmbeddingsT {
	p := make(L.Parameters)
	p.Add("w", initw(xDim, embedDim))

	return &EmbeddingsT{
		EmbedDim:   embedDim,
		Parameters: p,
	}
}

type EmbeddingsT struct {
	EmbedDim int
	L.Parameters
}

func (l *EmbeddingsT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *EmbeddingsT) Forward(x ...*variable.Variable) []*variable.Variable {
	ids := make([]int, len(x[0].Data.Data))
	for i, v := range x[0].Data.Data {
		ids[i] = int(v)
	}
	w := l.Parameters["w"]

	shape := append(append([]int{}, x[0].Shape()...), l.EmbedDim)
	y := F.Reshape(shape...)(F.GetItem(0, ids)(w))
	return []*variable.Variable{
		y,
	}
}
