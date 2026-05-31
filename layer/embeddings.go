package layer

import (
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*EmbeddingsT)(nil)

func Embeddings(xdim, embeddim int) *EmbeddingsT {
	p := make(L.Parameters)
	p.Add("w", variable.From(tensor.Randn([]int{xdim, embeddim})))

	return &EmbeddingsT{
		xdim:       xdim,
		embeddim:   embeddim,
		Parameters: p,
	}
}

type EmbeddingsT struct {
	xdim     int
	embeddim int
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

	shape := append(x[0].Shape(), l.embeddim)
	y := variable.GetItem(ids, 0)(w).Reshape(shape...)
	return []*variable.Variable{
		y,
	}
}
