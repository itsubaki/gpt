package layer

import (
	F "github.com/itsubaki/autograd/function"
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
	var indicies []int
	for _, v := range x[0].Data.Data {
		indicies = append(indicies, int(v))
	}

	w := l.Parameters["w"]
	y := F.GetItem(indicies, 0)(w)

	return []*variable.Variable{
		y,
	}
}
