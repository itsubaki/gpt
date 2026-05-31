package layer

import (
	randv2 "math/rand/v2"

	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ layer.Layer = (*EmbeddingsT)(nil)

type EmbeddingsT struct {
	s randv2.Source
	layer.Parameters
}

func (l *EmbeddingsT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *EmbeddingsT) Forward(x ...*variable.Variable) []*variable.Variable {
	return x
}
