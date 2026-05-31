package layer

import (
	randv2 "math/rand/v2"

	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ layer.Layer = (*LinearT)(nil)

type LinearT struct {
	s randv2.Source
	layer.Parameters
}

func (l *LinearT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	return x
}
