package layer

import (
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ layer.Layer = (*BlockT)(nil)

type BlockT struct {
	layer.Parameters
}

func (l *BlockT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *BlockT) Forward(x ...*variable.Variable) []*variable.Variable {
	return x
}
