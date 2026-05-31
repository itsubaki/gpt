package layer

import (
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*BlockT)(nil)

type BlockT struct {
	L.Parameters
}

func (l *BlockT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *BlockT) Forward(x ...*variable.Variable) []*variable.Variable {
	return x
}
