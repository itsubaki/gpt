package layer

import (
	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*FFNT)(nil)

func FFN(xdim, hiddendim int) *FFNT {
	return &FFNT{
		Layers: L.Layers{
			"l1": Linear(xdim, hiddendim, true),
			"l2": Linear(hiddendim, xdim, true),
		},
	}
}

type FFNT struct {
	L.Layers
}

func (l *FFNT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *FFNT) Forward(x ...*variable.Variable) []*variable.Variable {
	x0 := l.Layers["l1"].First(x...)
	x1 := F.GELU(x0)
	x2 := l.Layers["l2"].First(x1)
	return []*variable.Variable{x2}
}
