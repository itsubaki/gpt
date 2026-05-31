package layer

import (
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ layer.Layer = (*FFNT)(nil)

func FFN(xdim int) *FFNT {
	fnn := &FFNT{
		xDim:        xdim,
		hiddenDim:   4 * xdim,
		dropoutRate: 0.1,
		Layers:      make(layer.Layers),
	}

	bias := true
	fnn.Add("l1", Linear(xdim, 4*xdim, bias))
	fnn.Add("l2", Linear(4*xdim, xdim, bias))
	return fnn
}

type FFNT struct {
	xDim        int
	hiddenDim   int
	dropoutRate float64
	layer.Layers
}

func (l *FFNT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *FFNT) Forward(x ...*variable.Variable) []*variable.Variable {
	x0 := l.Layers["l1"].First(x...)
	x1 := F.GELU(x0)
	x2 := l.Layers["l2"].First(x1)
	x3 := F.DropoutSimple(l.dropoutRate)(x2)
	return []*variable.Variable{x3}
}
