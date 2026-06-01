package layer

import (
	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*LinearT)(nil)

func Linear(xdim, hiddendim int, bias bool) *LinearT {
	p := make(L.Parameters)
	p.Add("w", initw(xdim, hiddendim))
	if bias {
		p.Add("b", variable.Zeros(1, hiddendim))
	}

	return &LinearT{
		Parameters: p,
	}
}

type LinearT struct {
	L.Parameters
}

func (l *LinearT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	return []*variable.Variable{
		F.Linear(l.xparams(x[0])...),
	}
}

func (l *LinearT) xparams(x *variable.Variable) []*variable.Variable {
	xp := []*variable.Variable{x, l.Parameters["w"]}
	if b, ok := l.Parameters["b"]; ok {
		xp = append(xp, b)
	}

	return xp
}

func initw(x, y int) *variable.Variable {
	return variable.From(tensor.Randn([]int{x, y}))
}
