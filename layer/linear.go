package layer

import (
	"math"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

var _ layer.Layer = (*LinearT)(nil)

func Linear(xdim, hiddendim int, bias bool) *LinearT {
	p := make(layer.Parameters)
	if bias {
		p.Add("b", variable.Zeros(1, hiddendim))
	}

	return &LinearT{
		xdim:       xdim,
		hiddendim:  hiddendim,
		Parameters: p,
	}
}

type LinearT struct {
	xdim      int
	hiddendim int
	layer.Parameters
}

func (l *LinearT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *LinearT) Forward(x ...*variable.Variable) []*variable.Variable {
	if _, ok := l.Parameters["w"]; !ok {
		l.Add("w", initw(l.xdim, l.hiddendim))
	}

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

func initw(xdim, hiddendim int) *variable.Variable {
	w := tensor.Randn([]int{xdim, hiddendim})
	xavier := 1.0 / math.Sqrt(float64(xdim))
	return variable.From(tensor.MulC(xavier, w))
}
