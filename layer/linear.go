package layer

import (
	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*LinearT)(nil)

func Linear(xDim, hiddenDim int) *LinearT {
	p := make(L.Parameters)
	p.Add("w", initw(xDim, hiddenDim))

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
		F.Linear([]*variable.Variable{
			x[0],
			l.Parameters["w"],
		}...),
	}
}

func initw(x, y int) *variable.Variable {
	return variable.From(tensor.Normal([]int{x, y}, 0, 0.02))
}
