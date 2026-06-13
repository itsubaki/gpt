package layer

import (
	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*SwiGLUT)(nil)

func SwiGLU(xDim int) *SwiGLUT {
	hiddenDim := int(xDim * 8 / 3)
	return &SwiGLUT{
		Layers: L.Layers{
			"W": Linear(xDim, hiddenDim, false),
			"V": Linear(xDim, hiddenDim, false),
			"O": Linear(hiddenDim, xDim, false),
		},
	}
}

type SwiGLUT struct {
	L.Layers
}

func (l *SwiGLUT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *SwiGLUT) Forward(x ...*variable.Variable) []*variable.Variable {
	a := l.Layers["W"].First(x...)
	b := l.Layers["V"].First(x...)

	gated := F.Mul(F.Mul(a, F.Sigmoid(a)), b)
	o := l.Layers["O"].First(gated)
	return []*variable.Variable{
		o,
	}
}
