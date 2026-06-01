package layer

import (
	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*RMSNormT)(nil)

func RMSNorm(embeddim int) *RMSNormT {
	p := make(L.Parameters)
	p.Add("gamma", variable.Ones(embeddim))

	return &RMSNormT{
		eps:        1e-5,
		Parameters: p,
	}
}

type RMSNormT struct {
	eps float64
	L.Parameters
}

func (l *RMSNormT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *RMSNormT) Forward(x ...*variable.Variable) []*variable.Variable {
	shape := x[0].Shape()
	last := len(shape) - 1
	shape[last] = 1

	// rms = sqrt(mean(x^2) + eps)
	// y = gamma * x / rms
	x2 := F.Pow(2)(x[0])
	ms := F.Reshape(shape...)(F.Mean(last)(x2)) // keepdims
	rms := F.Pow(0.5)(F.AddC(l.eps, ms))
	gamma := l.Parameters["gamma"]
	y := F.Mul(gamma, F.Div(x[0], rms))
	return []*variable.Variable{
		y,
	}
}
