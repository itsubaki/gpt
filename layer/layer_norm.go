package layer

import (
	F "github.com/itsubaki/autograd/function"
	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*LayerNormT)(nil)

func LayerNorm(embeddim int) *LayerNormT {
	p := make(L.Parameters)
	p.Add("gamma", variable.Ones(embeddim))
	p.Add("beta", variable.Zeros(embeddim))

	return &LayerNormT{
		eps:        1e-5,
		Parameters: p,
	}
}

type LayerNormT struct {
	eps float64
	L.Parameters
}

func (l *LayerNormT) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *LayerNormT) Forward(x ...*variable.Variable) []*variable.Variable {
	mean := F.Mean(-1)(x[0])
	variance := F.Variance(-1)(x[0], mean)

	// normx = (x - mean) / sqrt(variance + eps)
	sub := F.Sub(x[0], mean)
	addc := F.AddC(l.eps, variance)
	sqrt := F.Pow(0.5)(addc)
	normx := F.Div(sub, sqrt)

	y := F.Add(F.Mul(l.Parameters["gamma"], normx), l.Parameters["beta"])
	return []*variable.Variable{
		y,
	}
}
