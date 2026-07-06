package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// Min returns a function that computes the element-wise minimum of two variables.
func Min(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &MinT{},
	}).First(x...)
}

type MinT struct {
	mask *tensor.Tensor[float64]
}

func (f *MinT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.mask = tensor.F2(x[0].Data, x[1].Data, mask)

	y := tensor.F2(x[0].Data, x[1].Data, min)
	return []*variable.Variable{
		variable.From(y),
	}
}

func (f *MinT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gx0 := tensor.Mul(gy[0].Data, f.mask)
	gx1 := tensor.Mul(gy[0].Data, tensor.SubC(1, f.mask))

	return []*variable.Variable{
		variable.From(gx0),
		variable.From(gx1),
	}
}

func mask(a, b float64) float64 {
	if a <= b {
		return 1
	}

	return 0
}

func min(a, b float64) float64 {
	if a <= b {
		return a
	}

	return b
}
