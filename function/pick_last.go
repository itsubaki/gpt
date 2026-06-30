package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// PickLast returns a function that selects the last element from x[0] along the last axis, based on the provided labels.
// Higher-order derivatives are not supported in this implementation.
func PickLast(labels *tensor.Tensor[int]) func(x ...*variable.Variable) *variable.Variable {
	return func(x ...*variable.Variable) *variable.Variable {
		return (&variable.Function{
			Forwarder: &PickLastT{
				labels: labels,
			},
		}).First(x...)
	}
}

type PickLastT struct {
	shape  []int
	labels *tensor.Tensor[int]
}

func (f *PickLastT) Forward(x ...*variable.Variable) []*variable.Variable {
	f.shape = x[0].Data.Shape

	B, C := f.shape[0], f.shape[1]
	y := tensor.Zeros[float64](B, C)

	for b := range B {
		for c := range C {
			v := f.labels.At(b, c)
			y.Set([]int{b, c}, x[0].Data.At(b, c, v))
		}
	}

	return []*variable.Variable{
		variable.From(y),
	}
}

func (f *PickLastT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gx := tensor.Zeros[float64](f.shape...)
	B, C := f.labels.Shape[0], f.labels.Shape[1]
	for b := range B {
		for c := range C {
			v := f.labels.At(b, c)
			gx.Set([]int{b, c, v}, gy[0].At(b, c))
		}
	}

	return []*variable.Variable{
		variable.From(gx),
		nil,
	}
}
