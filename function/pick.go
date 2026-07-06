package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// Pick returns a function that selects the last element from x[0] along the last axis, based on the provided labels.
func Pick(labels *tensor.Tensor[int]) func(x ...*variable.Variable) *variable.Variable {
	return func(x ...*variable.Variable) *variable.Variable {
		return (&variable.Function{
			Forwarder: &PickT{
				labels: labels,
			},
		}).First(x...)
	}
}

type PickT struct {
	shape  []int
	labels *tensor.Tensor[int]
}

func (f *PickT) Forward(x ...*variable.Variable) []*variable.Variable {
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

func (f *PickT) Backward(gy ...*variable.Variable) []*variable.Variable {
	gx := tensor.Zeros[float64](f.shape...)
	B, C := f.labels.Shape[0], f.labels.Shape[1]
	for b := range B {
		for c := range C {
			v := f.labels.At(b, c)
			gx.AddAt([]int{b, c, v}, gy[0].At(b, c))
		}
	}

	return []*variable.Variable{
		variable.From(gx),
	}
}
