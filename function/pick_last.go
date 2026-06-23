package function

import (
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// PickLast returns a function that picks the last element from the input tensor along the last axis.
// Higher-order derivatives are not supported in this implementation.
func PickLast(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &PickLastT{},
	}).First(x...)
}

type PickLastT struct {
	shape  []int
	labels *tensor.Tensor[float64]
}

func (f *PickLastT) Forward(x ...*variable.Variable) []*variable.Variable {
	probs, labels := x[0].Data, x[1].Data
	f.shape, f.labels = probs.Shape, labels

	B, C := f.shape[0], f.shape[1]
	y := tensor.Zeros[float64](B, C)

	for b := range B {
		for c := range C {
			v := labels.At(b, c)
			y.Set([]int{b, c}, probs.At(b, c, int(v)))
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
			gx.Set([]int{b, c, int(v)}, gy[0].At(b, c))
		}
	}

	return []*variable.Variable{
		variable.From(gx),
		nil,
	}
}
