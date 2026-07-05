package function

import "github.com/itsubaki/autograd/variable"

func Min(x ...*variable.Variable) *variable.Variable {
	return (&variable.Function{
		Forwarder: &MinT{},
	}).First(x...)
}

type MinT struct {
	mask []bool
}

func (f *MinT) Forward(x ...*variable.Variable) []*variable.Variable {
	return nil
}

func (f *MinT) Backward(gy ...*variable.Variable) []*variable.Variable {
	return nil
}
