package layer

import (
	"math"

	L "github.com/itsubaki/autograd/layer"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

var _ L.Layer = (*RoPET)(nil)

func RoPE(theta float64, keydim, maxContextLen int) *RoPET {
	if keydim%2 != 0 {
		panic("keydim must be even")
	}

	positions := tensor.Arange(0, maxContextLen)
	invFreq := tensor.F(tensor.Arange(0, keydim/2), func(v int) float64 {
		return 1.0 / math.Pow(theta, 2.0*float64(v)/float64(keydim))
	})

	pos2d := tensor.Expand(positions, 1) // [:, None]
	freq2d := tensor.Expand(invFreq, 0)  // [None, :]

	angles := tensor.Mul(tensor.Float64(pos2d), freq2d)
	cos := tensor.Cos(angles)
	sin := tensor.Sin(angles)

	return &RoPET{
		cos: cos,
		sin: sin,
	}
}

type RoPET struct {
	cos *tensor.Tensor[float64]
	sin *tensor.Tensor[float64]
	L.Parameters
}

func (l *RoPET) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *RoPET) Forward(x ...*variable.Variable) []*variable.Variable {
	// TODO: implement RoPE
	return nil
}
