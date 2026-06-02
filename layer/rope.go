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
	half := keydim / 2

	pos := tensor.Arange(0, maxContextLen)
	invFreq := tensor.F(tensor.Arange(0, half), func(v int) float64 {
		return 1.0 / math.Pow(theta, 2.0*float64(v)/float64(keydim))
	})

	pos2d := tensor.Expand(pos, 1)                      // (maxContextLen, 1)
	freq2d := tensor.Expand(invFreq, 0)                 // (1, half)
	angles := tensor.Mul(tensor.Float64(pos2d), freq2d) // (maxContextLen, half)

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
	v, shape := x[0], x[0].Shape()
	B, H, T, D := shape[0], shape[1], shape[2], shape[3]
	if D%2 != 0 {
		panic("keydim must be even")
	}
	half := D / 2

	strideB := H * T * D
	strideH := T * D
	strideT := D

	cos := l.cos.Data
	sin := l.sin.Data

	data := v.Data.Data
	y := make([]float64, len(data))
	for b := range B {
		baseB := b * strideB
		for h := range H {
			baseH := baseB + h*strideH
			for t := range T {
				baseT := baseH + t*strideT
				angle := t * half
				for i := range half {
					evenIdx := baseT + 2*i
					oddIdx := baseT + 2*i + 1

					even := data[evenIdx]
					odd := data[oddIdx]

					c := cos[angle+i]
					s := sin[angle+i]

					y[evenIdx] = even*c - odd*s
					y[oddIdx] = even*s + odd*c
				}
			}
		}
	}

	return []*variable.Variable{
		variable.From(tensor.New(shape, y)),
	}
}
