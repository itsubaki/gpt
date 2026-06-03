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
	ids := tensor.Arange(0, half)
	invFreq := tensor.F(ids, func(k int) float64 {
		// 1/(theta^(2k/d))
		return 1.0 / math.Pow(theta, 2.0*float64(k)/float64(keydim))
	})

	pos2d := tensor.Expand(pos, 1)                      // (maxContextLen, 1)
	freq2d := tensor.Expand(invFreq, 0)                 // (1, half)
	angles := tensor.Mul(tensor.Float64(pos2d), freq2d) // (maxContextLen, half)

	return &RoPET{
		cos: tensor.Cos(angles).Data,
		sin: tensor.Sin(angles).Data,
	}
}

type RoPET struct {
	cos          []float64
	sin          []float64
	L.Parameters // not used
}

func (l *RoPET) First(x ...*variable.Variable) *variable.Variable {
	return l.Forward(x...)[0]
}

func (l *RoPET) Forward(x ...*variable.Variable) []*variable.Variable {
	shape, stride := x[0].Shape(), x[0].Stride()
	B, H, T, D := shape[0], shape[1], shape[2], shape[3]
	sB, sH, sT, sD := stride[0], stride[1], stride[2], stride[3]

	if D%2 != 0 {
		panic("keydim must be even")
	}
	half := D / 2

	for b := range B {
		baseB := b * sB
		for h := range H {
			baseH := baseB + h*sH
			for t := range T {
				baseT := baseH + t*sT
				angle := t * half
				for i := range half {
					c := l.cos[angle+i]
					s := l.sin[angle+i]

					evenIdx := baseT + (2*i)*sD
					oddIdx := baseT + (2*i+1)*sD

					even := x[0].Data.Data[evenIdx]
					odd := x[0].Data.Data[oddIdx]

					x[0].Data.Data[evenIdx] = even*c - odd*s
					x[0].Data.Data[oddIdx] = even*s + odd*c
				}
			}
		}
	}

	return []*variable.Variable{
		x[0],
	}
}
