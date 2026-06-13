package function

import (
	"math"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

// RoPE implements the Rotary Position Embedding (RoPE) function.
// Higher-order derivatives are not supported in this implementation.
func RoPE(theta float64, embedDim, contextLen int) func(x ...*variable.Variable) *variable.Variable {
	cos := make([][]float64, contextLen)
	sin := make([][]float64, contextLen)

	for pos := range contextLen {
		cos[pos] = make([]float64, embedDim/2)
		sin[pos] = make([]float64, embedDim/2)

		for i := range embedDim / 2 {
			pow := float64(2*i) / float64(embedDim)
			freq := 1.0 / math.Pow(theta, pow)
			angle := float64(pos) * freq

			cos[pos][i] = math.Cos(angle)
			sin[pos][i] = math.Sin(angle)
		}
	}

	return (&variable.Function{
		Forwarder: &RoPET{
			cos: cos,
			sin: sin,
		},
	}).First
}

type RoPET struct {
	cos [][]float64
	sin [][]float64
}

func (f *RoPET) Forward(x ...*variable.Variable) []*variable.Variable {
	shape := x[0].Shape()
	B, H, C, D := shape[0], shape[1], shape[2], shape[3]

	y := tensor.ZeroLike(x[0].Data)
	for b := range B {
		for h := range H {
			for pos := range C {
				for d := 0; d < D; d += 2 {
					i := d / 2
					cos := f.cos[pos][i]
					sin := f.sin[pos][i]

					x0 := x[0].At(b, h, pos, d)
					x1 := x[0].At(b, h, pos, d+1)

					y0 := x0*cos - x1*sin
					y1 := x0*sin + x1*cos

					y.Set([]int{b, h, pos, d}, y0)
					y.Set([]int{b, h, pos, d + 1}, y1)
				}
			}
		}
	}

	return []*variable.Variable{
		variable.From(y),
	}
}

func (f *RoPET) Backward(gy ...*variable.Variable) []*variable.Variable {
	shape := gy[0].Shape()
	B, H, C, D := shape[0], shape[1], shape[2], shape[3]

	gx := tensor.ZeroLike(gy[0].Data)
	for b := range B {
		for h := range H {
			for pos := range C {
				for d := 0; d < D; d += 2 {
					i := d / 2
					cos := f.cos[pos][i]
					sin := f.sin[pos][i]

					gy0 := gy[0].At(b, h, pos, d)
					gy1 := gy[0].At(b, h, pos, d+1)

					gx0 := gy0*cos + gy1*sin
					gx1 := -gy0*sin + gy1*cos

					gx.Set([]int{b, h, pos, d}, gx0)
					gx.Set([]int{b, h, pos, d + 1}, gx1)
				}
			}
		}
	}

	return []*variable.Variable{
		variable.From(gx),
	}
}
