package dataloader

import (
	"math/rand/v2"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

var (
	_ Dataset = (*TokenDataset)(nil)
	_ Dataset = (*SFTDataset)(nil)
)

type DataLoader struct {
	BatchSize int
	Dataset   Dataset
	Shuffle   bool
	indices   []int
	idx       int
}

func (l *DataLoader) Reset() {
	l.indices = make([]int, l.Dataset.Len())
	for i := range l.indices {
		l.indices[i] = i
	}

	if l.Shuffle {
		rand.Shuffle(len(l.indices), func(i, j int) {
			l.indices[i], l.indices[j] = l.indices[j], l.indices[i]
		})
	}

	l.idx = 0
}

func (l *DataLoader) Batch() (*variable.Variable, *variable.Variable) {
	if len(l.indices) == 0 {
		l.Reset()
	}

	xs := make([]*tensor.Tensor[float64], 0, l.BatchSize)
	ys := make([]*tensor.Tensor[float64], 0, l.BatchSize)
	for range l.BatchSize {
		if l.idx >= l.Dataset.Len() {
			l.Reset()
		}

		i := l.indices[l.idx]
		x, y := l.Dataset.GetItem(i)
		l.idx++

		xs = append(xs, tensor.Float64(tensor.New([]int{len(x)}, x)))
		ys = append(ys, tensor.Float64(tensor.New([]int{len(y)}, y)))
	}

	return variable.From(tensor.Stack(xs, 0)), variable.From(tensor.Stack(ys, 0))
}
