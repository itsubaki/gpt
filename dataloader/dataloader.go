package dataloader

import (
	"math/rand/v2"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
)

var (
	_ Dataset = (*TokenDataset)(nil)
	_ Dataset = (*AlpacaDataset)(nil)
)

type Dataset interface {
	Len() int
	ContextLen() int
	GetItem(i int) ([]int, []int)
}

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

	var xs, ys []int
	for range l.BatchSize {
		if l.idx >= l.Dataset.Len() {
			l.Reset()
		}

		i := l.indices[l.idx]
		x, y := l.Dataset.GetItem(i)
		l.idx++

		xs, ys = append(xs, x...), append(ys, y...)
	}

	shape := []int{l.BatchSize, l.Dataset.ContextLen()} // (B, C)
	tx := tensor.Float64(tensor.New(shape, xs))
	ty := tensor.Float64(tensor.New(shape, ys))
	return variable.From(tx), variable.From(ty)
}
