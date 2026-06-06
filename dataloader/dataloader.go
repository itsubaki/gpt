package dataloader

import (
	"math/rand/v2"

	"github.com/itsubaki/autograd/variable"
)

type DataLoader struct {
	BatchSize int
	Dataset   Dataset
	Cycle     bool
	Shuffle   bool
	indices   []int
	idx       int
}

func (l *DataLoader) Batch() (*variable.Variable, *variable.Variable) {
	if len(l.indices) == 0 {
		l.indices = make([]int, l.Dataset.Len())
		for i := range l.indices {
			l.indices[i] = i
		}
	}

	if l.idx >= l.Dataset.Len() {
		if !l.Cycle {
			panic("dataset exhausted")
		}

		// reset and shuffle
		l.idx = 0
		if l.Shuffle {
			rand.Shuffle(len(l.indices), func(i, j int) {
				l.indices[i], l.indices[j] = l.indices[j], l.indices[i]
			})
		}
	}

	i := l.indices[l.idx]
	x, y := l.Dataset.GetItem(i)
	l.idx++

	return newVariable(x).Reshape(l.BatchSize, -1), newVariable(y).Reshape(l.BatchSize, -1)
}

func newVariable(x []int) *variable.Variable {
	f := make([]float64, len(x))
	for i, v := range x {
		f[i] = float64(v)
	}

	return variable.New(f...)
}
