package dataloader

import (
	"math/rand"

	"github.com/itsubaki/autograd/variable"
)

type DataLoader struct {
	BatchSize int
	N         int
	Data      []*variable.Variable
	Label     []*variable.Variable
	Shuffle   bool
	pos       int
}

func (l *DataLoader) Batch() ([]*variable.Variable, []*variable.Variable) {
	if (l.pos == 0 || l.pos >= l.N) && l.Shuffle {
		for i := range l.N {
			j := rand.Intn(i + 1)
			l.Data[i], l.Data[j] = l.Data[j], l.Data[i]
			l.Label[i], l.Label[j] = l.Label[j], l.Label[i]
		}
	}

	if l.pos >= l.N {
		l.pos = 0
	}

	start := l.pos
	end := min(start+l.BatchSize, l.N)
	l.pos = end

	return l.Data[start:end], l.Label[start:end]
}
