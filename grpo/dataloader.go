package grpo

import "math/rand/v2"

type DataLoader struct {
	BatchSize int
	Dataset   *Dataset
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

func (l *DataLoader) Batch() ([]string, []string) {
	if len(l.indices) == 0 {
		l.Reset()
	}

	var prompts, gts []string
	for range l.BatchSize {
		if l.idx >= l.Dataset.Len() {
			l.Reset()
		}

		i := l.indices[l.idx]
		x, y := l.Dataset.GetItem(i)
		l.idx++

		prompts = append(prompts, x)
		gts = append(gts, y)
	}

	return prompts, gts
}
