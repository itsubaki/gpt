package tokenizer

import "iter"

type DefaultDict[T comparable] struct {
	Dict  map[T]int
	Order []T
}

func NewDefaultDict[T comparable]() *DefaultDict[T] {
	return &DefaultDict[T]{
		Dict:  make(map[T]int),
		Order: make([]T, 0),
	}
}

func (d *DefaultDict[T]) Len() int {
	return len(d.Dict)
}

func (d *DefaultDict[T]) Set(key T, value int) {
	if _, ok := d.Dict[key]; !ok {
		d.Order = append(d.Order, key)
	}

	d.Dict[key] = value
}

func (d *DefaultDict[T]) Seq2() iter.Seq2[T, int] {
	return func(yield func(T, int) bool) {
		for _, key := range d.Order {
			if !yield(key, d.Dict[key]) {
				return
			}
		}
	}
}
