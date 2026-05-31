package tokenizer

import "iter"

type DefaultDict[T comparable, U any] struct {
	Dict  map[T]U
	Order []T
}

func NewDefaultDict[T comparable, U any]() *DefaultDict[T, U] {
	return &DefaultDict[T, U]{
		Dict:  make(map[T]U),
		Order: make([]T, 0),
	}
}

func (d *DefaultDict[T, U]) Len() int {
	return len(d.Dict)
}

func (d *DefaultDict[T, U]) Set(key T, value U) {
	if _, ok := d.Dict[key]; !ok {
		d.Order = append(d.Order, key)
	}

	d.Dict[key] = value
}

func (d *DefaultDict[T, U]) Delete(key T) {
	delete(d.Dict, key)

	for i, k := range d.Order {
		if k == key {
			d.Order = append(d.Order[:i], d.Order[i+1:]...)
			break
		}
	}
}

func (d *DefaultDict[T, U]) Seq2() iter.Seq2[T, U] {
	return func(yield func(T, U) bool) {
		for _, key := range d.Order {
			if !yield(key, d.Dict[key]) {
				return
			}
		}
	}
}
