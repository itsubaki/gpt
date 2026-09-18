package tokenizer

import (
	"encoding/gob"
	"fmt"
	"io"
	"iter"
)

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

func (d *DefaultDict[T]) Incr(key T, value int) {
	if _, ok := d.Dict[key]; !ok {
		d.Order = append(d.Order, key)
	}

	d.Dict[key] += value
}

func (d *DefaultDict[T]) Get(key T) (int, bool) {
	value, ok := d.Dict[key]
	return value, ok
}

func (d *DefaultDict[T]) Value(key T) int {
	return d.Dict[key]
}

func (d *DefaultDict[T]) Delete(key T) {
	delete(d.Dict, key)

	for i, k := range d.Order {
		if k == key {
			d.Order = append(d.Order[:i], d.Order[i+1:]...)
			break
		}
	}
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

func (d *DefaultDict[T]) Save(w io.Writer) error {
	if err := gob.NewEncoder(w).Encode(d); err != nil {
		return fmt.Errorf("encode: %v", err)
	}

	return nil
}

func LoadDefaultDict(r io.Reader) (*DefaultDict[Pair], error) {
	var dict DefaultDict[Pair]
	if err := gob.NewDecoder(r).Decode(&dict); err != nil {
		return nil, fmt.Errorf("decode: %v", err)
	}

	return &dict, nil
}
