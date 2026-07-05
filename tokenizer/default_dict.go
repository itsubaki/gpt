package tokenizer

import (
	"encoding/gob"
	"fmt"
	"iter"
	"os"
)

type DefaultDict[T comparable] struct {
	dict  map[T]int
	order []T
}

func NewDefaultDict[T comparable]() *DefaultDict[T] {
	return &DefaultDict[T]{
		dict:  make(map[T]int),
		order: make([]T, 0),
	}
}

func (d *DefaultDict[T]) Len() int {
	return len(d.dict)
}

func (d *DefaultDict[T]) Set(key T, value int) {
	if _, ok := d.dict[key]; !ok {
		d.order = append(d.order, key)
	}

	d.dict[key] = value
}

func (d *DefaultDict[T]) Incr(key T, value int) {
	if _, ok := d.dict[key]; !ok {
		d.order = append(d.order, key)
	}

	d.dict[key] += value
}

func (d *DefaultDict[T]) Get(key T) (int, bool) {
	value, ok := d.dict[key]
	return value, ok
}

func (d *DefaultDict[T]) Value(key T) int {
	return d.dict[key]
}

func (d *DefaultDict[T]) Delete(key T) {
	delete(d.dict, key)

	for i, k := range d.order {
		if k == key {
			d.order = append(d.order[:i], d.order[i+1:]...)
			break
		}
	}
}

func (d *DefaultDict[T]) Seq2() iter.Seq2[T, int] {
	return func(yield func(T, int) bool) {
		for _, key := range d.order {
			if !yield(key, d.dict[key]) {
				return
			}
		}
	}
}

func Save(path string, dict *DefaultDict[Pair]) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	if err := gob.NewEncoder(f).Encode(dict); err != nil {
		return fmt.Errorf("encode: %v", err)
	}

	return nil
}

func Load(path string) (*DefaultDict[Pair], error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %v", err)
	}
	defer func() { _ = f.Close() }()

	var dict DefaultDict[Pair]
	if err := gob.NewDecoder(f).Decode(&dict); err != nil {
		return nil, fmt.Errorf("decode: %v", err)
	}

	return &dict, nil
}
