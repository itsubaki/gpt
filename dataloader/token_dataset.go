package dataloader

import (
	"encoding/gob"
	"os"
)

type TokenDataset struct {
	Tokens     []int
	ContextLen int
}

func (s *TokenDataset) Len() int {
	return len(s.Tokens) - s.ContextLen
}

func (s *TokenDataset) GetItem(i int) ([]int, []int) {
	x := s.Tokens[i : i+s.ContextLen]
	y := s.Tokens[i+1 : i+s.ContextLen+1]
	return x, y
}

func MustLoadTokens(path string) []int {
	return Must(LoadTokens(path))
}

func LoadTokens(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	var ids []int
	if err := gob.NewDecoder(f).Decode(&ids); err != nil {
		return nil, err
	}

	return ids, nil
}

func Must[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}

	return v
}
