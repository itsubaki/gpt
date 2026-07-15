package dataloader

import (
	"encoding/gob"
	"os"
)

type TokenDataset struct {
	tokens     []int
	contextLen int
}

func NewTokenDataset(tokens []int, contextLen int) *TokenDataset {
	return &TokenDataset{
		tokens:     tokens,
		contextLen: contextLen,
	}
}

func (s *TokenDataset) Len() int {
	return len(s.tokens) - s.contextLen
}

func (s *TokenDataset) ContextLen() int {
	return s.contextLen
}

func (s *TokenDataset) GetItem(i int) ([]int, []int) {
	x := s.tokens[i : i+s.contextLen]
	y := s.tokens[i+1 : i+s.contextLen+1]
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
