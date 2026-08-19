package model

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/itsubaki/autograd/layer"
)

func NewGPTFromState(path string) (*GPT, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	var saved *GPTState
	if err := gob.NewDecoder(f).Decode(&saved); err != nil {
		return nil, fmt.Errorf("decode: %v", err)
	}

	// restore model
	m := NewGPT(
		saved.VocabSize,
		saved.MaxContextLen,
		saved.EmbedDim,
		saved.NumOfHeads,
		saved.NumOfBlocks,
		saved.Theta,
	)

	if err := m.Load(saved.Params); err != nil {
		return nil, fmt.Errorf("load: %v", err)
	}

	return m, nil
}

type GPTState struct {
	VocabSize     int
	MaxContextLen int
	EmbedDim      int
	NumOfHeads    int
	NumOfBlocks   int
	Theta         float64
	Params        layer.Parameters
}

func (s *GPTState) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	if err := gob.NewEncoder(f).Encode(s); err != nil {
		return fmt.Errorf("encode: %v", err)
	}

	return nil
}
