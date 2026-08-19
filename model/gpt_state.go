package model

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/itsubaki/autograd/layer"
)

type GPTState struct {
	VocabSize     int
	MaxContextLen int
	EmbedDim      int
	NumOfHeads    int
	NumOfBlocks   int
	Theta         float64
	Params        layer.Parameters
}

func LoadGPTState(path string) (*GPTState, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	var s *GPTState
	if err := gob.NewDecoder(f).Decode(&s); err != nil {
		return nil, fmt.Errorf("decode: %v", err)
	}

	return s, nil
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
