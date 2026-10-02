package model

import (
	"encoding/gob"
	"fmt"
	"io"

	"github.com/itsubaki/autograd/tensor"
)

type GPTState struct {
	VocabSize     int
	MaxContextLen int
	EmbedDim      int
	NumOfHeads    int
	NumOfBlocks   int
	Theta         float32
	Params        map[string]*tensor.Tensor[float32]
}

func NewGPTStateFrom(r io.Reader) (*GPTState, error) {
	var s *GPTState
	if err := gob.NewDecoder(r).Decode(&s); err != nil {
		return nil, fmt.Errorf("decode: %v", err)
	}

	return s, nil
}

func (s *GPTState) Save(w io.Writer) error {
	if err := gob.NewEncoder(w).Encode(s); err != nil {
		return fmt.Errorf("encode: %v", err)
	}

	return nil
}
