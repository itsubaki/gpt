package model

import (
	"encoding/gob"
	"fmt"
	"os"

	M "github.com/itsubaki/autograd/model"
	O "github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/function"
	L "github.com/itsubaki/gpt/layer"
)

var _ O.Model = (*GPT)(nil)

var (
	_ M.Layer = (*L.MultiHeadAttentionT)(nil)
	_ M.Layer = (*L.BlockT)(nil)
	_ M.Layer = (*L.EmbeddingsT)(nil)
	_ M.Layer = (*L.FFNT)(nil)
	_ M.Layer = (*L.LayerNormT)(nil)
	_ M.Layer = (*L.LinearT)(nil)
	_ M.Layer = (*L.RMSNormT)(nil)
	_ M.Layer = (*L.SwiGLUT)(nil)
)

type GPT struct {
	VocabSize     int
	MaxContextLen int
	EmbedDim      int
	NumOfHeads    int
	NumOfBlocks   int
	Theta         float64
	M.Model
}

func NewGPT(vocabSize, maxContextLen, embedDim, numOfHeads, numOfBlocks int, theta float64, useCache ...bool) *GPT {
	gpt := &GPT{
		VocabSize:     vocabSize,
		MaxContextLen: maxContextLen,
		EmbedDim:      embedDim,
		NumOfHeads:    numOfHeads,
		NumOfBlocks:   numOfBlocks,
		Theta:         theta,
	}

	// Layers
	gpt.Add("embed", L.Embeddings(vocabSize, embedDim))      //
	gpt.Add("norm", L.RMSNorm(embedDim))                     // instead of LayerNorm(embedDim)
	gpt.Add("unembed", L.Linear(embedDim, vocabSize, false)) // no bias in unembedding layer

	rope := function.RoPE(theta, embedDim, maxContextLen)
	for i := range numOfBlocks {
		gpt.Add(newBlock(i, embedDim, numOfHeads, rope, useCache...))
	}

	return gpt
}

func (m *GPT) Forward(ids *variable.Variable) *variable.Variable {
	x := m.L["embed"].First(ids)
	for i := range m.NumOfBlocks {
		x = m.L[fmt.Sprintf("block[%d]", i)].First(x)
	}

	x = m.L["norm"].First(x)
	logits := m.L["unembed"].First(x) // (B, C, VocabSize)
	return logits
}

func (m *GPT) ClearCache() {
	for i := range m.NumOfBlocks {
		m.L[fmt.Sprintf("block[%d]", i)].(*L.BlockT).ClearCache()
	}
}

func newBlock(i int, embedDim, numOfHeads int, rope function.RoPEFunc, useCache ...bool) (string, *L.BlockT) {
	return fmt.Sprintf("block[%d]", i), L.Block(embedDim, numOfHeads, rope, useCache...)
}

func init() {
	gob.Register(&L.MultiHeadAttentionT{})
	gob.Register(&L.BlockT{})
	gob.Register(&L.EmbeddingsT{})
	gob.Register(&L.FFNT{})
	gob.Register(&L.LayerNormT{})
	gob.Register(&L.LinearT{})
	gob.Register(&L.RMSNormT{})
	gob.Register(&L.SwiGLUT{})
}

func NewGPTFrom(path string, useCache ...bool) (*GPT, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	var saved *GPT
	if err := gob.NewDecoder(f).Decode(&saved); err != nil {
		return nil, err
	}

	// restore model
	m := NewGPT(
		saved.VocabSize,
		saved.MaxContextLen,
		saved.EmbedDim,
		saved.NumOfHeads,
		saved.NumOfBlocks,
		saved.Theta,
		useCache...,
	)

	for k, v := range saved.Params() {
		if p, ok := m.Params()[k]; ok {
			p.Data = v.Data
		} else {
			panic(fmt.Sprintf("parameter %s not found in model", k))
		}
	}

	return m, nil
}

func (m *GPT) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	if err := gob.NewEncoder(f).Encode(m); err != nil {
		return fmt.Errorf("encode: %v", err)
	}

	return nil
}
