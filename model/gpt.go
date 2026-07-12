package model

import (
	"encoding/gob"
	"fmt"
	"os"

	"github.com/itsubaki/autograd/layer"
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

func NewGPT(
	vocabSize int,
	maxContextLen int,
	embedDim int,
	numOfHeads int,
	numOfBlocks int,
	theta float64,
) *GPT {
	gpt := &GPT{
		VocabSize:     vocabSize,
		MaxContextLen: maxContextLen,
		EmbedDim:      embedDim,
		NumOfHeads:    numOfHeads,
		NumOfBlocks:   numOfBlocks,
		Theta:         theta,
	}

	// Layers
	gpt.Add("embed", L.Embeddings(vocabSize, embedDim)) //
	gpt.Add("norm", L.RMSNorm(embedDim))                // instead of LayerNorm(embedDim)
	gpt.Add("unembed", L.Linear(embedDim, vocabSize))   // no bias in unembedding layer

	// Transformer blocks with RoPE
	rope := function.RoPE(theta, embedDim, maxContextLen)
	for i := range numOfBlocks {
		gpt.Add(newBlock(i, embedDim, numOfHeads, rope))
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

func (m *GPT) Eval() {
	for i := range m.NumOfBlocks {
		m.L[fmt.Sprintf("block[%d]", i)].(*L.BlockT).Eval()
	}
}

func (m *GPT) Train() {
	for i := range m.NumOfBlocks {
		m.L[fmt.Sprintf("block[%d]", i)].(*L.BlockT).Train()
	}
}

func (m *GPT) Load(params layer.Parameters) error {
	for k, v := range params {
		if p, ok := m.Params()[k]; ok {
			p.Data = v.Data
			continue
		}

		return fmt.Errorf("parameter %s not found in model", k)
	}

	m.ClearCache()
	return nil
}

func newBlock(i int, embedDim, numOfHeads int, rope function.RoPEFunc) (string, *L.BlockT) {
	return fmt.Sprintf("block[%d]", i), L.Block(embedDim, numOfHeads, rope)
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

func NewGPTFrom(path string) (*GPT, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	var saved *GPT
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

	if err := m.Load(saved.Params()); err != nil {
		return nil, fmt.Errorf("load: %v", err)
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
