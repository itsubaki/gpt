package model

import (
	"fmt"
	"iter"

	"github.com/itsubaki/autograd/layer"
	M "github.com/itsubaki/autograd/model"
	O "github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/tensor"
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
	for b := range m.Blocks() {
		x = b.First(x)
	}

	x = m.L["norm"].First(x)
	logits := m.L["unembed"].First(x) // (B, C, V)
	return logits
}

func (m *GPT) ClearCache() {
	for b := range m.Blocks() {
		b.ClearCache()
	}
}

func (m *GPT) Eval() {
	for b := range m.Blocks() {
		b.Eval()
	}
}

func (m *GPT) Train() {
	for b := range m.Blocks() {
		b.Train()
	}
}

func (m *GPT) Blocks() iter.Seq[*L.BlockT] {
	return func(yield func(*L.BlockT) bool) {
		for i := range m.NumOfBlocks {
			block := m.L[fmt.Sprintf("block[%d]", i)].(*L.BlockT)
			if !yield(block) {
				return
			}
		}
	}
}

func newBlock(i int, embedDim, numOfHeads int, rope function.RoPEFunc) (string, *L.BlockT) {
	return fmt.Sprintf("block[%d]", i), L.Block(embedDim, numOfHeads, rope)
}

func NewGPTFrom(path string) (*GPT, error) {
	s, err := LoadGPTState(path)
	if err != nil {
		return nil, fmt.Errorf("load state: %v", err)
	}

	// restore model
	m := NewGPT(
		s.VocabSize,
		s.MaxContextLen,
		s.EmbedDim,
		s.NumOfHeads,
		s.NumOfBlocks,
		s.Theta,
	)

	if err := m.Load(s.Params); err != nil {
		return nil, fmt.Errorf("load: %v", err)
	}

	return m, nil
}

func (m *GPT) Save(path string) error {
	s := &GPTState{
		VocabSize:     m.VocabSize,
		MaxContextLen: m.MaxContextLen,
		EmbedDim:      m.EmbedDim,
		NumOfHeads:    m.NumOfHeads,
		NumOfBlocks:   m.NumOfBlocks,
		Theta:         m.Theta,
		Params:        m.Params(),
	}

	if err := s.Save(path); err != nil {
		return fmt.Errorf("save state: %v", err)
	}

	return nil
}

func (m *GPT) Load(params layer.Parameters) error {
	for k, v := range params {
		if p, ok := m.Params()[k]; ok {
			p.Data = tensor.Clone(v.Data)
			continue
		}

		return fmt.Errorf("parameter %s not found in model", k)
	}

	m.ClearCache()
	return nil
}

func (m *GPT) State() *GPTState {
	return &GPTState{
		VocabSize:     m.VocabSize,
		MaxContextLen: m.MaxContextLen,
		EmbedDim:      m.EmbedDim,
		NumOfHeads:    m.NumOfHeads,
		NumOfBlocks:   m.NumOfBlocks,
		Theta:         m.Theta,
		Params:        m.Params(),
	}
}
