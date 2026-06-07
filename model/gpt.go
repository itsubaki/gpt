package model

import (
	"encoding/gob"
	"fmt"
	"os"

	M "github.com/itsubaki/autograd/model"
	O "github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
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
	_ M.Layer = (*L.RoPET)(nil)
	_ M.Layer = (*L.SwiGLUT)(nil)
)

type GPT struct {
	VocabSize     int
	MaxContextLen int
	Embeddim      int
	NumOfHeads    int
	NumOfBlocks   int
	FFDim         int
	Theta         float64
	M.Model
}

func NewGPT(vocabSize, maxContextLen, embeddim, numOfHeads, numOfBlocks, ffdim int, theta float64) *GPT {
	gpt := &GPT{
		VocabSize:     vocabSize,
		MaxContextLen: maxContextLen,
		Embeddim:      embeddim,
		NumOfHeads:    numOfHeads,
		NumOfBlocks:   numOfBlocks,
		FFDim:         ffdim,
		Theta:         theta,
	}

	// Layers
	gpt.Add("embed", L.Embeddings(vocabSize, embeddim))
	gpt.Add("norm", L.RMSNorm(embeddim))                     // instead of LayerNorm(embeddim)
	gpt.Add("unembed", L.Linear(embeddim, vocabSize, false)) // no bias in unembedding layer

	rope := L.RoPE(theta, int(embeddim/numOfHeads), maxContextLen)
	for i := range numOfBlocks {
		gpt.Add(newBlock(i, embeddim, numOfHeads, ffdim, rope))
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

func newBlock(i int, embeddim, numOfHeads, ffdim int, rope *L.RoPET) (string, *L.BlockT) {
	return fmt.Sprintf("block[%d]", i), L.Block(embeddim, numOfHeads, ffdim, rope)
}

func init() {
	gob.Register(&L.MultiHeadAttentionT{})
	gob.Register(&L.BlockT{})
	gob.Register(&L.EmbeddingsT{})
	gob.Register(&L.FFNT{})
	gob.Register(&L.LayerNormT{})
	gob.Register(&L.LinearT{})
	gob.Register(&L.RMSNormT{})
	gob.Register(&L.RoPET{})
	gob.Register(&L.SwiGLUT{})
}

func NewGPTFrom(path string) (*GPT, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	// Decode into a temporary GPT to read the metadata and saved weights.
	// Unexported fields in layer structs (e.g. embeddim, numOfHeads, rope) are
	// not preserved by gob, so we reconstruct the model via NewGPT and then
	// copy the saved parameter values in.
	var saved GPT
	if err := gob.NewDecoder(f).Decode(&saved); err != nil {
		return nil, err
	}

	m := NewGPT(
		saved.VocabSize,
		saved.MaxContextLen,
		saved.Embeddim,
		saved.NumOfHeads,
		saved.NumOfBlocks,
		saved.FFDim,
		saved.Theta,
	)

	savedParams := saved.Params()
	for key, p := range m.Params() {
		if src, ok := savedParams[key]; ok {
			copy(p.Data.Data, src.Data.Data)
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
