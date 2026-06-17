package model

import (
	"encoding/gob"
	"fmt"
	"os"

	F "github.com/itsubaki/autograd/function"
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
	M.Model
}

func NewGPT(vocabSize, maxContextLen, embedDim, numOfHeads, numOfBlocks int) *GPT {
	gpt := &GPT{
		VocabSize:     vocabSize,
		MaxContextLen: maxContextLen,
		EmbedDim:      embedDim,
		NumOfHeads:    numOfHeads,
		NumOfBlocks:   numOfBlocks,
	}

	// Layers
	gpt.Add("embed", L.Embeddings(vocabSize, embedDim))        //
	gpt.Add("posembed", L.Embeddings(maxContextLen, embedDim)) //
	gpt.Add("norm", L.RMSNorm(embedDim))                       // instead of LayerNorm(embedDim)
	gpt.Add("unembed", L.Linear(embedDim, vocabSize, false))   // no bias in unembedding layer

	for i := range numOfBlocks {
		gpt.Add(newBlock(i, embedDim, numOfHeads))
	}

	return gpt
}

func (m *GPT) Forward(ids *variable.Variable) *variable.Variable {
	_, C := ids.Shape()[0], ids.Shape()[1]
	pos := variable.From(tensor.Arange(0, float64(C)))

	emb := m.L["embed"].First(ids)
	posemb := m.L["posembed"].First(pos)
	x := F.Add(emb, posemb)

	for i := range m.NumOfBlocks {
		x = m.L[fmt.Sprintf("block[%d]", i)].First(x)
	}

	x = m.L["norm"].First(x)
	logits := m.L["unembed"].First(x) // (B, C, VocabSize)
	return logits
}

func newBlock(i int, embedDim, numOfHeads int) (string, *L.BlockT) {
	return fmt.Sprintf("block[%d]", i), L.Block(embedDim, numOfHeads)
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
	gob.Register(&function.RoPET{})
}

func NewGPTFrom(path string) (*GPT, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	var saved *GPT
	if err := gob.NewDecoder(f).Decode(&saved); err != nil {
		return nil, err
	}

	return saved, nil
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
