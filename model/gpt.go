package model

import (
	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
)

var (
	_ model.Layer = (*L.MultiHeadAttentionT)(nil)
	_ model.Layer = (*L.BlockT)(nil)
	_ model.Layer = (*L.EmbeddingsT)(nil)
	_ model.Layer = (*L.FFNT)(nil)
	_ model.Layer = (*L.LayerNormT)(nil)
	_ model.Layer = (*L.LinearT)(nil)
)

type GPT struct {
	vocabSize     int
	maxContextLen int
	embeddim      int
	numOfHead     int
	numOfBlock    int
	ffdim         int
	dropoutRate   float64
	embed         *L.EmbeddingsT
	posembed      *L.EmbeddingsT
	blocks        []*L.BlockT
	norm          *L.LayerNormT
	unembed       *L.LinearT
	model.Model
}

func NewGPT(vocabSize, maxContextLen, embeddim, numOfHead, numOfBlock, ffdim int, dropoutRate float64) *GPT {
	blocks := make([]*L.BlockT, numOfBlock)
	for i := range blocks {
		blocks[i] = L.Block(embeddim, numOfHead, ffdim, dropoutRate)
	}

	return &GPT{
		vocabSize:     vocabSize,
		maxContextLen: maxContextLen,
		embeddim:      embeddim,
		numOfHead:     numOfHead,
		numOfBlock:    numOfBlock,
		ffdim:         ffdim,
		dropoutRate:   dropoutRate,
		embed:         L.Embeddings(vocabSize, embeddim),
		posembed:      L.Embeddings(maxContextLen, embeddim),
		blocks:        blocks,
		norm:          L.LayerNorm(embeddim),
		unembed:       L.Linear(embeddim, vocabSize, true),
	}
}

func (m *GPT) Forward(ids *variable.Variable) *variable.Variable {
	_, C := ids.Shape()[0], ids.Shape()[1]

	// embeddings
	pos := variable.From(tensor.Arange(0, float64(C)))
	emb := m.embed.First(ids)
	posemb := m.posembed.First(pos)

	// pos encoding
	x := F.DropoutSimple(m.dropoutRate)(F.Add(emb, posemb))

	// blocks
	for _, block := range m.blocks {
		x = block.First(x)
	}
	x = m.norm.First(x)

	// unembedding
	logits := m.unembed.First(x)
	return logits
}
