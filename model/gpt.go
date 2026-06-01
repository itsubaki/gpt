package model

import (
	"fmt"
	"os"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
	"github.com/itsubaki/gpt/progress"
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
	numOfBlock int
	model.Model
}

func NewGPT(vocabSize, maxContextLen, embeddim, numOfHead, numOfBlock, ffdim int) *GPT {
	gpt := &GPT{
		numOfBlock: numOfBlock,
	}

	gpt.Add("embed", L.Embeddings(vocabSize, embeddim))
	gpt.Add("posembed", L.Embeddings(maxContextLen, embeddim))
	for i := range numOfBlock {
		gpt.Add(fmt.Sprintf("block[%d]", i), L.Block(embeddim, numOfHead, ffdim))
	}
	gpt.Add("norm", L.LayerNorm(embeddim))
	gpt.Add("unembed", L.Linear(embeddim, vocabSize, true))

	return gpt
}

func (m *GPT) Forward(ids *variable.Variable) *variable.Variable {
	_, C := ids.Shape()[0], ids.Shape()[1]
	pos := variable.From(tensor.Arange(0, float64(C)))

	// embeddings
	emb := m.L["embed"].First(ids)
	posemb := m.L["posembed"].First(pos)

	// pos encoding
	x := F.Add(emb, posemb)

	// blocks
	bar := progress.NewProgressBar("Transformer Blocks", m.numOfBlock, os.Stdout)
	for i := range m.numOfBlock {
		x = m.L[fmt.Sprintf("block[%d]", i)].First(x)
		bar.Update(i + 1)
	}
	x = m.L["norm"].First(x)

	// unembedding
	logits := m.L["unembed"].First(x)
	return logits
}
