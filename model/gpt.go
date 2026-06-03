package model

import (
	"fmt"
	"io"
	"os"

	F "github.com/itsubaki/autograd/function"
	M "github.com/itsubaki/autograd/model"
	O "github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
	"github.com/itsubaki/gpt/progress"
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
	numOfBlocks int
	writer      io.Writer
	M.Model
}

func NewGPT(vocabSize, maxContextLen, embeddim, numOfHeads, numOfBlocks, ffdim int) *GPT {
	gpt := &GPT{
		numOfBlocks: numOfBlocks,
		writer:      os.Stdout,
	}

	gpt.Add("embed", L.Embeddings(vocabSize, embeddim))
	gpt.Add("posembed", L.Embeddings(maxContextLen, embeddim))
	for i := range numOfBlocks {
		gpt.Add(fmt.Sprintf("block[%d]", i), L.Block(embeddim, numOfHeads, ffdim))
	}
	gpt.Add("norm", L.RMSNorm(embeddim)) // instead of LayerNorm(embeddim)
	gpt.Add("unembed", L.Linear(embeddim, vocabSize, false))

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
	bar := progress.NewProgressBar("Transformer Blocks", m.numOfBlocks, m.writer)
	for i := range m.numOfBlocks {
		x = m.L[fmt.Sprintf("block[%d]", i)].First(x)
		bar.Update(i + 1)
	}
	x = m.L["norm"].First(x)

	// unembedding
	logits := m.L["unembed"].First(x)
	return logits
}
