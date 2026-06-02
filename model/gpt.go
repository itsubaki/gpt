package model

import (
	"fmt"
	"io"
	"os"

	"github.com/itsubaki/autograd/model"
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
	_ model.Layer = (*L.RMSNormT)(nil)
	_ model.Layer = (*L.RoPET)(nil)
	_ model.Layer = (*L.SwiGLUT)(nil)
)

type GPT struct {
	numOfBlocks int
	writer      io.Writer
	model.Model
}

func NewGPT(vocabSize, maxContextLen, embeddim, numOfHeads, numOfBlocks, ffdim int, theta float64) *GPT {
	gpt := &GPT{
		numOfBlocks: numOfBlocks,
		writer:      os.Stdout,
	}

	gpt.Add("embed", L.Embeddings(vocabSize, embeddim))
	rope := L.RoPE(theta, int(embeddim/numOfHeads), maxContextLen)
	for i := range numOfBlocks {
		gpt.Add(fmt.Sprintf("block[%d]", i), L.Block(embeddim, numOfHeads, ffdim, rope))
	}

	gpt.Add("norm", L.RMSNorm(embeddim)) // instead of LayerNorm(embeddim)
	gpt.Add("unembed", L.Linear(embeddim, vocabSize, false))

	return gpt
}

func (m *GPT) Forward(ids *variable.Variable) *variable.Variable {
	bar := progress.NewProgressBar("Transformer Blocks", m.numOfBlocks, m.writer)

	x := m.L["embed"].First(ids)
	for i := range m.numOfBlocks {
		x = m.L[fmt.Sprintf("block[%d]", i)].First(x)
		bar.Update(i + 1)
	}

	x = m.L["norm"].First(x)
	logits := m.L["unembed"].First(x) // (B, C, VocabSize)
	return logits
}
