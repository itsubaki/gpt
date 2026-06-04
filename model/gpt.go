package model

import (
	"fmt"
	"io"
	"os"

	M "github.com/itsubaki/autograd/model"
	O "github.com/itsubaki/autograd/optimizer"
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
	_ M.Layer = (*L.RoPET)(nil)
	_ M.Layer = (*L.SwiGLUT)(nil)
)

type GPT struct {
	numOfBlocks int
	writer      io.Writer
	M.Model
}

func NewGPT(vocabSize, maxContextLen, embeddim, numOfHeads, numOfBlocks, ffdim int, theta float64) *GPT {
	gpt := &GPT{
		numOfBlocks: numOfBlocks,
		writer:      os.Stdout,
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

func newBlock(i int, embeddim, numOfHeads, ffdim int, rope *L.RoPET) (string, *L.BlockT) {
	return fmt.Sprintf("block[%d]", i), L.Block(embeddim, numOfHeads, ffdim, rope)
}
