package model

import (
	randv2 "math/rand/v2"

	"github.com/itsubaki/autograd/model"
	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
)

var (
	_ model.Layer = (*L.EmbeddingsT)(nil)
	_ model.Layer = (*L.LayerNormT)(nil)
	_ model.Layer = (*L.LinearT)(nil)
)

type GPT struct {
	s randv2.Source
	model.Model
}

func (m *GPT) Forward(x *variable.Variable) *variable.Variable {
	return x
}
