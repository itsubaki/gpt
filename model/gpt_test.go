package model_test

import (
	"fmt"
	"reflect"
	"slices"

	"github.com/itsubaki/gpt/model"
)

func ExampleGPT_Params() {
	m := model.NewGPT(1000, 256, 394, 6, 6)

	keys := make([]string, 0, len(m.Params()))
	for k := range m.Params() {
		keys = append(keys, k)
	}

	slices.Sort(keys)
	for _, k := range keys {
		fmt.Println(k)
	}

	// Output:
	// block[0].attn.Wk.w
	// block[0].attn.Wo.w
	// block[0].attn.Wq.w
	// block[0].attn.Wv.w
	// block[0].ffn.O.w
	// block[0].ffn.V.w
	// block[0].ffn.W.w
	// block[0].norm1.gamma
	// block[0].norm2.gamma
	// block[1].attn.Wk.w
	// block[1].attn.Wo.w
	// block[1].attn.Wq.w
	// block[1].attn.Wv.w
	// block[1].ffn.O.w
	// block[1].ffn.V.w
	// block[1].ffn.W.w
	// block[1].norm1.gamma
	// block[1].norm2.gamma
	// block[2].attn.Wk.w
	// block[2].attn.Wo.w
	// block[2].attn.Wq.w
	// block[2].attn.Wv.w
	// block[2].ffn.O.w
	// block[2].ffn.V.w
	// block[2].ffn.W.w
	// block[2].norm1.gamma
	// block[2].norm2.gamma
	// block[3].attn.Wk.w
	// block[3].attn.Wo.w
	// block[3].attn.Wq.w
	// block[3].attn.Wv.w
	// block[3].ffn.O.w
	// block[3].ffn.V.w
	// block[3].ffn.W.w
	// block[3].norm1.gamma
	// block[3].norm2.gamma
	// block[4].attn.Wk.w
	// block[4].attn.Wo.w
	// block[4].attn.Wq.w
	// block[4].attn.Wv.w
	// block[4].ffn.O.w
	// block[4].ffn.V.w
	// block[4].ffn.W.w
	// block[4].norm1.gamma
	// block[4].norm2.gamma
	// block[5].attn.Wk.w
	// block[5].attn.Wo.w
	// block[5].attn.Wq.w
	// block[5].attn.Wv.w
	// block[5].ffn.O.w
	// block[5].ffn.V.w
	// block[5].ffn.W.w
	// block[5].norm1.gamma
	// block[5].norm2.gamma
	// embed.w
	// norm.gamma
	// posembed.w
	// unembed.w
}

func ExampleGPT_save() {
	m0 := model.NewGPT(1000, 256, 394, 6, 6)
	if err := m0.Save("../testdata/model_gpt.gob.test"); err != nil {
		panic(err)
	}

	m1, err := model.NewGPTFrom("../testdata/model_gpt.gob.test")
	if err != nil {
		panic(err)
	}

	fmt.Println(m0.VocabSize, m0.MaxContextLen, m0.EmbedDim, m0.NumOfHeads, m0.NumOfBlocks)
	fmt.Println(m1.VocabSize, m1.MaxContextLen, m1.EmbedDim, m1.NumOfHeads, m1.NumOfBlocks)

	keys := make([]string, 0, len(m1.Params()))
	for k := range m1.Params() {
		keys = append(keys, k)
	}

	slices.Sort(keys)
	for _, k := range keys {
		if !reflect.DeepEqual(m0.Params()[k].Data.Shape, m1.Params()[k].Data.Shape) {
			panic(fmt.Sprintf("parameter %s shape not equal", k))
		}

		if !reflect.DeepEqual(m0.Params()[k].Data, m1.Params()[k].Data) {
			panic(fmt.Sprintf("parameter %s not equal", k))
		}
	}

	// Output:
	// 1000 256 394 6 6
	// 1000 256 394 6 6
}
