package model_test

import (
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"slices"

	"github.com/itsubaki/gpt/model"
)

func ExampleGPT_Params() {
	m := model.NewGPT(1000, 256, 394, 6, 6, 10000)

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
	// unembed.w
}

func ExampleGPT_save() {
	dir, err := os.MkdirTemp("", "ExampleGPT_save")
	if err != nil {
		panic(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "model_gpt.gob")

	m0 := model.NewGPT(1000, 256, 394, 6, 6, 10000)
	if err := m0.Save(path); err != nil {
		panic(err)
	}

	m1, err := model.NewGPTFrom(path)
	if err != nil {
		panic(err)
	}

	fmt.Println(m0.VocabSize, m0.MaxContextLen, m0.EmbedDim, m0.NumOfHeads, m0.NumOfBlocks, m0.Theta)
	fmt.Println(m1.VocabSize, m1.MaxContextLen, m1.EmbedDim, m1.NumOfHeads, m1.NumOfBlocks, m1.Theta)

	keys := make([]string, 0, len(m1.Params()))
	for k := range m1.Params() {
		keys = append(keys, k)
	}

	slices.Sort(keys)
	for _, k := range keys {
		fmt.Println(k, m0.Params()[k].Data.Shape)
		fmt.Println(k, m1.Params()[k].Data.Shape)

		if !reflect.DeepEqual(m0.Params()[k].Data, m1.Params()[k].Data) {
			panic(fmt.Sprintf("parameter %s not equal", k))
		}
	}

	// Output:
	// 1000 256 394 6 6 10000
	// 1000 256 394 6 6 10000
	// block[0].attn.Wk.w [394 390]
	// block[0].attn.Wk.w [394 390]
	// block[0].attn.Wo.w [390 394]
	// block[0].attn.Wo.w [390 394]
	// block[0].attn.Wq.w [394 390]
	// block[0].attn.Wq.w [394 390]
	// block[0].attn.Wv.w [394 390]
	// block[0].attn.Wv.w [394 390]
	// block[0].ffn.O.w [1050 394]
	// block[0].ffn.O.w [1050 394]
	// block[0].ffn.V.w [394 1050]
	// block[0].ffn.V.w [394 1050]
	// block[0].ffn.W.w [394 1050]
	// block[0].ffn.W.w [394 1050]
	// block[0].norm1.gamma [394]
	// block[0].norm1.gamma [394]
	// block[0].norm2.gamma [394]
	// block[0].norm2.gamma [394]
	// block[1].attn.Wk.w [394 390]
	// block[1].attn.Wk.w [394 390]
	// block[1].attn.Wo.w [390 394]
	// block[1].attn.Wo.w [390 394]
	// block[1].attn.Wq.w [394 390]
	// block[1].attn.Wq.w [394 390]
	// block[1].attn.Wv.w [394 390]
	// block[1].attn.Wv.w [394 390]
	// block[1].ffn.O.w [1050 394]
	// block[1].ffn.O.w [1050 394]
	// block[1].ffn.V.w [394 1050]
	// block[1].ffn.V.w [394 1050]
	// block[1].ffn.W.w [394 1050]
	// block[1].ffn.W.w [394 1050]
	// block[1].norm1.gamma [394]
	// block[1].norm1.gamma [394]
	// block[1].norm2.gamma [394]
	// block[1].norm2.gamma [394]
	// block[2].attn.Wk.w [394 390]
	// block[2].attn.Wk.w [394 390]
	// block[2].attn.Wo.w [390 394]
	// block[2].attn.Wo.w [390 394]
	// block[2].attn.Wq.w [394 390]
	// block[2].attn.Wq.w [394 390]
	// block[2].attn.Wv.w [394 390]
	// block[2].attn.Wv.w [394 390]
	// block[2].ffn.O.w [1050 394]
	// block[2].ffn.O.w [1050 394]
	// block[2].ffn.V.w [394 1050]
	// block[2].ffn.V.w [394 1050]
	// block[2].ffn.W.w [394 1050]
	// block[2].ffn.W.w [394 1050]
	// block[2].norm1.gamma [394]
	// block[2].norm1.gamma [394]
	// block[2].norm2.gamma [394]
	// block[2].norm2.gamma [394]
	// block[3].attn.Wk.w [394 390]
	// block[3].attn.Wk.w [394 390]
	// block[3].attn.Wo.w [390 394]
	// block[3].attn.Wo.w [390 394]
	// block[3].attn.Wq.w [394 390]
	// block[3].attn.Wq.w [394 390]
	// block[3].attn.Wv.w [394 390]
	// block[3].attn.Wv.w [394 390]
	// block[3].ffn.O.w [1050 394]
	// block[3].ffn.O.w [1050 394]
	// block[3].ffn.V.w [394 1050]
	// block[3].ffn.V.w [394 1050]
	// block[3].ffn.W.w [394 1050]
	// block[3].ffn.W.w [394 1050]
	// block[3].norm1.gamma [394]
	// block[3].norm1.gamma [394]
	// block[3].norm2.gamma [394]
	// block[3].norm2.gamma [394]
	// block[4].attn.Wk.w [394 390]
	// block[4].attn.Wk.w [394 390]
	// block[4].attn.Wo.w [390 394]
	// block[4].attn.Wo.w [390 394]
	// block[4].attn.Wq.w [394 390]
	// block[4].attn.Wq.w [394 390]
	// block[4].attn.Wv.w [394 390]
	// block[4].attn.Wv.w [394 390]
	// block[4].ffn.O.w [1050 394]
	// block[4].ffn.O.w [1050 394]
	// block[4].ffn.V.w [394 1050]
	// block[4].ffn.V.w [394 1050]
	// block[4].ffn.W.w [394 1050]
	// block[4].ffn.W.w [394 1050]
	// block[4].norm1.gamma [394]
	// block[4].norm1.gamma [394]
	// block[4].norm2.gamma [394]
	// block[4].norm2.gamma [394]
	// block[5].attn.Wk.w [394 390]
	// block[5].attn.Wk.w [394 390]
	// block[5].attn.Wo.w [390 394]
	// block[5].attn.Wo.w [390 394]
	// block[5].attn.Wq.w [394 390]
	// block[5].attn.Wq.w [394 390]
	// block[5].attn.Wv.w [394 390]
	// block[5].attn.Wv.w [394 390]
	// block[5].ffn.O.w [1050 394]
	// block[5].ffn.O.w [1050 394]
	// block[5].ffn.V.w [394 1050]
	// block[5].ffn.V.w [394 1050]
	// block[5].ffn.W.w [394 1050]
	// block[5].ffn.W.w [394 1050]
	// block[5].norm1.gamma [394]
	// block[5].norm1.gamma [394]
	// block[5].norm2.gamma [394]
	// block[5].norm2.gamma [394]
	// embed.w [1000 394]
	// embed.w [1000 394]
	// norm.gamma [394]
	// norm.gamma [394]
	// unembed.w [394 1000]
	// unembed.w [394 1000]
}
