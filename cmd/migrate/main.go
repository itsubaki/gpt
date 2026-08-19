package main

import "github.com/itsubaki/gpt/model"

func main() {
	{
		m, err := model.NewGPTFrom("testdata/model_gpt.gob")
		if err != nil {
			panic(err)
		}

		state := &model.GPTState{
			VocabSize:     m.VocabSize,
			MaxContextLen: m.MaxContextLen,
			EmbedDim:      m.EmbedDim,
			NumOfHeads:    m.NumOfHeads,
			NumOfBlocks:   m.NumOfBlocks,
			Theta:         m.Theta,
			Params:        m.Params(),
		}

		if err := state.Save("testdata/model_gpt_state.gob"); err != nil {
			panic(err)
		}
	}

	{
		m, err := model.NewGPTFrom("testdata/model_gpt_sft.gob")
		if err != nil {
			panic(err)
		}

		state := &model.GPTState{
			VocabSize:     m.VocabSize,
			MaxContextLen: m.MaxContextLen,
			EmbedDim:      m.EmbedDim,
			NumOfHeads:    m.NumOfHeads,
			NumOfBlocks:   m.NumOfBlocks,
			Theta:         m.Theta,
			Params:        m.Params(),
		}

		if err := state.Save("testdata/model_gpt_sft_state.gob"); err != nil {
			panic(err)
		}
	}

	{
		m, err := model.NewGPTFrom("testdata/model_gpt_grpo.gob")
		if err != nil {
			panic(err)
		}

		state := &model.GPTState{
			VocabSize:     m.VocabSize,
			MaxContextLen: m.MaxContextLen,
			EmbedDim:      m.EmbedDim,
			NumOfHeads:    m.NumOfHeads,
			NumOfBlocks:   m.NumOfBlocks,
			Theta:         m.Theta,
			Params:        m.Params(),
		}

		if err := state.Save("testdata/model_gpt_grpo_state.gob"); err != nil {
			panic(err)
		}
	}
}
