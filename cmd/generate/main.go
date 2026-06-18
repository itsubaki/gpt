package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/itsubaki/gpt/model"
	"github.com/itsubaki/gpt/tokenizer"
)

func main() {
	var mergeRulesPath, modelPath, prompt string
	var temperature float64
	var maxNewTokens int
	flag.StringVar(&mergeRulesPath, "merge-rules-path", "testdata/merge_rules.gob", "path to the merge rules gob file")
	flag.StringVar(&modelPath, "model-path", "testdata/model_gpt.gob", "path to the model gob file")
	flag.StringVar(&prompt, "prompt", "def", "prompt for text generation")
	flag.Float64Var(&temperature, "temperature", 1.0, "temperature for sampling")
	flag.IntVar(&maxNewTokens, "max-new-tokens", 200, "maximum number of new tokens to generate")
	flag.Parse()

	// model from gob file
	m, err := model.NewGPTFrom(modelPath)
	if err != nil {
		panic(err)
	}

	fmt.Println("model parameters:")
	fmt.Println(" VocabSize    :", m.VocabSize)
	fmt.Println(" MaxContextLen:", m.MaxContextLen)
	fmt.Println(" EmbedDim     :", m.EmbedDim)
	fmt.Println(" NumOfHeads   :", m.NumOfHeads)
	fmt.Println(" NumOfBlocks  :", m.NumOfBlocks)
	fmt.Println("------------------------------")

	// tokenizer
	tknizer, err := tokenizer.NewBPETokenizerFrom(mergeRulesPath)
	if err != nil {
		panic(err)
	}

	// prompt
	fmt.Println("prompt:", prompt)
	fmt.Println(" temperature   :", temperature)
	fmt.Println(" max new tokens:", maxNewTokens)
	fmt.Println("------------------------------")

	// generate text
	now := time.Now()
	ch := model.GenerateText(
		m,
		m.MaxContextLen,
		tknizer,
		prompt,
		maxNewTokens,
		temperature,
	)

	var ids []int
	for id := range ch {
		ids = append(ids, id)
		fmt.Printf("%v,", id)
	}

	fmt.Println()
	fmt.Println("------------------------------")
	fmt.Println(tknizer.Decode(ids))
	fmt.Println("------------------------------")
	fmt.Println("generation time:", time.Since(now))
}
