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
	var maxNewTokens, count int
	flag.StringVar(&mergeRulesPath, "merge-rules-path", "testdata/merge_rules.gob", "path to the merge rules gob file")
	flag.StringVar(&modelPath, "model-path", "testdata/model_gpt.gob", "path to the model gob file")
	flag.StringVar(&prompt, "prompt", "def", "prompt for text generation")
	flag.Float64Var(&temperature, "temperature", 1.0, "temperature for sampling")
	flag.IntVar(&maxNewTokens, "max-new-tokens", 256, "maximum number of new tokens to generate")
	flag.IntVar(&count, "count", 1, "number of times to generate text")
	flag.Parse()

	// model from gob file
	m, err := model.NewGPTFrom(modelPath)
	if err != nil {
		panic(err)
	}

	// tokenizer
	tknizer, err := tokenizer.NewBPETokenizerFrom(mergeRulesPath)
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
	fmt.Println("prompt:", prompt)
	fmt.Println(" temperature   :", temperature)
	fmt.Println(" max new tokens:", maxNewTokens)
	fmt.Println("------------------------------")

	for range count {
		// generate text
		now := time.Now()
		ch := model.GenerateChan(
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
		fmt.Println("generation time:", time.Since(now))
		fmt.Println("------------------------------")
		fmt.Println(tknizer.Decode(ids))
		fmt.Println("------------------------------")
	}
}
