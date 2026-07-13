package main

import (
	"flag"
	"fmt"
	"regexp"

	"github.com/itsubaki/gpt/cmd/grpo/grpo"
	"github.com/itsubaki/gpt/model"
	"github.com/itsubaki/gpt/tokenizer"
)

var re = regexp.MustCompile(`(?s)### Instruction:\s*([^\n]+)\s*### Response:\s*([^\n]+)`)

func main() {
	var mergeRulesPath, modelPath string
	var temperature float64
	var maxNewTokens, batchSize int
	flag.StringVar(&mergeRulesPath, "merge-rules-path", "testdata/merge_rules.gob", "path to the merge rules gob file")
	flag.StringVar(&modelPath, "model-path", "testdata/model_gpt_grpo.gob", "path to the model gob file")
	flag.Float64Var(&temperature, "temperature", 1.0, "temperature for sampling")
	flag.IntVar(&maxNewTokens, "max-new-tokens", 256, "maximum number of new tokens to generate")
	flag.IntVar(&batchSize, "batch-size", 32, "size of each batch")
	flag.Parse()

	// model from gob file
	m, err := model.NewGPTFrom(modelPath)
	if err != nil {
		panic(err)
	}
	m.Eval()

	// tokenizer
	tknizer, err := tokenizer.NewBPETokenizerFrom(mergeRulesPath)
	if err != nil {
		panic(err)
	}

	// dataset and dataloader
	dataset := grpo.NewDataset(tknizer)
	dataloader := &grpo.DataLoader{
		BatchSize: batchSize,
		Shuffle:   true,
		Dataset:   dataset,
	}

	fmt.Println("model parameters:")
	fmt.Println(" VocabSize    :", m.VocabSize)
	fmt.Println(" MaxContextLen:", m.MaxContextLen)
	fmt.Println(" EmbedDim     :", m.EmbedDim)
	fmt.Println(" NumOfHeads   :", m.NumOfHeads)
	fmt.Println(" NumOfBlocks  :", m.NumOfBlocks)
	fmt.Println("------------------------------")
	fmt.Println(" temperature   :", temperature)
	fmt.Println(" max new tokens:", maxNewTokens)
	fmt.Println(" batch size     :", batchSize)
	fmt.Println("------------------------------")

	var correct int
	prompts, gts := dataloader.Batch()
	for i, prompt := range prompts {
		response := model.GenerateText(
			m,
			m.MaxContextLen,
			tknizer,
			prompt,
			maxNewTokens,
			temperature,
		)

		matched := re.FindStringSubmatch(response)
		fmt.Printf("%-6s %v\n",
			matched[1]+matched[2],
			matched[2] == gts[i],
		)

		if matched[2] == gts[i] {
			correct++
		}
	}

	fmt.Println("accuracy:", float64(correct)/float64(batchSize)*100, "%")
}
