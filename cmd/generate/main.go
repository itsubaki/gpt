package main

import (
	"flag"
	"fmt"
	"math/rand/v2"
	"time"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
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
	useCache := true
	m, err := model.NewGPTFrom(modelPath, useCache)
	if err != nil {
		panic(err)
	}

	fmt.Println("model parameters:")
	fmt.Println(" VocabSize    :", m.VocabSize)
	fmt.Println(" MaxContextLen:", m.MaxContextLen)
	fmt.Println(" Embeddim     :", m.Embeddim)
	fmt.Println(" NumOfHeads   :", m.NumOfHeads)
	fmt.Println(" NumOfBlocks  :", m.NumOfBlocks)
	fmt.Println("------------------------------")

	// tokenizer
	mergeRules, err := tokenizer.Load(mergeRulesPath)
	if err != nil {
		panic(err)
	}

	tknizer := tokenizer.NewBPETokenizer(mergeRules)

	// generate text
	now := time.Now()
	generatedText := Generate(
		m,
		tknizer,
		prompt,
		maxNewTokens,
		temperature,
	)
	fmt.Println()
	fmt.Println("------------------------------")
	fmt.Println(generatedText)
	fmt.Println("------------------------------")

	fmt.Println("generation time:", time.Since(now))
}

var _ Tokenizer = (*tokenizer.BPETokenizer)(nil)

var _ Model = (*model.GPT)(nil)

type Tokenizer interface {
	Encode(text string) []int
	Decode(tokens []int) string
	EndTokenID() int
}

type Model interface {
	Forward(x *variable.Variable) *variable.Variable
	ClearCache()
}

func Generate(
	model Model,
	tokenizer Tokenizer,
	prompt string,
	maxNewTokens int,
	temperature float64,
) string {
	// encode prompt to token IDs
	ids := tokenizer.Encode(prompt)
	generatedIDs := make([]int, len(ids))
	copy(generatedIDs, ids)

	// clear KV cache before generation
	model.ClearCache()
	func() {
		// disable gradient tracking for generation
		defer variable.Nograd().End()

		// feed prompt tokens one by one to populate the KV cache
		var x *variable.Variable
		for _, id := range ids {
			x = newVariable([]int{id}).Reshape(1, 1) // (1, 1)
			x = model.Forward(x)                     // (1, 1, V)
		}

		// generate tokens
		for range maxNewTokens {
			// get logits for the next token
			logits := F.Reshape(-1)(x) // (1, 1, V) -> (V)

			// sample next token
			var nextID int
			if temperature == 0 {
				nextID = tensor.Argmax(logits.Data, 0).At()
			} else {
				probs := F.Softmax(-1)(F.MulC(1.0/temperature, logits))
				nextID = multinominal(probs)
			}
			fmt.Printf("%v,", nextID)

			// stop if end token is generated
			if nextID == tokenizer.EndTokenID() {
				break
			}
			generatedIDs = append(generatedIDs, nextID)

			// next token only
			x = newVariable([]int{nextID}).Reshape(1, 1) // (1, 1)
			x = model.Forward(x)                         // (1, 1, V)
		}
	}()

	// decode generated tokens to text
	generatedText := tokenizer.Decode(generatedIDs)
	return generatedText
}

func newVariable(x []int) *variable.Variable {
	f := make([]float64, len(x))
	for i, v := range x {
		f[i] = float64(v)
	}

	return variable.New(f...)
}

func multinominal(probs *variable.Variable) int {
	r := rand.Float64()

	var cum float64
	for i := range probs.Size() {
		cum += probs.At(i)
		if r < cum {
			return i
		}
	}

	return probs.Size() - 1
}
