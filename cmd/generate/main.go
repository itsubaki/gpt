package main

import (
	"flag"
	"fmt"
	"math/rand/v2"

	F "github.com/itsubaki/autograd/function"
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

	m, err := model.NewGPTFrom(modelPath)
	if err != nil {
		panic(fmt.Errorf("new model from %q: %v", modelPath, err))
	}

	fmt.Println("model parameters:")
	fmt.Println(" VocaSize     :", m.VocabSize)
	fmt.Println(" MaxContextLen:", m.MaxContextLen)
	fmt.Println(" Embeddim     :", m.Embeddim)
	fmt.Println(" NumOfHeads   :", m.NumOfHeads)
	fmt.Println(" NumOfBlocks  :", m.NumOfBlocks)
	fmt.Println(" FFDim        :", m.FFDim)
	fmt.Println(" Theta        :", m.Theta)

	// tokenizer
	mergeRules, ok := tokenizer.Load(mergeRulesPath)
	if !ok {
		panic("failed to load merge rules")
	}

	// generate text
	generatedText := Generate(
		m,
		m.MaxContextLen,
		tokenizer.NewBPETokenizer(mergeRules),
		prompt,
		maxNewTokens,
		temperature,
	)

	fmt.Println("generated text:")
	fmt.Println(generatedText)
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
}

func Generate(
	model Model,
	maxConextLen int,
	tokenizer Tokenizer,
	prompt string,
	maxNewTokens int,
	temperature float64,
) string {
	ids := tokenizer.Encode(prompt)
	generatedIDs := make([]int, len(ids))
	copy(generatedIDs, ids)

	func() {
		// disable gradient tracking for generation
		defer variable.Nograd().End()

		// generate tokens
		for range maxNewTokens {
			if len(ids) > maxConextLen {
				// keep only the last maxContextLen tokens as input
				ids = ids[len(ids)-maxConextLen:]
			}

			// forward
			x := newVariable(ids).Reshape(1, -1)                     // (1, C)
			logits := model.Forward(x)                               // (1, C, V)
			logits = F.GetItem(1, []int{logits.Size(1) - 1})(logits) // (1, 1, V)
			logits = F.Reshape(-1)(logits)                           // (V)

			// sample next token
			probs := F.Softmax(-1)(F.MulC(1.0/temperature, logits))
			nextID := multinominal(probs)

			// stop if end token is generated
			if nextID == tokenizer.EndTokenID() {
				break
			}

			// append next token to input and generated tokens
			ids = append(ids, nextID)
			generatedIDs = append(generatedIDs, nextID)
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
