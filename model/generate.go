package model

import (
	"fmt"
	"math/rand/v2"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/tokenizer"
)

var (
	_ Tokenizer = (*tokenizer.BPETokenizer)(nil)
	_ Model     = (*GPT)(nil)
)

type Tokenizer interface {
	Encode(text string) []int
	Decode(tokens []int) string
	EndTokenID() int
}

type Model interface {
	Forward(x *variable.Variable) *variable.Variable
}

func GenerateText(
	model Model,
	maxContextLen int,
	tokenizer Tokenizer,
	prompt string,
	maxNewTokens int,
	temperature float64,
	debug bool,
) string {
	ids := tokenizer.Encode(prompt)
	generatedIDs := make([]int, len(ids))
	copy(generatedIDs, ids)

	if debug {
		for _, id := range ids {
			fmt.Printf("%v,", id)
		}
	}

	func() {
		// disable gradient tracking for generation
		defer variable.Nograd().End()

		// generate tokens
		for range maxNewTokens {
			if len(ids) > maxContextLen {
				// keep only the last maxContextLen tokens as input
				ids = ids[len(ids)-maxContextLen:]
			}

			// forward
			x := newVariable(ids).Reshape(1, -1)                     // (1, C)
			logits := model.Forward(x)                               // (1, C, V)
			logits = F.GetItem(1, []int{logits.Size(1) - 1})(logits) // (1, 1, V)
			logits = F.Reshape(-1)(logits)                           // (V)

			// sample next token
			var nextID int
			if temperature == 0 {
				nextID = tensor.Argmax(logits.Data, 0).At()
			} else {
				probs := F.Softmax(-1)(F.MulC(1.0/temperature, logits))
				nextID = multinominal(probs)
			}

			if debug {
				fmt.Printf("%v,", nextID)
			}

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
	return tokenizer.Decode(generatedIDs)
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
