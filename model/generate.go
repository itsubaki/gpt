package model

import (
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
	ClearCache()
}

func GenerateText(
	model Model,
	maxContextLen int,
	tokenizer Tokenizer,
	prompt string,
	maxNewTokens int,
	temperature float64,
) <-chan int {
	ch := make(chan int)

	go func() {
		defer close(ch)

		// generation
		func() {
			// disable gradient tracking for generation
			defer variable.Nograd().End()

			// clear cache before generation
			model.ClearCache()

			// encode prompt
			ids := tokenizer.Encode(prompt)

			// feed prompt tokens one by one to populate the KV cache
			var x *variable.Variable
			for _, id := range ids {
				x = newVariable([]int{id}).Reshape(1, 1) // (1, 1)
				x = model.Forward(x)                     // (1, 1, V)
				ch <- id
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

				// stop if end token is generated
				if nextID == tokenizer.EndTokenID() {
					break
				}

				// send next token to channel
				ch <- nextID

				// next token only
				x = newVariable([]int{nextID}).Reshape(1, 1) // (1, 1)
				x = model.Forward(x)                         // (1, 1, V)
			}
		}()
	}()

	return ch
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
