package layer_test

import (
	"fmt"
	"math/rand"

	"github.com/itsubaki/autograd/variable"
	L "github.com/itsubaki/gpt/layer"
)

func ExampleEmbeddings() {
	embeddim := 32
	vocabSize := 100
	maxContextLen := 10

	tokens := make([]float64, maxContextLen)
	for i := range tokens {
		tokens[i] = float64(rand.Intn(vocabSize))
	}

	x := variable.New(tokens...).Reshape(1, maxContextLen)
	emb := L.Embeddings(vocabSize, embeddim)
	output := emb.First(x)

	fmt.Println(x.Shape())
	fmt.Println(output.Shape())

	// Output:
	// [1 10]
	// [1 10 32]
}
