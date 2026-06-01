package layer_test

import "github.com/itsubaki/gpt/layer"

func ExampleRoPE() {
	theta := 0.5
	keydim := 4
	maxContextLen := 100

	_ = layer.RoPE(theta, keydim, maxContextLen)

	// Output:
}
