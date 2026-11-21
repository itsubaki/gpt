package grpo_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/itsubaki/gpt/cmd/grpo/grpo"
)

var _ grpo.Tokenizer = (*MockTokenizer)(nil)

type MockTokenizer struct{}

func (t *MockTokenizer) Encode(text string) []int {
	var encoded []int
	for _, r := range text {
		encoded = append(encoded, int(r))
	}

	return encoded
}

func ExampleDataset() {
	dataset := grpo.NewDataset(&MockTokenizer{})
	fmt.Println(dataset.Len())
	fmt.Println("---")

	prompt, gt := dataset.GetItem(4)
	fmt.Print(prompt)
	fmt.Println(gt)

	// Output:
	// 81
	// ---
	// ### Instruction:
	// 1+5=
	//
	// ### Response:
	// 6
}

func ExampleDataset_GetBatch() {
	dataset := grpo.NewDataset(&MockTokenizer{})
	ids, masks, err := dataset.GetBatch(
		[]string{
			"### Instruction:\n1+1=\n\n### Response:\n2",
			"### Instruction:\n9+9=\n\n### Response:\n18",
		},
		[]string{
			"2",
			"18",
		},
		256,
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(ids.Shape())
	fmt.Println(masks.Shape())

	// Output:
	// [2 41]
	// [2 41]
}

func TestDatasetGetBatch_contextLimit(t *testing.T) {
	cases := []struct {
		name     string
		response string
		want     int
	}{
		{
			name:     "below limit",
			response: "1234",
			want:     7,
		},
		{
			name:     "at limit",
			response: "12345",
			want:     8,
		},
	}

	for _, c := range cases {
		dataset := grpo.NewDataset(&MockTokenizer{})

		ids, masks, err := dataset.GetBatch([]string{"abc", "p"}, []string{c.response, "1"}, 8)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		if ids.Size(0) != 2 || ids.Size(1) != c.want || masks.Size(0) != 2 || masks.Size(1) != c.want {
			t.Fatalf("%s: batch shapes = %v, %v; want [2 %d]", c.name, ids.Shape(), masks.Shape(), c.want)
		}

		for pos := range c.want {
			var wantMask float32
			if pos >= 3 {
				wantMask = 1
			}

			if masks.At(0, pos) != wantMask {
				t.Fatalf("%s: mask at position %d = %v, want %v", c.name, pos, masks.At(0, pos), wantMask)
			}

			if pos >= 2 && (ids.At(1, pos) != 0 || masks.At(1, pos) != 0) {
				t.Fatalf("%s: padding at position %d is not zero", c.name, pos)
			}
		}
	}
}

func TestDatasetGetBatch_invalid(t *testing.T) {
	cases := []struct {
		name          string
		maxContextLen int
		want          string
	}{
		{
			name:          "one token above limit",
			maxContextLen: 8,
			want:          "exceeding max context length 8",
		},
		{
			name:          "zero context length",
			maxContextLen: 0,
			want:          "max context length must be positive",
		},
		{
			name:          "negative context length",
			maxContextLen: -1,
			want:          "max context length must be positive",
		},
	}

	for _, c := range cases {
		dataset := grpo.NewDataset(&MockTokenizer{})

		ids, masks, err := dataset.GetBatch([]string{"abc"}, []string{"123456"}, c.maxContextLen)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Fatalf("%s: error = %v, want %q", c.name, err, c.want)
		}

		if ids != nil || masks != nil {
			t.Fatalf("%s: invalid batch returned tensors", c.name)
		}
	}
}
