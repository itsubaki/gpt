package grpo_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/cmd/grpo/grpo"
	"github.com/itsubaki/gpt/model"
	"github.com/itsubaki/gpt/tokenizer"
)

type generationLimitModel struct {
	logits      *variable.Variable
	calls       int
	position    int
	maxPosition int
}

func (m *generationLimitModel) ClearCache() {
	m.position = 0
}

func (m *generationLimitModel) Forward(ids *variable.Variable) *variable.Variable {
	m.calls++
	m.position += ids.Size()
	m.maxPosition = max(m.maxPosition, m.position)
	return m.logits
}

func newGenerationLimitModel(vocabSize, nextID int) *generationLimitModel {
	logits := tensor.Full([]int{1, 1, vocabSize}, float32(math.Inf(-1)))
	logits.Set([]int{0, 0, nextID}, 0)

	return &generationLimitModel{
		logits: variable.From(logits),
	}
}

func ExampleGenerateGroup() {
	bpe := tokenizer.NewBPETokenizer(tokenizer.NewDefaultDict[tokenizer.Pair]())
	m := newGenerationLimitModel(bpe.VocabSize, int('2'))

	prompts, responses, advantages, err := grpo.GenerateGroup(m, bpe, []string{"abc"}, []string{"22222"}, 2, 8)
	if err != nil {
		panic(err)
	}

	fmt.Println(prompts)
	fmt.Println(responses)
	fmt.Println(advantages)

	// Output:
	// [abc abc]
	// [22222 22222]
	// [0 0]
}

func TestGenerateGroup_contextLimit(t *testing.T) {
	rules := tokenizer.NewDefaultDict[tokenizer.Pair]()
	rules.Set(tokenizer.Pair{'a', 'b'}, 256)
	bpe := tokenizer.NewBPETokenizer(rules)

	cases := []struct {
		name          string
		prompt        string
		maxContextLen int
		nextID        int
		want          int
	}{
		{
			name:          "fills remaining context",
			prompt:        "xyz",
			maxContextLen: 8,
			nextID:        int('2'),
			want:          5,
		},
		{
			name:          "counts merged tokens",
			prompt:        "ab",
			maxContextLen: 8,
			nextID:        int('2'),
			want:          7,
		},
		{
			name:          "counts UTF-8 tokens",
			prompt:        "あ",
			maxContextLen: 8,
			nextID:        int('2'),
			want:          5,
		},
		{
			name:          "uses context beyond 1000 tokens",
			prompt:        "p",
			maxContextLen: 1005,
			nextID:        int('2'),
			want:          1004,
		},
		{
			name:          "stops at end token",
			prompt:        "p",
			maxContextLen: 8,
			nextID:        bpe.EndTokenID(),
			want:          0,
		},
	}

	for _, c := range cases {
		m := newGenerationLimitModel(bpe.VocabSize, c.nextID)

		prompts, responses, advantages, err := grpo.GenerateGroup(m, bpe, []string{c.prompt}, []string{"2"}, 2, c.maxContextLen)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		if len(prompts) != 2 || len(responses) != 2 || len(advantages) != 2 {
			t.Fatalf("%s: group sizes = %d, %d, %d; want 2", c.name, len(prompts), len(responses), len(advantages))
		}

		for i := range responses {
			if prompts[i] != c.prompt || responses[i] != strings.Repeat("2", c.want) {
				t.Fatalf("%s: sample %d: prompt = %q, response tokens = %d; want %d", c.name, i, prompts[i], len(bpe.Encode(responses[i])), c.want)
			}
		}

		if m.maxPosition > c.maxContextLen {
			t.Fatalf("%s: model received %d tokens, exceeding context %d", c.name, m.maxPosition, c.maxContextLen)
		}
	}
}

func TestGenerateGroup_invalid(t *testing.T) {
	cases := []struct {
		name          string
		prompt        string
		maxContextLen int
		want          string
	}{
		{
			name:          "prompt fills context",
			prompt:        "12345678",
			maxContextLen: 8,
			want:          "leaving no room for a response",
		},
		{
			name:          "prompt exceeds context",
			prompt:        "123456789",
			maxContextLen: 8,
			want:          "leaving no room for a response",
		},
		{
			name:          "empty prompt",
			prompt:        "",
			maxContextLen: 8,
			want:          "has no tokens",
		},
		{
			name:          "zero context length",
			prompt:        "p",
			maxContextLen: 0,
			want:          "max context length must be positive",
		},
	}

	for _, c := range cases {
		bpe := tokenizer.NewBPETokenizer(tokenizer.NewDefaultDict[tokenizer.Pair]())
		m := newGenerationLimitModel(bpe.VocabSize, int('2'))

		if _, _, _, err := grpo.GenerateGroup(m, bpe, []string{c.prompt}, []string{"2"}, 1, c.maxContextLen); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Fatalf("%s: error = %v, want %q", c.name, err, c.want)
		}

		if m.calls != 0 {
			t.Fatalf("%s: model called %d times for invalid prompt", c.name, m.calls)
		}
	}
}

func TestGenerateGroup_trainAtContextLimit(t *testing.T) {
	cases := []struct {
		name          string
		prompt        string
		maxContextLen int
	}{
		{
			name:          "context length 8",
			prompt:        "abc",
			maxContextLen: 8,
		},
		{
			name:          "context length 16",
			prompt:        "abcd",
			maxContextLen: 16,
		},
	}

	for _, c := range cases {
		bpe := tokenizer.NewBPETokenizer(tokenizer.NewDefaultDict[tokenizer.Pair]())
		m := model.NewGPT(bpe.VocabSize, c.maxContextLen, 2, 1, 1, 10000)

		params := m.Params()
		for name, p := range params {
			var value float32
			if name == "embed.w" || strings.HasSuffix(name, ".gamma") {
				value = 1
			}

			for i := range p.Data.Data {
				p.Data.Data[i] = value
			}
		}

		params["unembed.w"].Data.Set([]int{0, int('2')}, 100)
		params["unembed.w"].Data.Set([]int{1, int('2')}, 100)
		m.Eval()
		dataset := grpo.NewDataset(bpe)

		prompts, responses, advantages, err := grpo.GenerateGroup(m, bpe, []string{c.prompt}, []string{"2"}, 2, m.MaxContextLen)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		ids, mask, err := dataset.GetBatch(prompts, responses, m.MaxContextLen)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		if ids.Size(1) != m.MaxContextLen {
			t.Fatalf("%s: batch length = %d, want %d", c.name, ids.Size(1), m.MaxContextLen)
		}

		oldModel, err := model.NewGPTFromState(m.State())
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}

		oldModel.Eval()
		m.Train()

		loss := grpo.Loss(m, oldModel, ids, mask, advantages, 0.2)
		if value := float64(loss.At()); math.IsNaN(value) || math.IsInf(value, 0) {
			t.Fatalf("%s: non-finite loss: %v", c.name, value)
		}

		loss.Backward()
	}
}
