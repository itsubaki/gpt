package model_test

import (
	"context"
	"testing"
	"testing/synctest"

	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/model"
)

type generationModel struct {
	calls int
}

func (*generationModel) ClearCache() {}

func (m *generationModel) Forward(*variable.Variable) *variable.Variable {
	m.calls++
	return variable.New(1, 0, 0).Reshape(1, 1, 3)
}

type generationTokenizer struct{}

func (generationTokenizer) Encode(string) []int { return []int{1} }
func (generationTokenizer) Decode([]int) string { return "" }
func (generationTokenizer) EndTokenID() int     { return 2 }

func TestGenerateTokens_done(t *testing.T) {
	cases := []struct {
		receive   []int
		wantCalls int
	}{
		{
			// Cancel while sending the prompt.
			receive:   nil,
			wantCalls: 1,
		},
		{
			// Consume the prompt, then cancel while sending a generated token.
			receive:   []int{1},
			wantCalls: 1,
		},
	}

	for _, c := range cases {
		synctest.Test(t, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			m := &generationModel{}
			tokens := model.GenerateTokens(ctx, m, 8, generationTokenizer{}, "prompt", 2, 0)
			for _, want := range c.receive {
				if id := <-tokens; id != want {
					t.Fatalf("token = %d, want %d", id, want)
				}
			}

			// Leave the prompt or generated token waiting for a receiver.
			synctest.Wait()
			cancel()

			// Cancellation must finish generation without a receiver.
			synctest.Wait()

			select {
			case id, ok := <-tokens:
				if ok {
					t.Fatalf("received token %d, want a closed channel", id)
				}
			default:
				t.Fatal("token channel was not closed after cancellation")
			}

			if m.calls != c.wantCalls {
				t.Fatalf("Forward called %d times, want %d", m.calls, c.wantCalls)
			}
		})
	}
}
