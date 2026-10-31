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

func TestGenerateTokensDone(t *testing.T) {
	for _, name := range []string{
		"prompt",
		"generated",
	} {
		synctest.Test(t, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			m := &generationModel{}
			tokens := model.GenerateTokens(ctx, m, 8, generationTokenizer{}, "prompt", 2, 0)
			if name == "generated" {
				if id := <-tokens; id != 1 {
					t.Fatalf("prompt token = %d, want 1", id)
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

			if m.calls != 1 {
				t.Fatalf("Forward called %d times, want 1", m.calls)
			}
		})
	}
}
