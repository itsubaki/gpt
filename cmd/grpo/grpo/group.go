package grpo

import (
	"context"
	"fmt"

	"github.com/itsubaki/gpt/model"
)

func GenerateGroup(
	m model.Model,
	tokenizer model.Tokenizer,
	prompts []string,
	gts []string,
	groupSize int,
	maxContextLen int,
) ([]string, []string, []float32, error) {
	if maxContextLen <= 0 {
		return nil, nil, nil, fmt.Errorf("max context length must be positive: %d", maxContextLen)
	}

	var allPrompts, allResponses []string
	var allAdvantages []float32
	for i := range prompts {
		prompt, responses := prompts[i], make([]string, groupSize)
		promptLen := len(tokenizer.Encode(prompt))
		if promptLen == 0 {
			return nil, nil, nil, fmt.Errorf("prompt %d has no tokens", i)
		}

		if promptLen >= maxContextLen {
			return nil, nil, nil, fmt.Errorf("prompt %d has %d tokens, leaving no room for a response within max context length %d", i, promptLen, maxContextLen)
		}

		maxNewTokens := maxContextLen - promptLen
		for j := range responses {
			fullText := model.GenerateText(
				context.Background(),
				m,
				tokenizer,
				prompt,
				maxNewTokens,
				1.0,
			)

			responses[j] = fullText[len(prompt):]
		}

		// calculate rewards and advantages
		var rewards []float32
		for j := range responses {
			r := Reward(gts[i], responses[j])
			rewards = append(rewards, r)
		}

		var mean float32
		for _, r := range rewards {
			mean += r
		}
		mean /= float32(len(rewards))

		var advantages []float32
		for _, r := range rewards {
			advantages = append(advantages, r-mean)
		}

		// append to all
		for j := range responses {
			allPrompts = append(allPrompts, prompt)
			allResponses = append(allResponses, responses[j])
			allAdvantages = append(allAdvantages, advantages[j])
		}
	}

	return allPrompts, allResponses, allAdvantages, nil
}
