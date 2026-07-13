package grpo

import "github.com/itsubaki/gpt/model"

func GenerateGroup(
	m model.Model,
	maxContextLen int,
	tokenizer model.Tokenizer,
	prompts []string,
	gts []string,
	groupSize int,
) ([]string, []string, []float64) {
	var allPrompts, allResponses []string
	var allAdvantages []float64
	for i := range prompts {
		prompt, responses := prompts[i], make([]string, groupSize)
		for j := range responses {
			fullText := model.GenerateText(m, maxContextLen, tokenizer, prompt, 1000, 1.0)
			responses[j] = fullText[len(prompt):]
		}

		// calculate rewards and advantages
		var rewards []float64
		for j := range responses {
			r := Reward(gts[i], responses[j])
			rewards = append(rewards, r)
		}

		var mean float64
		for _, r := range rewards {
			mean += r
		}
		mean /= float64(len(rewards))

		var advantages []float64
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

	return allPrompts, allResponses, allAdvantages
}
