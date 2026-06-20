package grpo_test

import (
	"testing"

	"github.com/itsubaki/gpt/grpo"
)

func TestReward(t *testing.T) {
	cases := []struct {
		groundTruth string
		response    string
		want        float64
	}{
		{
			groundTruth: "12",
			response:    "### Instruction:\n3+9=\n\n### Response:\n12",
			want:        1.0,
		},
		{
			groundTruth: "12",
			response:    "### Instruction:\n3+9=\n\n### Response:\n11",
			want:        0.0,
		},
		{
			groundTruth: "12",
			response:    "### Instruction:\n3+9=\n\n### Response:\n",
			want:        0.0,
		},
	}

	for _, c := range cases {
		reward := grpo.Reward(c.groundTruth, c.response)
		if reward != c.want {
			t.Errorf("Reward(%q, %q) == %f, want %f", c.groundTruth, c.response, reward, c.want)
		}
	}
}
