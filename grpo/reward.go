package grpo

import (
	"regexp"
	"strconv"
)

var re = regexp.MustCompile(`-?\d+`)

func Reward(groundTruth, response string) float64 {
	matches := re.FindAllString(response, -1)
	if len(matches) == 0 {
		return 0.0
	}

	pred, err := strconv.Atoi(matches[len(matches)-1])
	if err != nil {
		return 0.0
	}

	gt, err := strconv.Atoi(groundTruth)
	if err != nil {
		return 0.0
	}

	if pred == gt {
		return 1.0
	}

	return 0.0
}
