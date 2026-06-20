package grpo

import (
	"fmt"

	"github.com/itsubaki/gpt/dataloader"
)

type Dataset struct {
	Prompts      []string
	GroundTruths []string
}

func NewDataset() *Dataset {
	var prompts, gts []string
	for i := 1; i < 10; i++ {
		for j := 1; j < 10; j++ {
			// 1+1=2, 1+2=3, ..., 9+9=18
			prompt := dataloader.AlpacaFormat(fmt.Sprintf("%d+%d=", i, j))
			gt := fmt.Sprintf("%d", i+j)

			// append
			prompts = append(prompts, prompt)
			gts = append(gts, gt)
		}
	}

	return &Dataset{
		Prompts:      prompts,
		GroundTruths: gts,
	}
}

func (s *Dataset) Len() int {
	return len(s.Prompts)
}

func (s *Dataset) GetItem(i int) (string, string) {
	return s.Prompts[i], s.GroundTruths[i]
}

func (s *Dataset) GetBatch(prompts, gts []string) ([]int, []int) {
	return nil, nil
}
