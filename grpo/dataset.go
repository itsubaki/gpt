package grpo

import (
	"fmt"

	"github.com/itsubaki/autograd/tensor"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/dataloader"
)

type Tokenizer interface {
	Encode(inputText string) []int
}

type Dataset struct {
	Prompts      []string
	GroundTruths []string
	Tokenizer    Tokenizer
}

func NewDataset(tokenizer Tokenizer) *Dataset {
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
		Tokenizer:    tokenizer,
	}
}

func (s *Dataset) Len() int {
	return len(s.Prompts)
}

func (s *Dataset) GetItem(i int) (string, string) {
	return s.Prompts[i], s.GroundTruths[i]
}

func (s *Dataset) GetBatch(prompts, gts []string) (*variable.Variable, *variable.Variable) {
	var allIDs, allMasks [][]int
	for i := range prompts {
		promptIDs := s.Tokenizer.Encode(prompts[i])
		responseIDs := s.Tokenizer.Encode(gts[i])

		// ids
		ids := append(promptIDs, responseIDs...)
		allIDs = append(allIDs, ids)

		// mask
		var mask []int
		for range promptIDs {
			mask = append(mask, 0)
		}

		for range responseIDs {
			mask = append(mask, 1)
		}

		allMasks = append(allMasks, mask)
	}

	// pad to the same length
	var maxLen int
	for _, ids := range allIDs {
		if len(ids) > maxLen {
			maxLen = len(ids)
		}
	}

	var paddedIDs, paddedMasks []int
	for i := range allIDs {
		padLen := maxLen - len(allIDs[i])

		ids := append([]int{}, allIDs[i]...)
		ids = append(ids, make([]int, padLen)...)

		mask := append([]int{}, allMasks[i]...)
		mask = append(mask, make([]int, padLen)...)

		paddedIDs = append(paddedIDs, ids...)
		paddedMasks = append(paddedMasks, mask...)
	}

	// reshape to (batch_size, max_len)
	ids := variable.From(tensor.Float64(tensor.New([]int{len(paddedIDs)}, paddedIDs)))
	masks := variable.From(tensor.Float64(tensor.New([]int{len(paddedMasks)}, paddedMasks)))
	return ids.Reshape(len(prompts), maxLen), masks.Reshape(len(prompts), maxLen)
}
