package dataloader

import (
	"encoding/json"
	"fmt"
	"os"
)

type Tokenizer interface {
	Encode(inputText string) []int
}

type Sample struct {
	IDs    []int
	Labels []int
}

type Alpaca struct {
	Instruction string `json:"instruction"`
	Response    string `json:"response"`
}

type SFTDataset struct {
	tokenizer  Tokenizer
	contextLen int
	samples    []Sample
}

func NewSFTDataset(alpaca []Alpaca, tokenizer Tokenizer, contextLen int) *SFTDataset {
	samples := make([]Sample, 0, len(alpaca))
	for _, a := range alpaca {
		prompt := AlpacaFormat(a.Instruction)
		response := a.Response + "<|endoftext|>"

		// encode
		promptIDs := tokenizer.Encode(prompt)
		responseIDs := tokenizer.Encode(response)

		// input ids and labels
		ids := append(promptIDs, responseIDs...)
		labels := make([]int, len(promptIDs))
		for i := range labels {
			labels[i] = -100
		}
		labels = append(labels, responseIDs...)

		// shift
		ids = ids[:len(ids)-1]
		labels = labels[1:]

		// padding or truncate
		padLen := contextLen - len(ids)
		if padLen > 0 {
			// padding
			ids = append(ids, make([]int, padLen)...)
			pad := make([]int, padLen)
			for i := range pad {
				pad[i] = -100
			}
			labels = append(labels, pad...)
		} else if padLen < 0 {
			// truncate
			ids = ids[:contextLen]
			labels = labels[:contextLen]
		}

		// append sample
		samples = append(samples, Sample{
			IDs:    ids,
			Labels: labels,
		})
	}

	return &SFTDataset{
		tokenizer:  tokenizer,
		contextLen: contextLen,
		samples:    samples,
	}
}

func (s *SFTDataset) Len() int {
	return len(s.samples)
}

func (s *SFTDataset) GetItem(i int) ([]int, []int) {
	sample := s.samples[i]
	return sample.IDs, sample.Labels
}

func MustLoadAlpaca(path string) []Alpaca {
	return Must(LoadAlpaca(path))
}

func LoadAlpaca(path string) ([]Alpaca, error) {
	bytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var alpaca []Alpaca
	if err := json.Unmarshal(bytes, &alpaca); err != nil {
		return nil, err
	}

	return alpaca, nil
}

func AlpacaFormat(message string) string {
	return fmt.Sprintf("### Instruction:\n%s\n\n### Response:\n", message)
}
