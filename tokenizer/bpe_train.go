package tokenizer

import (
	"strings"
)

func TrainBPE(inputText string, vocabSize int, endToken ...string) *DefaultDict[Pair] {
	if len(endToken) == 0 {
		endToken = []string{"<|endoftext|>"}
	}
	texts := strings.Split(inputText, endToken[0])

	idsList := make([][]int, 0)
	for _, text := range texts {
		for _, preToken := range preTokenize(text) {
			idsList = append(idsList, text2IDs(preToken))
		}
	}

	numMerges := vocabSize - 256 - 1
	mergeRules := NewDefaultDict[Pair]()
	for step := range numMerges {
		counts := NewDefaultDict[Pair]()
		for _, ids := range idsList {
			counts = countPairs(ids, counts)
		}

		if counts.Len() == 0 {
			break
		}

		bestCount := -1
		var bestPair Pair
		for p, c := range counts.Seq2() {
			if c > bestCount {
				bestCount = c
				bestPair = p
			}
		}

		newID := 256 + step
		mergeRules.Set(bestPair, newID)
		for i := range idsList {
			idsList[i] = merge(idsList[i], bestPair, newID)
		}
	}

	return mergeRules
}

func countPairs(ids []int, counts ...*DefaultDict[Pair]) *DefaultDict[Pair] {
	cnts := NewDefaultDict[Pair]()
	if len(counts) > 0 {
		cnts = counts[0]
	}

	for i := range len(ids) - 1 {
		p := Pair{ids[i], ids[i+1]}
		cnts.Set(p, cnts.Dict[p]+1)
	}

	return cnts
}
