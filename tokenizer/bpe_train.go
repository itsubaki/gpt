package tokenizer

import (
	"math"
	"strings"
)

func trainBPE(inputText string, vocabSize int, endToken ...string) *DefaultDict[Pair] {
	if len(endToken) == 0 {
		endToken = []string{"<|endoftext|>"}
	}
	texts := strings.Split(inputText, endToken[0])

	idsList := make([][]int, len(texts))
	for i, text := range texts {
		idsList[i] = make([]int, 0)
		for _, preToken := range preTokenize(text) {
			bytes := []byte(preToken)
			for j := range bytes {
				idsList[i] = append(idsList[i], int(bytes[j]))
			}
		}
	}

	mergeRules := NewDefaultDict[Pair]()
	for step := range vocabSize - 256 - 1 {
		counts := NewDefaultDict[Pair]()
		for _, ids := range idsList {
			counts = countPairs(ids, counts)
		}

		if counts.Len() == 0 {
			break
		}

		bestCount := -1
		bestSeen := math.MaxInt
		var bestPair Pair
		for p, c := range counts.Seq2() {
			if c > bestCount || (c == bestCount && p[0] < bestSeen) {
				bestCount = c
				bestSeen = p[0]
				bestPair = p
			}
		}

		newID := 256 + step
		mergeRules.Set(bestPair, newID)
		for i, ids := range idsList {
			idsList[i] = merge(ids, bestPair, newID)
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
