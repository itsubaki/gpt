package tokenizer

import (
	"strconv"
	"strings"
)

type Pair [2]int

func TrainBPE(inputText string, vocabSize int, endToken ...string) *DefaultDict[Pair] {
	if len(endToken) == 0 {
		endToken = []string{"<|endoftext|>"}
	}
	texts := strings.Split(inputText, endToken[0])

	idsCounts := NewDefaultDict[string]()
	for _, text := range texts {
		for _, preToken := range preTokenize(text) {
			key := id2Key(text2IDs(preToken))
			idsCounts.Set(key, idsCounts.Dict[key]+1)
		}
	}

	numMerges := vocabSize - 256 - 1
	mergeRules := NewDefaultDict[Pair]()
	for step := range numMerges {
		counts := NewDefaultDict[Pair]()
		for tokens, count := range idsCounts.Seq2() {
			counts = countPairs(key2IDs(tokens), count, counts)
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

		newIDsCounts := NewDefaultDict[string]()
		for tokens, count := range idsCounts.Seq2() {
			newIDs := merge(key2IDs(tokens), bestPair, newID)
			newIDsKey := id2Key(newIDs)
			newIDsCounts.Set(newIDsKey, newIDsCounts.Dict[newIDsKey]+count)
		}

		idsCounts = newIDsCounts
	}

	return mergeRules
}

func countPairs(ids []int, weight int, counts ...*DefaultDict[Pair]) *DefaultDict[Pair] {
	cnts := NewDefaultDict[Pair]()
	if len(counts) > 0 {
		cnts = counts[0]
	}

	for i := range len(ids) - 1 {
		p := Pair{ids[i], ids[i+1]}
		cnts.Set(p, cnts.Dict[p]+weight)
	}

	return cnts
}

func id2Key(ids []int) string {
	if len(ids) == 0 {
		return ""
	}

	var b strings.Builder
	for i, id := range ids {
		if i > 0 {
			b.WriteByte(',')
		}

		b.WriteString(strconv.Itoa(id))
	}

	return b.String()
}

func key2IDs(key string) []int {
	if key == "" {
		return nil
	}

	parts := strings.Split(key, ",")
	ids := make([]int, len(parts))
	for i, part := range parts {
		id, err := strconv.Atoi(part)
		if err != nil {
			panic(err)
		}

		ids[i] = id
	}

	return ids
}
