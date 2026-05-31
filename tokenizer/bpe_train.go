package tokenizer

import (
	"strconv"
	"strings"
)

type Pair [2]int

func TrainBPE(inputText string, vocabSize int, endToken ...string) *DefaultDict[Pair, int] {
	if len(endToken) == 0 {
		endToken = []string{"<|endoftext|>"}
	}
	texts := strings.Split(inputText, endToken[0])

	idsCounts := NewDefaultDict[string, int]()
	for _, text := range texts {
		for _, preToken := range preTokenize(text) {
			key := id2Key(text2IDs(preToken))
			idsCounts.Set(key, idsCounts.Dict[key]+1)
		}
	}

	// cache
	pair2IDs := make(map[Pair]map[string]struct{})
	pairCounts := NewDefaultDict[Pair, int]()
	for key, count := range idsCounts.Seq2() {
		ids := key2IDs(key)
		pairCounts = countPairs(ids, count, pairCounts)
		for i := range ids[:len(ids)-1] {
			p := Pair{ids[i], ids[i+1]}
			if _, ok := pair2IDs[p]; !ok {
				pair2IDs[p] = make(map[string]struct{})
			}

			pair2IDs[p][key] = struct{}{}
		}
	}

	numMerges := vocabSize - 256 - 1
	mergeRules := NewDefaultDict[Pair, int]()
	for step := range numMerges {
		if pairCounts.Len() == 0 {
			break
		}

		bestCount := -1
		var bestPair Pair
		for pair, count := range pairCounts.Seq2() {
			if count > bestCount {
				bestCount = count
				bestPair = pair
			}
		}

		newID := 256 + step
		mergeRules.Set(bestPair, newID)

		affectedIDs := pair2IDs[bestPair]
		delete(pair2IDs, bestPair)
		for key := range affectedIDs {
			idsCount := idsCounts.Dict[key]
			ids := key2IDs(key)
			newIDs := merge(ids, bestPair, newID)

			// update
			idsCounts.Delete(key)
			idsCounts.Set(id2Key(newIDs), idsCount)

			// update old pair counts
			oldCounts := countPairs(ids, 1)
			for pair, count := range oldCounts.Seq2() {
				pairCounts.Set(pair, pairCounts.Dict[pair]-count*idsCount)
				if pairCounts.Dict[pair] < 1 {
					pairCounts.Delete(pair)
				}

				// update cache
				if pair2IDs[pair] != nil {
					delete(pair2IDs[pair], key)
					if len(pair2IDs[pair]) == 0 {
						delete(pair2IDs, pair)
					}
				}
			}

			// update new pair counts
			newCounts := countPairs(newIDs, 1)
			for pair, count := range newCounts.Seq2() {
				pairCounts.Set(pair, pairCounts.Dict[pair]+count*idsCount)

				// update cache
				if _, ok := pair2IDs[pair]; !ok {
					pair2IDs[pair] = make(map[string]struct{})
				}
				pair2IDs[pair][id2Key(newIDs)] = struct{}{}
			}
		}
	}

	return mergeRules
}

func countPairs(ids []int, weight int, counts ...*DefaultDict[Pair, int]) *DefaultDict[Pair, int] {
	cnts := NewDefaultDict[Pair, int]()
	if len(counts) > 0 {
		cnts = counts[0]
	}

	for i := range len(ids) - 1 {
		p := Pair{ids[i], ids[i+1]}
		cnts.Set(p, cnts.Dict[p]+weight)
	}

	return cnts
}

func text2IDs(text string) []int {
	bytes := []byte(text)
	ids := make([]int, len(bytes))
	for i := range bytes {
		ids[i] = int(bytes[i])
	}

	return ids
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
