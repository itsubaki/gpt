package tokenizer

import (
	"io"
	"strconv"
	"strings"

	"github.com/itsubaki/gpt/progress"
)

var Writer io.Writer = io.Discard

type Pair [2]int

func TrainBPE(inputText string, vocabSize int, endToken ...string) *DefaultDict[Pair] {
	if len(endToken) == 0 {
		endToken = []string{"<|endoftext|>"}
	}
	texts := strings.Split(inputText, endToken[0])

	idsCounts := NewDefaultDict[string]()
	for _, text := range texts {
		for _, preToken := range preTokenize(text) {
			idsCounts.Incr(ids2Key(text2IDs(preToken)), 1)
		}
	}

	// cache
	cache := NewCache()
	pairCounts := NewDefaultDict[Pair]()
	for key, count := range idsCounts.Seq2() {
		ids := key2IDs(key)
		pairCounts = countPairs(ids, count, pairCounts)
		for i := range ids[:len(ids)-1] {
			cache.Add(Pair{ids[i], ids[i+1]}, key)
		}
	}

	numMerges := vocabSize - 256 - 1
	mergeRules := NewDefaultDict[Pair]()
	bar := progress.NewProgressBar("Training BPE", numMerges, Writer)

	for step := range numMerges {
		bar.Update(step + 1)
		if pairCounts.Len() == 0 {
			break
		}

		// find best pair
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

		affectedIDs := cache.Get(bestPair)
		cache.Delete(bestPair)
		for key := range affectedIDs {
			ids := key2IDs(key)
			newIDs := merge(ids, bestPair, newID)

			// update
			idsCount := idsCounts.Value(key)
			idsCounts.Delete(key)
			idsCounts.Set(ids2Key(newIDs), idsCount)

			// update old pair counts
			oldCounts := countPairs(ids, 1)
			for pair, count := range oldCounts.Seq2() {
				pairCounts.Incr(pair, -count*idsCount)
				if pairCounts.Value(pair) < 1 {
					pairCounts.Delete(pair)
				}

				cache.Delete(pair, key)
			}

			// update new pair counts
			newCounts := countPairs(newIDs, 1)
			for pair, count := range newCounts.Seq2() {
				pairCounts.Incr(pair, count*idsCount)
				cache.Add(pair, ids2Key(newIDs))
			}
		}
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

func merge(ids []int, pair Pair, newID int) []int {
	merged := make([]int, 0)
	for i := 0; i < len(ids); {
		if i < len(ids)-1 && ids[i] == pair[0] && ids[i+1] == pair[1] {
			merged = append(merged, newID)
			i += 2
			continue
		}

		merged = append(merged, ids[i])
		i++
	}

	return merged
}

func text2IDs(text string) []int {
	bytes := []byte(text)
	ids := make([]int, len(bytes))
	for i := range bytes {
		ids[i] = int(bytes[i])
	}

	return ids
}

func ids2Key(ids []int) string {
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
