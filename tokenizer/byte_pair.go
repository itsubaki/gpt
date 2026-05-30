package tokenizer

import "math"

type BytePairEncodingTokenizer struct{}

type Pair [2]int

func count(ids []int) (map[Pair]int, map[Pair]int) {
	counts := make(map[Pair]int)
	firstSeen := make(map[Pair]int)

	var order int
	for i := range len(ids) - 1 {
		p := Pair{ids[i], ids[i+1]}
		counts[p]++

		if _, ok := firstSeen[p]; !ok {
			firstSeen[p] = order
			order++
		}
	}

	return counts, firstSeen
}

func merge(ids []int, pair Pair, newID int) []int {
	merged := make([]int, 0, len(ids))
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

func trainBPE(text string, vocabSize int) map[Pair]int {
	ids := make([]int, len(text))
	for i, b := range text {
		ids[i] = int(b)
	}

	mergeRules := make(map[Pair]int)
	for step := range vocabSize - 256 {
		counts, firstSeen := count(ids)
		if len(counts) == 0 {
			break
		}

		bestCount, bestSeen := -1, math.MaxInt
		var bestPair Pair
		for p, c := range counts {
			if c > bestCount || (c == bestCount && firstSeen[p] < bestSeen) {
				bestCount = c
				bestSeen = firstSeen[p]
				bestPair = p
			}
		}

		newID := 256 + step
		mergeRules[bestPair] = newID
		ids = merge(ids, bestPair, newID)
	}

	return mergeRules
}
