package tokenizer

import (
	"iter"
	"math"
)

type BPETokenizer struct {
	mergeRules *MergeRules
	idToBytes  map[int][]byte
	vocabSize  int
}

func NewBPETokenizer(mergeRules *MergeRules) *BPETokenizer {
	idToBytes := make(map[int][]byte)
	for i := range 256 {
		idToBytes[i] = []byte{byte(i)}
	}

	remaining := mergeRules.Len()
	for remaining > 0 {
		for pair, newID := range mergeRules.Seq2() {
			p0, p1 := idToBytes[pair[0]], idToBytes[pair[1]]
			idToBytes[newID] = append(p0, p1...)
			delete(mergeRules.rules, pair)
			remaining--
		}
	}

	return &BPETokenizer{
		mergeRules: mergeRules,
		idToBytes:  idToBytes,
		vocabSize:  len(idToBytes),
	}
}

func (t *BPETokenizer) Encode(text string) []int {
	ids := make([]int, len(text))
	for i, b := range text {
		ids[i] = int(b)
	}

	for pair, newID := range t.mergeRules.Seq2() {
		ids = merge(ids, pair, newID)
	}

	return ids
}

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

type MergeRules struct {
	rules map[Pair]int
	order []Pair
}

func NewMergeRules() *MergeRules {
	return &MergeRules{
		rules: make(map[Pair]int),
		order: make([]Pair, 0),
	}
}

func (r *MergeRules) Len() int {
	return len(r.rules)
}

func (r *MergeRules) Set(pair Pair, newID int) {
	r.rules[pair] = newID
	r.order = append(r.order, pair)
}

func (r *MergeRules) Seq2() iter.Seq2[Pair, int] {
	return func(yield func(Pair, int) bool) {
		for _, pair := range r.order {
			if !yield(pair, r.rules[pair]) {
				return
			}
		}
	}
}

func trainBPE(text string, vocabSize int) *MergeRules {
	ids := make([]int, len(text))
	for i, b := range text {
		ids[i] = int(b)
	}

	mergeRules := NewMergeRules()
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
		mergeRules.Set(bestPair, newID)
		ids = merge(ids, bestPair, newID)
	}

	return mergeRules
}
