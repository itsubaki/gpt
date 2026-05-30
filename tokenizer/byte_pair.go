package tokenizer

import (
	"iter"
	"math"
	"strings"
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

	for pair, newID := range mergeRules.Seq2() {
		p0, p1 := idToBytes[pair[0]], idToBytes[pair[1]]
		idToBytes[newID] = append(p0, p1...)
	}

	return &BPETokenizer{
		mergeRules: mergeRules,
		idToBytes:  idToBytes,
		vocabSize:  len(idToBytes),
	}
}

func (t *BPETokenizer) Encode(text string) []int {
	bytes := []byte(text)
	ids := make([]int, len(bytes))
	for i := range bytes {
		ids[i] = int(bytes[i])
	}

	for pair, newID := range t.mergeRules.Seq2() {
		ids = merge(ids, pair, newID)
	}

	return ids
}

func (t *BPETokenizer) Decode(ids []int) string {
	var bytes []byte
	for _, id := range ids {
		bytes = append(bytes, t.idToBytes[id]...)
	}

	return string(bytes)
}

type Pair [2]int

type Counts struct {
	Counts    map[Pair]int
	FirstSeen map[Pair]int
}

func countPairs(ids []int, counts ...*Counts) *Counts {
	cnt := make(map[Pair]int)
	firstSeen := make(map[Pair]int)
	if len(counts) > 0 {
		cnt = counts[0].Counts
		firstSeen = counts[0].FirstSeen
	}

	var order int
	for i := range len(ids) - 1 {
		p := Pair{ids[i], ids[i+1]}
		cnt[p]++

		if _, ok := firstSeen[p]; !ok {
			firstSeen[p] = order
			order++
		}
	}

	return &Counts{
		Counts:    cnt,
		FirstSeen: firstSeen,
	}
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

func (r *MergeRules) Delete(pair Pair) {
	delete(r.rules, pair)
	for i, p := range r.order {
		if p == pair {
			r.order = append(r.order[:i], r.order[i+1:]...)
			break
		}
	}
}

func trainBPE(inputText string, vocabSize int, endToken ...string) *MergeRules {
	if len(endToken) == 0 {
		endToken = []string{"<|endoftext|>"}
	}
	texts := strings.Split(inputText, endToken[0])

	idsList := make([][]int, len(texts))
	for i, text := range texts {
		bytes := []byte(text)
		idsList[i] = make([]int, len(bytes))
		for j := range bytes {
			idsList[i][j] = int(bytes[j])
		}
	}

	mergeRules := NewMergeRules()
	for step := range vocabSize - 256 - 1 {
		counts := &Counts{
			Counts:    make(map[Pair]int),
			FirstSeen: make(map[Pair]int),
		}
		for _, ids := range idsList {
			counts = countPairs(ids, counts)
		}

		cnt := counts.Counts
		if len(cnt) == 0 {
			break
		}

		bestCount := -1
		bestSeen := math.MaxInt
		firstSeen := counts.FirstSeen

		var bestPair Pair
		for p, c := range cnt {
			if c > bestCount || (c == bestCount && firstSeen[p] < bestSeen) {
				bestCount = c
				bestSeen = firstSeen[p]
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
