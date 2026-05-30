package tokenizer

type BytePairEncodingTokenizer struct{}

type Pair [2]int

func count(ids []int) map[Pair]int {
	counts := make(map[Pair]int)
	for i := 0; i < len(ids)-1; i++ {
		pair := Pair{ids[i], ids[i+1]}
		counts[pair]++
	}

	return counts
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
