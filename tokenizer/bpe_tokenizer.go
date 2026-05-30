package tokenizer

import "regexp"

type Pair [2]int

type BPETokenizer struct {
	mergeRules *DefaultDict[Pair]
	endToken   string
	endTokenID int
	idToBytes  map[int][]byte
	vocabSize  int
}

func NewBPETokenizer(mergeRules *DefaultDict[Pair], endToken ...string) *BPETokenizer {
	if len(endToken) == 0 {
		endToken = []string{"<|endoftext|>"}
	}

	idToBytes := make(map[int][]byte)
	for i := range 256 {
		idToBytes[i] = []byte{byte(i)}
	}

	for pair, newID := range mergeRules.Seq2() {
		p0, p1 := idToBytes[pair[0]], idToBytes[pair[1]]
		idToBytes[newID] = append(p0, p1...)
	}

	endTokenID := 256 + len(mergeRules.Order)
	idToBytes[endTokenID] = []byte(endToken[0])

	return &BPETokenizer{
		mergeRules: mergeRules,
		endToken:   endToken[0],
		endTokenID: endTokenID,
		idToBytes:  idToBytes,
		vocabSize:  len(idToBytes),
	}
}

func (t *BPETokenizer) encode(text string) []int {
	ids := text2IDs(text)
	for pair, newID := range t.mergeRules.Seq2() {
		ids = merge(ids, pair, newID)
	}

	return ids
}

func (t *BPETokenizer) Encode(inputText string) []int {
	texts := reSplit(inputText, t.endToken)

	var ids []int
	for _, text := range texts {
		if text == t.endToken {
			ids = append(ids, t.endTokenID)
			continue
		}

		for _, preToken := range preTokenize(text) {
			ids = append(ids, t.encode(preToken)...)
		}
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

func text2IDs(text string) []int {
	bytes := []byte(text)
	ids := make([]int, len(bytes))
	for i := range bytes {
		ids[i] = int(bytes[i])
	}

	return ids
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

func reSplit(inputText string, pattern string) []string {
	re := regexp.MustCompile(regexp.QuoteMeta(pattern))
	indices := re.FindAllStringIndex(inputText, -1)

	var last int
	var result []string
	for _, loc := range indices {
		start, end := loc[0], loc[1]
		if start > last {
			result = append(result, inputText[last:start])
		}

		result = append(result, inputText[start:end])
		last = end
	}

	if last < len(inputText) {
		result = append(result, inputText[last:])
	}

	return result
}
