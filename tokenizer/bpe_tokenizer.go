package tokenizer

import "regexp"

type BPETokenizer struct {
	mergeRules *DefaultDict[Pair, int]
	endToken   string
	endTokenID int
	ID2Bytes   map[int][]byte
	VocabSize  int
}

func NewBPETokenizer(mergeRules *DefaultDict[Pair, int], endToken ...string) *BPETokenizer {
	if len(endToken) == 0 {
		endToken = []string{"<|endoftext|>"}
	}

	id2Bytes := make(map[int][]byte)
	for i := range 256 {
		id2Bytes[i] = []byte{byte(i)}
	}

	for pair, newID := range mergeRules.Seq2() {
		p0, p1 := id2Bytes[pair[0]], id2Bytes[pair[1]]
		id2Bytes[newID] = append(p0, p1...)
	}

	endTokenID := 256 + mergeRules.Len()
	id2Bytes[endTokenID] = []byte(endToken[0])

	return &BPETokenizer{
		mergeRules: mergeRules,
		endToken:   endToken[0],
		endTokenID: endTokenID,
		ID2Bytes:   id2Bytes,
		VocabSize:  len(id2Bytes),
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
		bytes = append(bytes, t.ID2Bytes[id]...)
	}

	return string(bytes)
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
