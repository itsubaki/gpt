package tokenizer

type ByteTokenizer struct{}

func NewByteTokenizer() *ByteTokenizer {
	return &ByteTokenizer{}
}

func (t *ByteTokenizer) Encode(text string) []byte {
	return []byte(text)
}

func (t *ByteTokenizer) Decode(tokens []byte) string {
	return string(tokens)
}
