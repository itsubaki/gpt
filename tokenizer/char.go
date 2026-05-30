package tokenizer

type CharTokenizer struct{}

func NewCharTokenizer() *CharTokenizer {
	return &CharTokenizer{}
}

func (t *CharTokenizer) Encode(text string) []rune {
	return []rune(text)
}

func (t *CharTokenizer) Decode(tokens []rune) string {
	return string(tokens)
}
