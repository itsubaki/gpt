package tokenizer

import "regexp"

var (
	pattern = `'(?:[sdmt]|ll|ve|re)| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+`
	re      = regexp.MustCompile(pattern)
)

func preTokenize(text string) []string {
	return re.FindAllString(text, -1)
}
