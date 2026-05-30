package tokenizer

import "regexp"

func preTokenize(text string) []string {
	pattern := `'(?:[sdmt]|ll|ve|re)| ?\pL+| ?\pN+| ?[^\s\pL\pN]+|\s+`
	re := regexp.MustCompile(pattern)
	return re.FindAllString(text, -1)
}
