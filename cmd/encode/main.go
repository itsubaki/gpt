package main

import (
	"flag"
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func main() {
	var mergeRulesPath, text string
	flag.StringVar(&mergeRulesPath, "merge-rules-path", "testdata/merge_rules.gob", "path to the merge rules gob file")
	flag.StringVar(&text, "text", "Hello world!!", "text to encode")
	flag.Parse()

	tknizer, err := tokenizer.NewBPETokenizerFrom(mergeRulesPath)
	if err != nil {
		panic(err)
	}

	ids := tknizer.Encode(text)
	for _, id := range ids {
		decodeed := tknizer.Decode([]int{id})
		fmt.Printf("%q(%3d) ", decodeed, id)
	}

	fmt.Println()
}
