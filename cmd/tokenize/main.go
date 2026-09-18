package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/itsubaki/gpt/tokenizer"
)

func main() {
	var mergeRulesPath, text string
	flag.StringVar(&mergeRulesPath, "merge-rules-path", "testdata/merge_rules.gob", "path to the merge rules gob file")
	flag.StringVar(&text, "text", "Hello world!!", "text to encode")
	flag.Parse()

	// open file
	f, err := os.Open(mergeRulesPath)
	if err != nil {
		panic(err)
	}
	defer func() { _ = f.Close() }()

	// create BPE tokenizer from merge rules file
	bpeTokenizer, err := tokenizer.NewBPETokenizerFrom(f)
	if err != nil {
		panic(err)
	}

	ids := bpeTokenizer.Encode(text)
	for _, id := range ids {
		decodeed := bpeTokenizer.Decode([]int{id})
		fmt.Printf("%q(%3d) ", decodeed, id)
	}

	fmt.Println()
}
