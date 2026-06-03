package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/itsubaki/gpt/tokenizer"
)

func main() {
	var filepath, mergeRulesPath, tokenIDsPath string
	var vocabSize int
	flag.StringVar(&filepath, "f", "testdata/tiny_codes.txt", "path to the input file")
	flag.StringVar(&mergeRulesPath, "r", "testdata/merge_rules.gob", "path to the merge rules gob file")
	flag.StringVar(&tokenIDsPath, "t", "testdata/tiny_codes.gob", "path to the token IDs gob file")
	flag.IntVar(&vocabSize, "vocab-size", 1000, "vocabulary size")
	flag.Parse()

	data, err := os.ReadFile(filepath)
	if err != nil {
		panic(err)
	}

	mergeRules, ok := tokenizer.Load(mergeRulesPath)
	if !ok {
		tokenizer.Writer = os.Stdout
		mergeRules = tokenizer.TrainBPE(string(data), vocabSize)
		if err := tokenizer.Save(mergeRulesPath, mergeRules); err != nil {
			panic(err)
		}

		fmt.Println("saved merge rules to", mergeRulesPath)
	}

	tknizer := tokenizer.NewBPETokenizer(mergeRules)
	for key := range keys(tknizer.ID2Bytes) {
		fmt.Printf("%3d -> %q\n", key, tknizer.Decode([]int{key}))
	}

	sample := string([]rune(string(data)))
	byteCount := len([]byte(sample))

	now := time.Now()
	ids := tknizer.Encode(sample)

	fmt.Println()
	fmt.Println("byte count:", byteCount)
	fmt.Println("token count:", len(ids))
	fmt.Println("compression ratio:", float64(byteCount)/float64(len(ids)))
	fmt.Println("encoding time:", time.Since(now))

}

func keys(m map[int][]byte) []int {
	var keys []int
	for k := range m {
		keys = append(keys, k)
	}

	sort.Ints(keys)
	return keys
}
