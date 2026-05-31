package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"os"
	"sort"
	"time"

	"github.com/itsubaki/gpt/tokenizer"
)

func main() {
	var filepath, gobpath string
	var vocabSize int
	flag.StringVar(&filepath, "f", "testdata/tiny_codes.txt", "path to the input file")
	flag.StringVar(&gobpath, "g", "testdata/merge_rules.gob", "path to the merge rules gob file")
	flag.IntVar(&vocabSize, "vocab-size", 1000, "vocabulary size")
	flag.Parse()

	data, err := os.ReadFile(filepath)
	if err != nil {
		panic(err)
	}

	mergeRules, ok := load(gobpath)
	if !ok {
		now := time.Now()
		fmt.Println("training BPE...")
		mergeRules = tokenizer.TrainBPE(string(data), vocabSize)
		if err := save(gobpath, mergeRules); err != nil {
			panic(err)
		}

		fmt.Println("saved merge rules to", gobpath)
		fmt.Println("elapsed time:", time.Since(now))
	}

	tknizer := tokenizer.NewBPETokenizer(mergeRules)
	for key := range keys(tknizer.ID2Bytes) {
		fmt.Printf("%3d -> %q\n", key, tknizer.Decode([]int{key}))
	}

	sample := string([]rune(string(data)))
	byteCount := len([]byte(sample))

	now := time.Now()
	ids := tknizer.Encode(sample)

	fmt.Println("byte count:", byteCount)
	fmt.Println("token count:", len(ids))
	fmt.Println("compression ratio:", float64(byteCount)/float64(len(ids)))
	fmt.Println("elapsed time:", time.Since(now))
}

func keys(m map[int][]byte) []int {
	var keys []int
	for k := range m {
		keys = append(keys, k)
	}

	sort.Ints(keys)
	return keys
}

func save(filename string, dict *tokenizer.DefaultDict[tokenizer.Pair, int]) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	if err := gob.NewEncoder(f).Encode(dict); err != nil {
		return fmt.Errorf("encode: %v", err)
	}

	return nil
}

func load(filename string) (*tokenizer.DefaultDict[tokenizer.Pair, int], bool) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, false
	}
	defer func() { _ = f.Close() }()

	var dict tokenizer.DefaultDict[tokenizer.Pair, int]
	if err := gob.NewDecoder(f).Decode(&dict); err != nil {
		return nil, false
	}

	return &dict, true
}
