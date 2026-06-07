package main

import (
	"encoding/gob"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/itsubaki/gpt/tokenizer"
)

func main() {
	var path, mergeRulesPath string
	var vocabSize int
	flag.StringVar(&path, "f", "testdata/tiny_codes.txt", "path to the input file")
	flag.StringVar(&mergeRulesPath, "r", "testdata/merge_rules.gob", "path to the merge rules gob file")
	flag.IntVar(&vocabSize, "vocab-size", 1000, "vocabulary size")
	flag.Parse()

	data, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}

	mergeRules, err := tokenizer.Load(mergeRulesPath)
	if err != nil {
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

	bin := strings.TrimSuffix(path, filepath.Ext(path)) + ".bin"
	if err := save(bin, ids); err != nil {
		panic(err)
	}
}

func keys(m map[int][]byte) []int {
	var keys []int
	for k := range m {
		keys = append(keys, k)
	}

	sort.Ints(keys)
	return keys
}

func save(filename string, ids []int) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("create file: %v", err)
	}
	defer func() { _ = f.Close() }()

	if err := gob.NewEncoder(f).Encode(ids); err != nil {
		return fmt.Errorf("encode: %v", err)
	}

	return nil
}
