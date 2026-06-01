# gpt

[![PkgGoDev](https://pkg.go.dev/badge/github.com/itsubaki/gpt)](https://pkg.go.dev/github.com/itsubaki/gpt)
[![Go Report Card](https://goreportcard.com/badge/github.com/itsubaki/gpt)](https://goreportcard.com/report/github.com/itsubaki/gpt)
[![tests](https://github.com/itsubaki/gpt/workflows/tests/badge.svg)](https://github.com/itsubaki/gpt/actions)

GPT-based chatbot in Go from scratch

## BPE Tokenizer

```shell
% make dl
curl -fs -o testdata/tiny_codes.txt https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes.txt
```

```shell
go run ./cmd/tokenize -f testdata/tiny_codes.txt -vocab-size 1000
Training BPE 100%|██████████████████████████████| 743/743 [1.8s<0.0s, 880.1 it/s]]]
saved merge rules to testdata/merge_rules.gob
...
995 -> "are"
996 -> ")."
997 -> " my"
998 -> "emain"
999 -> "<|endoftext|>"

byte count: 6487033
token count: 2640742
compression ratio: 2.456519038967078
tokenization elapsed time: 1.459157917s
```

## References

- [ゼロから作るDeep Learning ❻](https://www.oreilly.co.jp/books/9784814401611/)
- [oreilly-japan/deep-learning-from-scratch-6](https://github.com/oreilly-japan/deep-learning-from-scratch-6)
