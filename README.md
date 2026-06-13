# gpt

[![PkgGoDev](https://pkg.go.dev/badge/github.com/itsubaki/gpt)](https://pkg.go.dev/github.com/itsubaki/gpt)
[![Go Report Card](https://goreportcard.com/badge/github.com/itsubaki/gpt)](https://goreportcard.com/report/github.com/itsubaki/gpt)
[![tests](https://github.com/itsubaki/gpt/workflows/tests/badge.svg)](https://github.com/itsubaki/gpt/actions)

GPT implementation in Go from scratch

## Train BPE Tokenizer

```shell
% make dl
curl -fs -o testdata/tiny_codes.txt https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes.txt
```

```shell
% make tokenize
go run ./cmd/tokenize -f testdata/tiny_codes.txt -vocab-size 1000
Training BPE 100%|██████████████████████████████| 743/743 [1.8s<0.0s, 880.1 it/s]
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
encoding time: 1.459157917s
saved tokens to testdata/tiny_codes.bin
```

## Pre-Train GPT

```shell
% make pretrain
go run ./cmd/pretrain/main.go --max-iters 200
Pre-Training   100%|██████████████████████████████| 200/200 [158.1s<0.0s, 1.2 it/s]
```

## Generate Text

```shell
make generate
go run ./cmd/generate/main.go --prompt 'def add(a, b):'
```

```
model parameters:
 VocabSize    : 1000
 MaxContextLen: 256
 EmbedDim     : 192
 NumOfHeads   : 6
 NumOfBlocks  : 6
------------------------------
300,890,40,97,44,358,281,259,312,358,390,365,58,272,301,428,97,41,259,301,273,347,358,271,307,40,97,44,358,41,10,999,
------------------------------
def add(a, b):
    if b == 0:
        return (a)
    return a + b

print(a, b)
```

## References

- [ゼロから作るDeep Learning ❻](https://www.oreilly.co.jp/books/9784814401611/)
- [oreilly-japan/deep-learning-from-scratch-6](https://github.com/oreilly-japan/deep-learning-from-scratch-6)
