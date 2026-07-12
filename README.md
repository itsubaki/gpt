# gpt

[![PkgGoDev](https://pkg.go.dev/badge/github.com/itsubaki/gpt)](https://pkg.go.dev/github.com/itsubaki/gpt)
[![tests](https://github.com/itsubaki/gpt/workflows/tests/badge.svg)](https://github.com/itsubaki/gpt/actions)

GPT implementation in Go from scratch.

```
Token IDs
    ↓
Embedding
    ↓
┌─────────────────────────────┐
│ Transformer Block × N       │
│                             │
│ RMSNorm                     │
│   ↓                         │
│ Multi-Head Attention + RoPE │
│   ↓                         │
│ Residual                    │
│   ↓                         │
│ RMSNorm                     │
│   ↓                         │
│ SwiGLU                      │
│   ↓                         │
│ Residual                    │
└─────────────────────────────┘
    ↓
RMSNorm
    ↓
Linear
    ↓
Logits
```

## How to run

```shell
% make testdata example
```

```shell
curl -fs -o testdata/merge_rules.gob   https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/merge_rules.gob
curl -fs -o testdata/tiny_codes.bin    https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/tiny_codes.bin
curl -fs -o testdata/model_gpt.gob     https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/model_gpt.gob
curl -fs -o testdata/model_gpt_sft.gob https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/model_gpt_sft.gob
```

```python
### Instruction:
Write is_prime function

### Response:
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

## Train BPE Tokenizer

```shell
% make dl
curl -fs -o testdata/tiny_codes.txt      https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes.txt
curl -fs -o testdata/tiny_codes_sft.json https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes_sft.json
```

```shell
% make tokenize
go run ./cmd/tokenize -f testdata/tiny_codes.txt -vocab-size 1000
Training BPE 100%|██████████████████████████████| 743/743
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
go run ./cmd/pretrain/main.go
Pre-Training 100%|██████████████████████████████| 20000/20000
```

<img src="https://github.com/itsubaki/gpt/blob/gob/loss.png">

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

## Supervised Fine-Tuning

```shell
%  make sft
go run ./cmd/sft/main.go
SFT 100%|██████████████████████████████| 500/500
```

<img src="https://github.com/itsubaki/gpt/blob/gob/loss_sft.png">

## Chat

```shell
% make chat
go run ./cmd/chat/main.go --prompt 'Write add function'
```

```
model parameters:
 VocabSize    : 1000
 MaxContextLen: 256
 EmbedDim     : 192
 NumOfHeads   : 6
 NumOfBlocks  : 6
------------------------------
35,35,35,955,435,117,387,58,10,87,903,890,618,271,35,35,35,608,101,966,58,10,300,890,40,97,44,358,281,259,301,273,347,358,999,
------------------------------
### Instruction:
Write add function

### Response:
def add(a, b):
    return a + b
```

```
### Instruction:
Hi, who are you?

### Response:
I'm an AI assistant. What do you need help with?
```

```
### Instruction:
3+9

### Response:
12
```

## GRPO

```shell
%  make grpo
go run ./cmd/grpo/main.go
GRPO 100%|██████████████████████████████| 500/500
```

## References

- [ゼロから作るDeep Learning ❻](https://www.oreilly.co.jp/books/9784814401611/)
- [oreilly-japan/deep-learning-from-scratch-6](https://github.com/oreilly-japan/deep-learning-from-scratch-6)
