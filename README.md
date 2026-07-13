# gpt

[![PkgGoDev](https://pkg.go.dev/badge/github.com/itsubaki/gpt)](https://pkg.go.dev/github.com/itsubaki/gpt)
[![tests](https://github.com/itsubaki/gpt/workflows/tests/badge.svg)](https://github.com/itsubaki/gpt/actions)

An implementation of GPT in Go from scratch.

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

## Quick Start

```shell
% make testdata example
```

```shell
curl -fs -o testdata/merge_rules.gob    https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/merge_rules.gob
curl -fs -o testdata/tiny_codes.bin     https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/tiny_codes.bin
curl -fs -o testdata/model_gpt.gob      https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/model_gpt.gob
curl -fs -o testdata/model_gpt_sft.gob  https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/model_gpt_sft.gob
curl -fs -o testdata/model_gpt_grpo.gob https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/model_gpt_grpo.gob
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

## BPE Tokenizer Training

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

## Pre-Training

```shell
% make pretrain
go run ./cmd/pretrain/main.go
Pre-Training 100%|██████████████████████████████| 20000/20000
```

<img src="https://github.com/itsubaki/gpt/blob/gob/loss.png">

```shell
make generate
go run ./cmd/generate/main.go --prompt 'def add(a, b):'
```

```
def add(a, b):
    if b == 0:
        return (a)
    return a + b

print(a, b)
```

## Supervised Fine-Tuning (SFT)

```shell
%  make sft
go run ./cmd/sft/main.go
SFT          100%|██████████████████████████████| 500/500
```

<img src="https://github.com/itsubaki/gpt/blob/gob/loss_sft.png">

```shell
% make chat
go run ./cmd/chat/main.go --prompt 'Write loop'
```

```
### Instruction:
Write loop

### Response:
for i in range(10):
    print(i)
```

```
### Instruction:
Hi, who are you?

### Response:
I'm an AI assistant. What do you need help with?
```

```
### Instruction:
3+9=

### Response:
12
```

## Group Relative Policy Optimization (GRPO)

```shell
%  make grpo
go run ./cmd/grpo/main.go
GRPO         100%|██████████████████████████████| 100/100
```

<img src="https://github.com/itsubaki/gpt/blob/gob/loss_grpo.png">


```shell
% make eval
go run ./cmd/eval/main.go --batch-size 100
6+8=14 true
5+5=10 true
8+8=15 false
...
7+2=9  true

accuracy: 99 %
```

## References

- [ゼロから作るDeep Learning ❻](https://www.oreilly.co.jp/books/9784814401611/)
- [oreilly-japan/deep-learning-from-scratch-6](https://github.com/oreilly-japan/deep-learning-from-scratch-6)
