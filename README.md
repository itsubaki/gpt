# gpt

[![PkgGoDev](https://pkg.go.dev/badge/github.com/itsubaki/gpt)](https://pkg.go.dev/github.com/itsubaki/gpt)
[![tests](https://github.com/itsubaki/gpt/workflows/tests/badge.svg)](https://github.com/itsubaki/gpt/actions)

## Quick Start

``` shell
% make testdata example
```

```shell
curl -fs -o testdata/merge_rules.gob    https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/embed-dim-256s/testdata/merge_rules.gob
curl -fs -o testdata/tiny_codes.bin     https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/embed-dim-256s/testdata/tiny_codes.bin
curl -fs -o testdata/model_gpt.gob      https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/embed-dim-256s/testdata/model_gpt.gob
curl -fs -o testdata/model_gpt_sft.gob  https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/embed-dim-256s/testdata/model_gpt_sft.gob
curl -fs -o testdata/model_gpt_grpo.gob https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/embed-dim-256s/testdata/model_gpt_grpo.gob
```

```python
### Instruction:
Write a is_prime function

### Response:
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True
```

## Architecture

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

## BPE Tokenizer Training

```shell
% make dl
curl -fs -o testdata/tiny_codes.txt      https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes.txt
curl -fs -o testdata/tiny_codes_sft.json https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes_sft.json
```

```shell
% make bpetrain
go run cmd/bpetrain/main.go -vocab-size 1000
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

```shell
% make tokenize
go run cmd/tokenize/main.go --text 'def is_prime(n: int) -> bool: return n >= 2 and all(n % i for i in range(2, int(n**0.5) + 1))'
"def"(300) " is"(382) "_"( 95) "prime"(830) "("( 40) "n"(110) ":"( 58) " int"(888) ")"( 41) " -"(440) ">"( 62) " b"(358) "o"(111) "ol"(412) ":"( 58) " "( 32) "return"(301) " n"(289) " >"(523) "="( 61) " 2"(373) " and"(409) " all"(905) "("( 40) "n"(110) " %"(590) " i"(284) " for"(406) " i"(284) " in"(286) " range"(391) "("( 40) "2"( 50) ","( 44) " int"(888) "("( 40) "n"(110) "**"(910) "0"( 48)"."( 46) "5"( 53) ")"( 41) " +"(347) " 1"(313) "))"(376)
```

## Pre-Training

```shell
% make pretrain
go run cmd/pretrain/main.go
Pre-Training 100%|██████████████████████████████| 20000/20000
```

<img src="https://github.com/itsubaki/gpt/blob/embed-dim-256s/loss.png">

```shell
% make generate
go run cmd/generate/main.go --prompt 'def add(a, b):'
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
go run cmd/sft/main.go
SFT          100%|██████████████████████████████| 500/500
```

<img src="https://github.com/itsubaki/gpt/blob/embed-dim-256s/loss_sft.png">

```shell
% make chat
go run cmd/chat/main.go --prompt 'Write a loop'
```

```
### Instruction:
Write a loop

### Response:
for i in range(5):
    print(i)
else:
    print('done')
```

```
### Instruction:
Who are you?

### Response:
I'm CodeBot. How can I assist you today?
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
go run cmd/grpo/main.go
GRPO         100%|██████████████████████████████| 100/100
```

<img src="https://github.com/itsubaki/gpt/blob/embed-dim-256s/loss_grpo.png">

```shell
% make eval
go run cmd/eval/main.go --model-path testdata/model_gpt_sft.gob  --batch-size 20
4+6=10 true
9+5=15 false
5+3=6  false
9+4=13 true
...
accuracy: 60 %

go run cmd/eval/main.go --model-path testdata/model_gpt_grpo.gob --batch-size 20
7+3=10 true
3+5=8  true
6+4=10 true
9+4=13 true
...
accuracy: 100 %
```

## References

- [ゼロから作るDeep Learning ❻](https://www.oreilly.co.jp/books/9784814401611/)
- [oreilly-japan/deep-learning-from-scratch-6](https://github.com/oreilly-japan/deep-learning-from-scratch-6)
