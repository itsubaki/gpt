SHELL := /bin/bash

RELEASE_TAG ?= weights-v0
WEIGHTS := \
	merge_rules.gob \
	tiny_codes.bin \
	model_gpt.gob \
	model_gpt_sft.gob \
	model_gpt_grpo.gob

test:
	go test -cover $(shell { go list ./... | grep -v '/cmd/'; go list ./cmd/grpo/grpo; }) -v -coverprofile=coverage.txt -covermode=atomic
	go tool cover -html=coverage.txt -o coverage.html

lint:
	golangci-lint run

pprof:
	go tool pprof -http=:8080 cpu.prof

update:
	GOPROXY=direct go get github.com/itsubaki/autograd@HEAD
	go get -u
	go mod tidy
	pinact run -u

install:
	go install github.com/itsubaki/plot@latest

.PHONY: release
release:
	gh release create $(RELEASE_TAG) \
		$(addprefix testdata/,$(WEIGHTS)) \
		--repo itsubaki/gpt \
		--title "$(RELEASE_TAG)" \
		--notes "GPT model weights" \
		--latest

.PHONY: testdata
testdata:
	@for file in $(WEIGHTS); do \
		curl -fLsS --retry 3 -o testdata/$$file https://github.com/itsubaki/gpt/releases/download/$(RELEASE_TAG)/$$file || exit 1; \
	done

dl:
	curl -fs -o testdata/tiny_codes.txt      https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes.txt
	curl -fs -o testdata/tiny_codes_sft.json https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes_sft.json

bpetrain:
	rm -f testdata/merge_rules.gob
	rm -f testdata/tiny_codes.bin
	go run cmd/bpetrain/main.go -f testdata/tiny_codes.txt -vocab-size 1000

tokenize:
	go run cmd/tokenize/main.go --text 'def is_prime(n: int) -> bool: return n >= 2 and all(n % i for i in range(2, int(n**0.5) + 1))'

pretrain:
	caffeinate -i go run cmd/pretrain/main.go
	plot loss.csv

generate:
	go run cmd/generate/main.go

sft:
	caffeinate -i go run cmd/sft/main.go
	plot loss_sft.csv

chat:
	go run cmd/chat/main.go

.PHONY: grpo
grpo:
	caffeinate -i go run cmd/grpo/main.go
	plot loss_grpo.csv

example:
	go run cmd/generate/main.go --temperature 0.3 --prompt 'def add(a, b):'
	go run cmd/generate/main.go --temperature 0.3 --prompt 'def factorial(n):'
	go run cmd/generate/main.go --temperature 0.3 --prompt 'def fibonacci(n):'
	go run cmd/generate/main.go --temperature 0.3 --prompt 'def is_prime(n):'
	go run cmd/generate/main.go --prompt 'def'
	go run cmd/chat/main.go --prompt 'Write a is_prime function'
	go run cmd/chat/main.go --prompt 'Who are you?'
	go run cmd/chat/main.go --prompt '3+9='

eval:
	go run cmd/eval/main.go --model-path testdata/model_gpt_sft.gob  --batch-size 20
	go run cmd/eval/main.go --model-path testdata/model_gpt_grpo.gob --batch-size 20
