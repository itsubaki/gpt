SHELL := /bin/bash

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

.PHONY: testdata
testdata:
	curl -fs -o testdata/merge_rules.gob    https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/merge_rules.gob
	curl -fs -o testdata/tiny_codes.bin     https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/tiny_codes.bin
	curl -fs -o testdata/model_gpt.gob      https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/model_gpt.gob
	curl -fs -o testdata/model_gpt_sft.gob  https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/model_gpt_sft.gob
	curl -fs -o testdata/model_gpt_grpo.gob https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/model_gpt_grpo.gob

dl:
	curl -fs -o testdata/tiny_codes.txt      https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes.txt
	curl -fs -o testdata/tiny_codes_sft.json https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes_sft.json

tokenize:
	rm -f testdata/merge_rules.gob
	rm -f testdata/tiny_codes.bin
	go run ./cmd/tokenize -f testdata/tiny_codes.txt -vocab-size 1000

pretrain:
	caffeinate -i go run ./cmd/pretrain/main.go
	plot -x-max 21000 -y-max 8 loss.csv

generate:
	go run ./cmd/generate/main.go

sft:
	caffeinate -i go run ./cmd/sft/main.go
	plot -x-max 510 -y-max 7 loss_sft.csv

chat:
	go run ./cmd/chat/main.go

.PHONY: grpo
grpo:
	caffeinate -i go run ./cmd/grpo/main.go
	plot -x-max 110 -y-max 110 loss_grpo.csv

example:
	go run ./cmd/generate/main.go --temperature 0.3 --prompt 'def add(a, b):'
	go run ./cmd/generate/main.go --temperature 0.3 --prompt 'def factorial(n):'
	go run ./cmd/generate/main.go --temperature 0.3 --prompt 'def fibonacci(n):'
	go run ./cmd/generate/main.go --temperature 0.3 --prompt 'def is_prime(n):'
	go run ./cmd/generate/main.go --prompt 'def'
	go run ./cmd/chat/main.go --prompt 'Write is_prime function'
	go run ./cmd/chat/main.go --prompt 'Hi, who are you?'
	go run ./cmd/chat/main.go --prompt '3+9='

eval:
	go run ./cmd/eval/main.go --model-path testdata/model_gpt_sft.gob  --batch-size 20
	go run ./cmd/eval/main.go --model-path testdata/model_gpt_grpo.gob --batch-size 20
