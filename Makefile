SHELL := /bin/bash

test:
	go test -cover $(shell go list ./... | grep -v /cmd/ ) -v -coverprofile=coverage.txt -covermode=atomic
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
	curl -fs -o testdata/merge_rules.gob   https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/merge_rules.gob
	curl -fs -o testdata/tiny_codes.bin    https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/tiny_codes.bin
	curl -fs -o testdata/model_gpt.gob     https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/model_gpt.gob
	curl -fs -o testdata/model_gpt_sft.gob https://raw.githubusercontent.com/itsubaki/gpt/refs/heads/gob/testdata/model_gpt_sft.gob

dl:
	curl -fs -o testdata/tiny_codes.txt      https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes.txt
	curl -fs -o testdata/tiny_codes_sft.json https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes_sft.json

tokenize:
	rm -f testdata/merge_rules.gob
	rm -f testdata/tiny_codes.bin
	go run ./cmd/tokenize -f testdata/tiny_codes.txt -vocab-size 1000

pretrain:
	caffeinate -i go run ./cmd/pretrain/main.go
	plot loss.csv

generate:
	go run ./cmd/generate/main.go

sft:
	caffeinate -i go run ./cmd/sft/main.go
	plot loss_sft.csv

chat:
	go run ./cmd/chat/main.go

example:
	go run ./cmd/generate/main.go --model-path testdata/model_gpt.gob --temperature 0.3 --prompt 'def add(a, b):'
	go run ./cmd/generate/main.go --model-path testdata/model_gpt.gob --temperature 0.3 --prompt 'def factorial(n):'
	go run ./cmd/generate/main.go --model-path testdata/model_gpt.gob --temperature 0.3 --prompt 'def fibonacci(n):'
	go run ./cmd/generate/main.go --model-path testdata/model_gpt.gob --temperature 0.3 --prompt 'def is_prime(n):'
	go run ./cmd/generate/main.go --model-path testdata/model_gpt.gob --prompt 'def'
	go run ./cmd/chat/main.go --prompt 'Write is_prime function'
	go run ./cmd/chat/main.go --prompt 'Hi, who are you?'
	go run ./cmd/chat/main.go --prompt '3+7'
