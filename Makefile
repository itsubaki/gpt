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

dl:
	curl -fs -o testdata/tiny_codes.txt https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes.txt

tokenize:
	rm -f testdata/merge_rules.gob
	rm -f testdata/tiny_codes.bin
	go run ./cmd/tokenize -f testdata/tiny_codes.txt -vocab-size 1000

gpt:
	go run ./cmd/gpt/main.go
	plot loss.csv
