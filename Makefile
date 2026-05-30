SHELL := /bin/bash

test:
	go test -cover $(shell go list ./... | grep -v /cmd/ ) -v -coverprofile=coverage.txt -covermode=atomic
	go tool cover -html=coverage.txt -o coverage.html

lint:
	golangci-lint run

dl:
	curl -fs -o testdata/tiny_codes.txt https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-6/refs/heads/main/codebot/tiny_codes.txt


tokenize:
	go run ./cmd/tokenizer -f testdata/tiny_codes.txt -vocab-size 1000