package main

import (
	"flag"
)

func main() {
	var mergeRulesPath, modelPath string
	var maxIters, batchSize, groupSize, updatePerGeneration int
	var learningRate, epsilon float64
	flag.StringVar(&mergeRulesPath, "merge-rules-path", "testdata/merge_rules.gob", "path to the merge rules file")
	flag.StringVar(&modelPath, "model-path", "testdata/model_gpt.gob", "path to the pre-trained model gob file")
	flag.Float64Var(&learningRate, "learning-rate", 7e-6, "learning rate for the optimizer")
	flag.Float64Var(&epsilon, "epsilon", 0.2, "clipping range")
	flag.IntVar(&maxIters, "max-iters", 500, "number of maximum iterations")
	flag.IntVar(&batchSize, "batch-size", 32, "size of each batch")
	flag.IntVar(&groupSize, "group-size", 8, "size of each group")
	flag.IntVar(&updatePerGeneration, "update-per-generation", 2, "number of updates per generation")
	flag.Parse()
}
