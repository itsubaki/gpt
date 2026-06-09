package main

import (
	"encoding/csv"
	"encoding/gob"
	"flag"
	"fmt"
	"os"
	"runtime/pprof"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/gpt/dataloader"
	"github.com/itsubaki/gpt/model"
	"github.com/itsubaki/gpt/progress"
)

func main() {
	// parameters
	var vocabSize, contextLen, embeddim, numOfHeads, numOfBlocks int
	var maxLR, beta1, beta2, weightDecay, clip float64
	var maxIters, batchSize int
	var tokensPath, modelPath string
	var usePProf bool
	flag.IntVar(&maxIters, "max-iters", 40000, "number of maximum iterations")
	flag.IntVar(&batchSize, "batch-size", 32, "batch size")
	flag.IntVar(&vocabSize, "vocab-size", 1000, "vocabulary size")
	flag.IntVar(&contextLen, "context-len", 128, "maximum context length")
	flag.IntVar(&embeddim, "embeddim", 192, "embedding dimension")
	flag.IntVar(&numOfHeads, "num-of-heads", 6, "number of heads")
	flag.IntVar(&numOfBlocks, "num-of-blocks", 6, "number of blocks")
	flag.Float64Var(&maxLR, "max-learning-rate", 3e-4, "maximum learning rate")
	flag.Float64Var(&beta1, "beta1", 0.9, "beta1 for AdamW optimizer")
	flag.Float64Var(&beta2, "beta2", 0.999, "beta2 for AdamW optimizer")
	flag.Float64Var(&weightDecay, "weight-decay", 0.01, "weight decay for AdamW optimizer")
	flag.Float64Var(&clip, "clip", 1.0, "gradient clipping value")
	flag.StringVar(&tokensPath, "tokens-path", "testdata/tiny_codes.bin", "path to the tokens gob file")
	flag.StringVar(&modelPath, "model-path", "testdata/model_gpt.gob", "path to the model gob file")
	flag.BoolVar(&usePProf, "pprof", false, "enable pprof")
	flag.Parse()

	if usePProf {
		f, err := os.Create("cpu.prof")
		if err != nil {
			panic(err)
		}
		defer func() {
			if err := f.Close(); err != nil {
				fmt.Println(err)
			}
		}()

		if err := pprof.StartCPUProfile(f); err != nil {
			panic(err)
		}
		defer pprof.StopCPUProfile()
	}

	// model
	m := model.NewGPT(
		vocabSize,
		contextLen,
		embeddim,
		numOfHeads,
		numOfBlocks,
	)

	// optimizer
	o := optimizer.AdamW{
		Alpha:       maxLR,
		Beta1:       beta1,
		Beta2:       beta2,
		WeightDecay: weightDecay,
		Hook: []optimizer.Hook{
			hook.ClipGrad(clip),
		},
	}

	// dataloader
	tokens, err := load(tokensPath)
	if err != nil {
		panic(err)
	}

	loader := dataloader.DataLoader{
		BatchSize: batchSize,
		Shuffle:   true,
		Dataset: &dataloader.TokenDataset{
			Tokens:     tokens,
			ContextLen: contextLen,
		},
	}

	// progress bar
	bar := progress.NewProgressBar("Pre-Training", maxIters, os.Stdout)
	bar.Update(0)

	// save loss to csv
	f, err := os.Create("loss.csv")
	if err != nil {
		panic(err)
	}
	defer func() { _ = f.Close() }()

	w := csv.NewWriter(f)
	defer w.Flush()

	// training loop
	minLoss := 1.0
	for i := range maxIters {
		// batch
		x, y := loader.Batch()

		// forward
		logits := m.Forward(x)
		loss := F.CrossEntropy(
			F.Reshape(-1, logits.Size(-1))(logits), // (B, C, V) -> (B*C, V)
			F.Reshape(-1)(y),                       // (B, C) -> (B*C)
		)

		// backward and update
		m.Cleargrads()
		loss.Backward()
		o.Update(m)

		// update progress bar
		bar.Update(i+1, fmt.Sprintf("loss=%.4f", loss.At()))

		// flush loss
		if err := write(w, i, loss.At()); err != nil {
			panic(err)
		}

		// save model if loss is improved
		if loss.At() < minLoss {
			if err := m.Save(modelPath + ".min"); err != nil {
				panic(err)
			}

			minLoss = loss.At()
			fmt.Println()
			fmt.Printf("iter %d: loss=%.4f (saved)\n", i, loss.At())
		}

		if i%100 == 0 {
			if err := m.Save(modelPath); err != nil {
				panic(err)
			}

			fmt.Println()
			fmt.Printf("iter %d: loss=%.4f (saved)\n", i, loss.At())
		}
	}

	// save final model
	if err := m.Save(modelPath); err != nil {
		panic(err)
	}

	fmt.Println()
}

func load(path string) ([]int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()

	var ids []int
	if err := gob.NewDecoder(f).Decode(&ids); err != nil {
		return nil, err
	}

	return ids, nil
}

func write(w *csv.Writer, iter int, loss float64) error {
	if err := w.Write([]string{
		fmt.Sprintf("%d", iter),
		fmt.Sprintf("%.4f", loss),
	}); err != nil {
		return err
	}

	w.Flush()
	if err := w.Error(); err != nil {
		return err
	}

	return nil
}
