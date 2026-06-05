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
	O "github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/gpt/dataloader"
	"github.com/itsubaki/gpt/model"
	"github.com/itsubaki/gpt/progress"
	"github.com/itsubaki/gpt/scheduler"
)

func main() {
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

	// hyper parameters
	var contextLen, vocabSize, batchSize, embeddim, numOfHeads, numOfBlocks int
	var theta, maxLR, beta1, beta2, clip, weightDecay float64
	var warmupIters, maxIters int
	flag.IntVar(&contextLen, "context-len", 256, "maximum context length")
	flag.IntVar(&vocabSize, "vocab-size", 1000, "vocabulary size")
	flag.IntVar(&batchSize, "batch-size", 16, "batch size")
	flag.IntVar(&embeddim, "embeddim", 192, "embedding dimension")
	flag.IntVar(&numOfHeads, "num-of-heads", 3, "number of heads")
	flag.IntVar(&numOfBlocks, "num-of-blocks", 3, "number of blocks")
	flag.Float64Var(&theta, "theta", 10000.0, "theta for positional encoding")
	flag.Float64Var(&maxLR, "max-learning-rate", 3e-4, "maximum learning rate")
	flag.Float64Var(&beta1, "beta1", 0.9, "beta1 for Adam optimizer")
	flag.Float64Var(&beta2, "beta2", 0.999, "beta2 for Adam optimizer")
	flag.Float64Var(&clip, "clip", 1.0, "gradient clipping value")
	flag.Float64Var(&weightDecay, "weight-decay", 0.01, "weight decay for AdamW optimizer")
	flag.IntVar(&warmupIters, "warmup-iters", 10, "number of warmup iterations")
	flag.IntVar(&maxIters, "max-iters", 200, "number of maximum iterations")
	flag.Parse()

	// model
	m := model.NewGPT(
		vocabSize,
		contextLen,
		embeddim,
		numOfHeads,
		numOfBlocks,
		4*embeddim, // ffdim
		theta,
	)

	// optimizer
	o := O.AdamW{
		Adam: O.Adam{
			Alpha: maxLR,
			Beta1: beta1,
			Beta2: beta2,
			Hook: []O.Hook{
				hook.ClipGrad(clip),
			},
		},
		WeightDecay: weightDecay,
	}

	// learning rate scheduler
	sched := scheduler.D2Z{
		MaxLearningRate: maxLR,
		WarmupIters:     warmupIters,
		MaxIters:        maxIters,
	}

	// dataloader
	tokens, err := load("testdata/tiny_codes.bin")
	if err != nil {
		fmt.Println("failed to load tokens:", err)
		return
	}

	loader := dataloader.DataLoader{
		BatchSize: batchSize,
		Cycle:     true,
		Shuffle:   true,
		Dataset: &dataloader.TokenDataset{
			Tokens:     tokens,
			ContextLen: contextLen,
		},
	}

	// progress bar
	bar := progress.NewProgressBar("iterations", maxIters, os.Stdout)
	bar.Update(0)

	// training loop
	losses := make([]float64, 0, maxIters)
	for i := range maxIters {
		// learning rate scheduling
		o.Alpha = sched.GetLearningRate(i)

		// batch
		x, y := loader.Batch()

		// forward
		logits := m.Forward(x)
		loss := F.CrossEntropy(
			F.Reshape(-1, logits.Size(-1))(logits), // (B, C, V) -> (B*C, V)
			F.Reshape(-1)(y),                       // (B, C) -> (B*C)
		)
		losses = append(losses, loss.At())

		// backward and update
		m.Cleargrads()
		loss.Backward()
		o.Update(m)

		// update progress bar
		bar.Update(i + 1)
	}

	// save losses to CSV
	if err := save("loss.csv", losses); err != nil {
		fmt.Println("failed to save losses:", err)
		return
	}
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

func save(path string, losses []float64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	w := csv.NewWriter(f)
	defer w.Flush()

	for i, loss := range losses {
		if err := w.Write([]string{
			fmt.Sprintf("%d", i),
			fmt.Sprintf("%.4f", loss),
		}); err != nil {
			return err
		}
	}

	return w.Error()
}
