package main

import (
	"encoding/csv"
	"encoding/gob"
	"flag"
	"fmt"
	"math"
	"os"
	"runtime/pprof"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/gpt/dataloader"
	"github.com/itsubaki/gpt/layer"
	"github.com/itsubaki/gpt/model"
	"github.com/itsubaki/gpt/progress"
	"github.com/itsubaki/gpt/scheduler"
)

func init() {
	gob.Register(&layer.LinearT{})
	gob.Register(&layer.BlockT{})
	gob.Register(&layer.RMSNormT{})
	gob.Register(&layer.EmbeddingsT{})
	gob.Register(&layer.MultiHeadAttentionT{})
	gob.Register(&layer.SwiGLUT{})
}

func main() {
	// parameters
	var contextLen, vocabSize, batchSize, embeddim, numOfHeads, numOfBlocks int
	var theta, maxLR, beta1, beta2, weightDecay, clip float64
	var warmupIters, maxIters int
	var usePProf bool
	flag.IntVar(&contextLen, "context-len", 256, "maximum context length")
	flag.IntVar(&vocabSize, "vocab-size", 1000, "vocabulary size")
	flag.IntVar(&batchSize, "batch-size", 32, "batch size")
	flag.IntVar(&embeddim, "embeddim", 384, "embedding dimension")
	flag.IntVar(&numOfHeads, "num-of-heads", 6, "number of heads")
	flag.IntVar(&numOfBlocks, "num-of-blocks", 6, "number of blocks")
	flag.Float64Var(&theta, "theta", 10000.0, "theta for positional encoding")
	flag.Float64Var(&maxLR, "max-learning-rate", 1e-4, "maximum learning rate")
	flag.Float64Var(&beta1, "beta1", 0.9, "beta1 for AdamW optimizer")
	flag.Float64Var(&beta2, "beta2", 0.999, "beta2 for AdamW optimizer")
	flag.Float64Var(&weightDecay, "weight-decay", 0.001, "weight decay for AdamW optimizer")
	flag.Float64Var(&clip, "clip", 1.0, "gradient clipping value")
	flag.IntVar(&warmupIters, "warmup-iters", 1000, "number of warmup iterations")
	flag.IntVar(&maxIters, "max-iters", 10000, "number of maximum iterations")
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
		4*embeddim, // ffdim
		theta,
	)

	// optimizer
	o := optimizer.AdamW{
		Adam: optimizer.Adam{
			Alpha: maxLR,
			Beta1: beta1,
			Beta2: beta2,
			Hook: []optimizer.Hook{
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
		fmt.Println("load tokens:", err)
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

	// save loss to csv
	f, err := os.Create("loss.csv")
	if err != nil {
		panic(err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	defer w.Flush()

	// training loop
	losses := make([]float64, 0, maxIters)
	minLoss := math.MaxFloat64
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

		if loss.At() < minLoss {
			minLoss = loss.At()
			if err := m.Save("testdata/model.gob"); err != nil {
				fmt.Println("save model:", err)
			}
		}

		losses = append(losses, loss.At())

		// backward and update
		m.Cleargrads()
		loss.Backward()
		o.Update(m)

		// update progress bar
		bar.Update(i + 1)

		// flush loss
		if err := w.Write([]string{
			fmt.Sprintf("%d", i),
			fmt.Sprintf("%.4f", losses[len(losses)-1]),
		}); err != nil {
			panic(err)
		}

		w.Flush()
		if err := w.Error(); err != nil {
			panic(err)
		}
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
