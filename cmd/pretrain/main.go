package main

import (
	"encoding/csv"
	"encoding/gob"
	"flag"
	"fmt"
	"math/rand/v2"
	"os"
	"runtime/pprof"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/dataloader"
	"github.com/itsubaki/gpt/model"
	"github.com/itsubaki/gpt/progress"
	"github.com/itsubaki/gpt/scheduler"
	"github.com/itsubaki/gpt/tokenizer"
)

func main() {
	// parameters
	var contextLen, vocabSize, batchSize, embeddim, numOfHeads, numOfBlocks int
	var theta, maxLR, beta1, beta2, clip, weightDecay float64
	var warmupIters, maxIters int
	var usePProf bool
	var mergeRulesPath, prompt string
	var temperature float64
	var maxNewTokens int
	flag.IntVar(&contextLen, "context-len", 256, "maximum context length")
	flag.IntVar(&vocabSize, "vocab-size", 1000, "vocabulary size")
	flag.IntVar(&batchSize, "batch-size", 32, "batch size")
	flag.IntVar(&embeddim, "embeddim", 384, "embedding dimension")
	flag.IntVar(&numOfHeads, "num-of-heads", 6, "number of heads")
	flag.IntVar(&numOfBlocks, "num-of-blocks", 6, "number of blocks")
	flag.Float64Var(&theta, "theta", 10000.0, "theta for positional encoding")
	flag.Float64Var(&maxLR, "max-learning-rate", 3e-4, "maximum learning rate")
	flag.Float64Var(&beta1, "beta1", 0.9, "beta1 for AdamW optimizer")
	flag.Float64Var(&beta2, "beta2", 0.999, "beta2 for AdamW optimizer")
	flag.Float64Var(&clip, "clip", 1.0, "gradient clipping value")
	flag.Float64Var(&weightDecay, "weight-decay", 0.01, "weight decay for AdamW optimizer")
	flag.IntVar(&warmupIters, "warmup-iters", 5, "number of warmup iterations")
	flag.IntVar(&maxIters, "max-iters", 1000, "number of maximum iterations")
	flag.BoolVar(&usePProf, "pprof", false, "enable pprof")
	flag.StringVar(&mergeRulesPath, "merge-rules-path", "testdata/merge_rules.gob", "path to the merge rules gob file")
	flag.StringVar(&prompt, "prompt", "def", "prompt for text generation")
	flag.Float64Var(&temperature, "temperature", 1.0, "temperature for sampling")
	flag.IntVar(&maxNewTokens, "max-new-tokens", 200, "maximum number of new tokens to generate")
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

	// tokenizer
	mergeRules, ok := tokenizer.Load(mergeRulesPath)
	if !ok {
		panic("failed to load merge rules")
	}

	// generate text
	generatedText := Generate(
		m,
		tokenizer.NewBPETokenizer(mergeRules),
		prompt,
		maxNewTokens,
		temperature,
	)

	fmt.Println(generatedText)
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

var _ Tokenizer = (*tokenizer.BPETokenizer)(nil)

var _ Model = (*model.GPT)(nil)

type Tokenizer interface {
	Encode(text string) []int
	Decode(tokens []int) string
	EndTokenID() int
}

type Model interface {
	Forward(x *variable.Variable) *variable.Variable
	MaxContextLen() int
}

func Generate(
	model Model,
	tokenizer Tokenizer,
	prompt string,
	maxNewTokens int,
	temperature float64,
) string {
	ids := tokenizer.Encode(prompt)
	generatedIDs := make([]int, len(ids))
	copy(generatedIDs, ids)

	func() {
		// disable gradient tracking for generation
		defer variable.Nograd().End()

		// generate tokens
		for range maxNewTokens {
			if len(ids) > model.MaxContextLen() {
				// keep only the last maxContextLen tokens as input
				ids = ids[len(ids)-model.MaxContextLen():]
			}

			// forward
			x := newVariable(ids).Reshape(1, -1)                     // (1, C)
			logits := model.Forward(x)                               // (1, C, V)
			logits = F.GetItem(1, []int{logits.Size(1) - 1})(logits) // (1, 1, V)
			logits = F.Reshape(-1)(logits)                           // (V)

			// sample next token
			probs := F.Softmax(-1)(F.MulC(1.0/temperature, logits))
			nextID := multinominal(probs)

			// stop if end token is generated
			if nextID == tokenizer.EndTokenID() {
				break
			}

			// append next token to input and generated tokens
			ids = append(ids, nextID)
			generatedIDs = append(generatedIDs, nextID)
		}
	}()

	// decode generated tokens to text
	generatedText := tokenizer.Decode(generatedIDs)
	return generatedText
}

func newVariable(x []int) *variable.Variable {
	f := make([]float64, len(x))
	for i, v := range x {
		f[i] = float64(v)
	}

	return variable.New(f...)
}

func multinominal(probs *variable.Variable) int {
	r := rand.Float64()

	var cum float64
	for i := range probs.Size() {
		cum += probs.At(i)
		if r < cum {
			return i
		}
	}

	return probs.Size() - 1
}
