package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"runtime/pprof"

	F "github.com/itsubaki/autograd/function"
	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/math"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/gpt/dataloader"
	"github.com/itsubaki/gpt/model"
	"github.com/itsubaki/gpt/progress"
	"github.com/itsubaki/gpt/tokenizer"
)

func main() {
	var contextLen int
	var mergeRulesPath, modelPath, alpacaPath, sftModelPath string
	var learningRate, beta1, beta2, weightDecay, clip float64
	var maxIters, batchSize int
	var usePProf bool
	var minLoss float64
	flag.IntVar(&contextLen, "context-len", 256, "maximum context length")
	flag.Float64Var(&learningRate, "learning-rate", 3e-4, "learning rate")
	flag.Float64Var(&beta1, "beta1", 0.9, "beta1 for AdamW optimizer")
	flag.Float64Var(&beta2, "beta2", 0.999, "beta2 for AdamW optimizer")
	flag.Float64Var(&weightDecay, "weight-decay", 0.01, "weight decay for AdamW optimizer")
	flag.Float64Var(&clip, "clip", 1.0, "gradient clipping value")
	flag.IntVar(&maxIters, "max-iters", 500, "number of maximum iterations for fine-tuning")
	flag.IntVar(&batchSize, "batch-size", 32, "batch size for fine-tuning")
	flag.StringVar(&mergeRulesPath, "merge-rules-path", "testdata/merge_rules.gob", "path to the merge rules file")
	flag.StringVar(&modelPath, "model-path", "testdata/model_gpt.gob", "path to the pre-trained model gob file")
	flag.StringVar(&alpacaPath, "alpaca-path", "testdata/tiny_codes_sft.json", "path to the Alpaca data JSON file")
	flag.StringVar(&sftModelPath, "sft-model-path", "testdata/model_gpt_sft.gob", "path to the SFT model gob file")
	flag.BoolVar(&usePProf, "pprof", false, "enable pprof")
	flag.Float64Var(&minLoss, "min-loss", 1.0, "minimum loss for saving the model")
	flag.Parse()

	if usePProf {
		f, err := os.Create("cpu_sft.prof")
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

	// tokenizer
	rulesFile, err := os.Open(mergeRulesPath)
	if err != nil {
		panic(err)
	}
	defer func() { _ = rulesFile.Close() }()

	mergeRules, err := tokenizer.NewDefaultDictFrom(rulesFile)
	if err != nil {
		panic(err)
	}

	// model from gob file
	modelFile, err := os.Open(modelPath)
	if err != nil {
		panic(err)
	}
	defer func() { _ = modelFile.Close() }()

	m, err := model.NewGPTFrom(modelFile)
	if err != nil {
		panic(err)
	}

	// optimizer
	o := optimizer.AdamW{
		Alpha:       float32(learningRate),
		Beta1:       float32(beta1),
		Beta2:       float32(beta2),
		WeightDecay: float32(weightDecay),
	}

	// dataloader
	loader := dataloader.DataLoader{
		BatchSize: batchSize,
		Shuffle:   true,
		Dataset: dataloader.NewAlpacaDataset(
			dataloader.MustLoadAlpaca(alpacaPath),
			tokenizer.NewBPETokenizer(mergeRules),
			contextLen,
		),
	}

	// progress bar
	bar := progress.NewProgressBar("SFT", maxIters, os.Stdout)
	bar.Update(0)

	// save loss to csv
	f, err := os.Create("loss_sft.csv")
	if err != nil {
		panic(err)
	}
	defer func() { _ = f.Close() }()

	w := csv.NewWriter(f)
	defer w.Flush()

	var min float32 = float32(minLoss)
	for i := range maxIters {
		// batch
		x, y := loader.Batch()

		// forward
		logits := m.Forward(x)
		loss := F.CrossEntropy(
			F.Reshape(-1, logits.Size(-1))(logits), // (B, C, V) -> (B*C, V)
			F.Reshape(-1)(y),                       // (B, C) -> (B*C)
			// ignore index -100
		)

		// backward and update
		m.Cleargrads()
		loss.Backward()
		hook.ClipGrad(float32(clip))(m.Params())
		o.Update(m.Params())

		// flush loss
		if err := write(w, i, loss.At()); err != nil {
			panic(err)
		}

		// model checkpoint
		if i%100 == 0 {
			if err := save(sftModelPath, m); err != nil {
				panic(err)
			}
		}

		if loss.At() < min {
			if err := save(sftModelPath+".min", m); err != nil {
				panic(err)
			}

			min = loss.At()
		}

		// update progress bar
		bar.Update(i+1, fmt.Sprintf("loss=%.4f(ppl=%.4f)", loss.At(), math.Exp(loss.At())))
	}

	// save final model
	if err := save(sftModelPath, m); err != nil {
		panic(err)
	}

	fmt.Println()
}

func write(w *csv.Writer, iter int, loss float32) error {
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

func save(path string, m *model.GPT) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create: %w", err)
	}
	defer func() { _ = f.Close() }()

	if err := m.Save(f); err != nil {
		return fmt.Errorf("save: %w", err)
	}

	return nil
}
