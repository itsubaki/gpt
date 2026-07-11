package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"

	"github.com/itsubaki/autograd/hook"
	"github.com/itsubaki/autograd/optimizer"
	"github.com/itsubaki/autograd/variable"
	"github.com/itsubaki/gpt/grpo"
	"github.com/itsubaki/gpt/model"
	"github.com/itsubaki/gpt/progress"
	"github.com/itsubaki/gpt/tokenizer"
)

func main() {
	var mergeRulesPath, modelPath, grpoModelPath string
	var maxIters, batchSize, groupSize, updatePerGeneration int
	var maxLR, beta1, beta2, weightDecay, clip float64
	var epsilon float64
	flag.StringVar(&mergeRulesPath, "merge-rules-path", "testdata/merge_rules.gob", "path to the merge rules file")
	flag.StringVar(&modelPath, "model-path", "testdata/model_gpt.gob", "path to the pre-trained model gob file")
	flag.StringVar(&grpoModelPath, "grpo-model-path", "testdata/model_gpt_grpo.gob", "path to the GRPO model gob file")
	flag.Float64Var(&maxLR, "max-learning-rate", 3e-4, "maximum learning rate")
	flag.Float64Var(&beta1, "beta1", 0.9, "beta1 for AdamW optimizer")
	flag.Float64Var(&beta2, "beta2", 0.999, "beta2 for AdamW optimizer")
	flag.Float64Var(&weightDecay, "weight-decay", 0.01, "weight decay for AdamW optimizer")
	flag.Float64Var(&clip, "clip", 1.0, "gradient clipping value")
	flag.Float64Var(&epsilon, "epsilon", 0.2, "clipping range")
	flag.IntVar(&maxIters, "max-iters", 500, "number of maximum iterations")
	flag.IntVar(&batchSize, "batch-size", 2, "size of each batch")
	flag.IntVar(&groupSize, "group-size", 8, "size of each group")
	flag.IntVar(&updatePerGeneration, "update-per-generation", 2, "number of updates per generation")
	flag.Parse()

	// model from gob file
	m, err := model.NewGPTFrom(modelPath)
	if err != nil {
		panic(err)
	}

	// old model from gob file
	oldModel, err := model.NewGPTFrom(modelPath)
	if err != nil {
		panic(err)
	}

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

	// tokenizer
	tknizer, err := tokenizer.NewBPETokenizerFrom(mergeRulesPath)
	if err != nil {
		panic(err)
	}

	dataset := grpo.NewDataset(tknizer)
	dataloader := &grpo.DataLoader{
		BatchSize: batchSize,
		Shuffle:   true,
		Dataset:   dataset,
	}

	// save loss to csv
	f, err := os.Create("loss_grpo.csv")
	if err != nil {
		panic(err)
	}
	defer func() { _ = f.Close() }()

	w := csv.NewWriter(f)
	defer w.Flush()

	// progress bar
	bar := progress.NewProgressBar("GRPO", maxIters, os.Stdout)
	bar.Update(0)

	var accs []float64
	var curacc float64
	var loss *variable.Variable
	for i := range maxIters {
		prompts, gts := dataloader.Batch()

		// update old model
		oldModel.Load(m.Params())

		// sample group of prompts and responses
		allPrompts, allResponses, allAdvantages := grpo.GenerateGroup(
			oldModel,
			oldModel.MaxContextLen,
			tknizer,
			prompts,
			gts,
			groupSize,
		)

		// get batch of ids and mask
		ids, mask := dataset.GetBatch(allPrompts, allResponses)

		for range updatePerGeneration {
			m.Cleargrads()
			loss = grpo.Loss(
				m,
				oldModel,
				ids,
				mask,
				allAdvantages,
				epsilon,
			)

			loss.Backward()
			o.Update(m)
		}

		// flush loss
		if err := write(w, i, loss.At()); err != nil {
			panic(err)
		}

		// checkpoint
		if i%100 == 0 {
			if err := m.Save(grpoModelPath); err != nil {
				panic(err)
			}
		}

		// evaluate accuracy
		if i%10 == 0 {
			func() {
				defer variable.Nograd().End()

				var correct, total float64
				for j := range dataset.Len() {
					prompt, gt := dataset.GetItem(j)
					response := model.GenerateText(
						m,
						m.MaxContextLen,
						tknizer,
						prompt,
						1000, // max new tokens
						0.0,  // temperature
					)

					reward := grpo.Reward(gt, response)
					if reward > 0 {
						correct++
					}

					total++
				}

				curacc = float64(correct) / float64(total)
				accs = append(accs, curacc)
			}()
		}

		// update progress bar
		bar.Update(i+1, fmt.Sprintf("loss=%.4f, acc=%.4f", loss.At(), curacc))
	}
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
