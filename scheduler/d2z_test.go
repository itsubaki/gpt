package scheduler_test

import (
	"fmt"

	"github.com/itsubaki/gpt/scheduler"
)

func ExampleD2Z() {
	sched := &scheduler.D2Z{
		MaxLearningRate: 0.1,
		WarmupIters:     5,
		MaxIters:        10,
	}

	for i := range 12 {
		lr := sched.GetLearningRate(i)
		fmt.Printf("%2d: %.2f\n", i, lr)
	}

	// Output:
	//  0: 0.00
	//  1: 0.02
	//  2: 0.04
	//  3: 0.06
	//  4: 0.08
	//  5: 0.10
	//  6: 0.08
	//  7: 0.06
	//  8: 0.04
	//  9: 0.02
	// 10: 0.00
	// 11: 0.00
}
