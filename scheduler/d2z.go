package scheduler

// D2Z implements Linear Decay to Zero learning rate scheduler.
type D2Z struct {
	MaxLearningRate float64
	WarmupIters     int
	MaxIters        int
}

// GetLearningRate returns the learning rate for the given iteration.
func (s *D2Z) GetLearningRate(it int) float64 {
	if it < s.WarmupIters {
		return s.MaxLearningRate * float64(it) / float64(s.WarmupIters)
	}

	if it < s.MaxIters {
		progress := float64(it-s.WarmupIters) / float64(s.MaxIters-s.WarmupIters)
		return s.MaxLearningRate * (1.0 - progress)
	}

	return 0.0
}
