package dataloader

var _ Dataset[int] = (*TokenDataset)(nil)

type TokenDataset struct {
	Tokens     []int
	ContextLen int
}

func (s *TokenDataset) Len() int {
	return len(s.Tokens) - s.ContextLen
}

func (s *TokenDataset) GetItem(i int) ([]int, []int) {
	x := s.Tokens[i : i+s.ContextLen]
	y := s.Tokens[i+1 : i+s.ContextLen+1]
	return x, y
}
