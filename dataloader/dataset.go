package dataloader

type Dataset interface {
	Len() int
	GetItem(i int) ([]int, []int)
}
