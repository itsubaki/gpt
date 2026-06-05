package dataloader

type Dataset[T any] interface {
	Len() int
	GetItem(i int) ([]T, []T)
}
