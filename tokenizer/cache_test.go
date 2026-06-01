package tokenizer_test

import (
	"fmt"

	"github.com/itsubaki/gpt/tokenizer"
)

func ExampleCache() {
	cache := tokenizer.NewCache()
	pair := tokenizer.Pair{1, 2}
	key := "test"

	cache.Add(pair, key)
	v := cache.Get(pair)
	fmt.Println(v)

	cache.Delete(pair, key)
	v = cache.Get(pair)
	fmt.Println(v)

	// Output:
	// map[test:{}]
	// map[]
}
