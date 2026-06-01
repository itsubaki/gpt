package tokenizer

type Cache struct {
	c map[Pair]map[string]struct{}
}

func NewCache() *Cache {
	return &Cache{
		c: make(map[Pair]map[string]struct{}),
	}
}

func (c *Cache) Add(pair Pair, key string) {
	if _, ok := c.c[pair]; !ok {
		c.c[pair] = make(map[string]struct{})
	}

	c.c[pair][key] = struct{}{}
}

func (c *Cache) Get(pair Pair) map[string]struct{} {
	return c.c[pair]
}

func (c *Cache) Delete(pair Pair, key ...string) {
	if c.c[pair] == nil {
		return
	}

	if len(key) == 0 {
		delete(c.c, pair)
		return
	}

	delete(c.c[pair], key[0])
	if len(c.c[pair]) == 0 {
		delete(c.c, pair)
	}
}
