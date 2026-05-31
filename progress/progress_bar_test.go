package progress_test

import (
	"bytes"
	"fmt"
	"strings"

	"github.com/itsubaki/gpt/progress"
)

func ExampleProgressBar() {
	var buf bytes.Buffer
	p := progress.NewProgressBar("TEST", 3, &buf)
	for i := range 4 {
		p.Update(i)
	}

	str := buf.String()
	for _, s := range []string{"TEST", "0/3", "1/3", "2/3", "3/3"} {
		if !strings.Contains(str, s) {
			panic(fmt.Sprintf("missing %q in progress bar output", s))
		}
	}

	// Output:
}
