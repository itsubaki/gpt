package progress

import (
	"fmt"
	"io"
	"strings"
	"time"
)

type ProgressBar struct {
	total     int
	width     int
	desc      string
	startTime time.Time
	lastTime  time.Time
	lastCount int
	writer    io.Writer
}

func NewProgressBar(desc string, total int, w io.Writer) *ProgressBar {
	return &ProgressBar{
		desc:      desc,
		total:     total,
		width:     30,
		startTime: time.Now(),
		lastTime:  time.Now(),
		writer:    w,
	}
}

func (p *ProgressBar) Update(current int) {
	now := time.Now()
	percent := float64(current) / float64(p.total)
	filled := int(percent * float64(p.width))
	if current >= p.total {
		filled = p.width
	}

	// bar
	bar := strings.Repeat("█", filled) + strings.Repeat("-", p.width-filled)
	elapsed := now.Sub(p.startTime).Seconds()

	// speed
	deltaCount := current - p.lastCount
	deltaTime := now.Sub(p.lastTime).Seconds()

	var speed float64
	if deltaTime > 0 {
		speed = float64(deltaCount) / deltaTime
	}

	// eta
	var eta float64
	if current > 0 {
		eta = elapsed / float64(current) * float64(p.total-current)
	}

	// print
	if _, err := fmt.Fprintf(p.writer, "\r%s %3.0f%%|%s| %d/%d [%.1fs<%s, %.1f it/s]",
		fmt.Sprintf("%-12s", p.desc),
		percent*100,
		bar,
		current,
		p.total,
		elapsed,
		format(eta),
		speed,
	); err != nil {
		fmt.Println(err)
	}

	// update last
	p.lastTime = now
	p.lastCount = current

	// finish
	if current >= p.total {
		if _, err := fmt.Fprint(p.writer, "\n"); err != nil {
			fmt.Println(err)
		}
	}
}

func format(sec float64) string {
	if sec >= 600*60 {
		return fmt.Sprintf("%.1fh", sec/3600)
	}

	if sec >= 600 {
		return fmt.Sprintf("%.1fm", sec/60)
	}

	return fmt.Sprintf("%.1fs", sec)
}
