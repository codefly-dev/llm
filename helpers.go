package llm

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"time"
)

// Inline helpers — ported from v1/pkg/shared to avoid external deps.

func fmtDur(d time.Duration) string {
	if d < time.Second {
		return fmt.Sprintf("%dms", d.Milliseconds())
	}
	if d < time.Minute {
		return fmt.Sprintf("%.1fs", d.Seconds())
	}
	return fmt.Sprintf("%.1fm", d.Minutes())
}

func shortHash(s string) string {
	h := sha256.Sum256([]byte(s))
	return hex.EncodeToString(h[:8])
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

func estTokens(s string) int {
	return (len(s) + 3) / 4
}
