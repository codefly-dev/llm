// Package retry holds the canonical exponential-backoff implementation used
// by pkg/llm and pkg/embedder retry middlewares. Both packages need the same
// backoff math; the previous duplication was a long-standing TODO (see the
// "Mirrors pkg/llm/retry.go" header that lived on pkg/embedder/retry.go).
//
// The retry loop itself stays in each consuming package — pkg/llm has Call /
// Stream / CallWithTools variants with different progress-tracking semantics,
// pkg/embedder has Embed / EmbedBatch. Trying to fold them into one generic
// "retry any operation" gets ugly because the early-exit conditions differ.
// The math is the part that is genuinely identical.
package retry

import (
	"context"
	"errors"
	"math/rand/v2"
	"time"

	"github.com/codefly-dev/llm/apierror"
)

// Sleep blocks for the duration returned by Backoff. Returns ctx.Err() if
// the context is cancelled during the wait.
func Sleep(ctx context.Context, attempt int, base time.Duration, lastErr error) error {
	timer := time.NewTimer(Backoff(attempt, base, lastErr))
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

// Backoff returns the delay before retrying after `attempt` failed tries.
// The base is doubled per attempt with up to 50% jitter. If lastErr carries
// an apierror.ProviderError with a RetryAfter that exceeds the computed
// delay, that RetryAfter is used instead — provider rate-limit hints win
// over our default schedule.
func Backoff(attempt int, base time.Duration, lastErr error) time.Duration {
	if attempt < 1 {
		attempt = 1
	}
	backoff := base * time.Duration(1<<(attempt-1))
	jitter := time.Duration(rand.Int64N(int64(backoff) / 2))
	delay := backoff + jitter
	var pe *apierror.ProviderError
	if lastErr != nil && errors.As(lastErr, &pe) && pe.RetryAfter > delay {
		delay = pe.RetryAfter
	}
	return delay
}
