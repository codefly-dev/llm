package ratelimit

import (
	"context"

	contract "github.com/codefly-dev/llm/contract"
	"golang.org/x/time/rate"
)

// Limiter caps API calls. It is assignable to contract.Limiter.
type Limiter = contract.Limiter

// NewInMemoryLimiter returns a process-local token-bucket limiter.
func NewInMemoryLimiter(requestsPerSec float64) Limiter {
	if requestsPerSec <= 0 {
		requestsPerSec = 10.0
	}
	return &inMemoryLimiter{limiter: rate.NewLimiter(rate.Limit(requestsPerSec), 1)}
}

type inMemoryLimiter struct {
	limiter *rate.Limiter
}

func (r *inMemoryLimiter) Allow(ctx context.Context) error {
	return r.limiter.Wait(ctx)
}

// NoopLimiter never blocks. Use in tests or cassette replay.
type NoopLimiter struct{}

func (NoopLimiter) Allow(context.Context) error { return nil }
