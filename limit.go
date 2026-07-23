package llm

// The rate-limiting primitive is transport-neutral and shared across every
// API client in Mind. The contract lives in pkg/spec; the in-process
// implementation lives in pkg/ratelimit.

import (
	"github.com/codefly-dev/llm/ratelimit"
)

// Re-exports from the shared rate-limit implementation.
type Limiter = ratelimit.Limiter

// NewInMemoryLimiter returns a process-local token-bucket limiter sized to
// requestsPerSec.
func NewInMemoryLimiter(requestsPerSec float64) Limiter {
	return ratelimit.NewInMemoryLimiter(requestsPerSec)
}

// NoopLimiter never blocks. Use in tests or when rate limiting should be
// disabled.
type NoopLimiter = ratelimit.NoopLimiter

// DefaultRate is the default requests-per-second for LLM calls. Tuned
// conservatively for a free-tier account; production callers should size
// their Limiter to their actual provider tier.
const DefaultRate = 10.0
