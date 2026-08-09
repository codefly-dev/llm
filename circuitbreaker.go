package llm

import (
	"context"
	"fmt"
	"sync"
	"time"

	"github.com/benbjohnson/clock"
)

// CircuitState represents the circuit breaker state.
type CircuitState int

const (
	CircuitClosed   CircuitState = iota // normal operation
	CircuitOpen                         // failing, reject calls
	CircuitHalfOpen                     // testing if service recovered
)

func (s CircuitState) String() string {
	switch s {
	case CircuitClosed:
		return "closed"
	case CircuitOpen:
		return "open"
	case CircuitHalfOpen:
		return "half-open"
	default:
		return "unknown"
	}
}

// CircuitBreakerConfig configures the circuit breaker.
type CircuitBreakerConfig struct {
	// FailureThreshold is the number of consecutive failures before opening. Default: 5.
	FailureThreshold int
	// ResetTimeout is how long the circuit stays open before half-open. Default: 30s.
	ResetTimeout time.Duration
	// HalfOpenMax is how many requests to allow in half-open state. Default: 1.
	HalfOpenMax int
}

func (c *CircuitBreakerConfig) defaults() {
	if c.FailureThreshold <= 0 {
		c.FailureThreshold = 5
	}
	if c.ResetTimeout <= 0 {
		c.ResetTimeout = 30 * time.Second
	}
	if c.HalfOpenMax <= 0 {
		c.HalfOpenMax = 1
	}
}

// CircuitBreakerMiddleware returns a middleware that implements circuit breaking.
// After FailureThreshold consecutive failures, the circuit opens and rejects calls
// immediately for ResetTimeout. Then it enters half-open state and allows one probe
// call. If that succeeds, the circuit closes; if it fails, it re-opens.
func CircuitBreakerMiddleware(cfg CircuitBreakerConfig, runtimeClock clock.Clock) Middleware {
	if runtimeClock == nil {
		panic("llm.CircuitBreakerMiddleware: runtime clock is required")
	}
	cfg.defaults()
	return func(next Client) Client {
		return &circuitBreakerMW{
			next:  next,
			cfg:   cfg,
			clock: runtimeClock,
			state: CircuitClosed,
		}
	}
}

type circuitBreakerMW struct {
	next  Client
	cfg   CircuitBreakerConfig
	clock clock.Clock

	mu              sync.Mutex
	state           CircuitState
	failures        int
	lastFailureTime time.Time
	halfOpenCount   int
}

// CircuitOpenError is returned when the circuit breaker is open.
type CircuitOpenError struct {
	Failures     int
	OpenSince    time.Time
	ResetTimeout time.Duration
}

func (e *CircuitOpenError) Error() string {
	return fmt.Sprintf("circuit breaker open: %d consecutive failures since %s (retry after %s)",
		e.Failures, e.OpenSince.Format(time.RFC3339), e.ResetTimeout)
}

func (m *circuitBreakerMW) checkState() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	switch m.state {
	case CircuitClosed:
		return nil
	case CircuitOpen:
		// Check if reset timeout has elapsed.
		if m.clock.Since(m.lastFailureTime) >= m.cfg.ResetTimeout {
			m.state = CircuitHalfOpen
			m.halfOpenCount = 0
			return nil
		}
		return &CircuitOpenError{
			Failures:     m.failures,
			OpenSince:    m.lastFailureTime,
			ResetTimeout: m.cfg.ResetTimeout,
		}
	case CircuitHalfOpen:
		if m.halfOpenCount >= m.cfg.HalfOpenMax {
			return &CircuitOpenError{
				Failures:     m.failures,
				OpenSince:    m.lastFailureTime,
				ResetTimeout: m.cfg.ResetTimeout,
			}
		}
		m.halfOpenCount++
		return nil
	}
	return nil
}

func (m *circuitBreakerMW) recordSuccess() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.failures = 0
	m.state = CircuitClosed
}

func (m *circuitBreakerMW) recordFailure() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.failures++
	m.lastFailureTime = m.clock.Now().UTC()
	if m.failures >= m.cfg.FailureThreshold {
		m.state = CircuitOpen
	}
}

func (m *circuitBreakerMW) recordResult(err error) {
	switch {
	case err == nil:
		m.recordSuccess()
	case IsReplayMiss(err):
		// A missing local certification artifact says nothing about provider
		// health. Preserve the existing breaker state and failure count.
	default:
		m.recordFailure()
	}
}

func (m *circuitBreakerMW) Call(ctx context.Context, prompt string) (string, error) {
	return m.CallWithOptions(ctx, prompt, RequestOptions{})
}

func (m *circuitBreakerMW) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error) {
	if err := m.checkState(); err != nil {
		return "", err
	}
	resp, err := CallWithOptions(ctx, m.next, prompt, opts)
	m.recordResult(err)
	return resp, err
}

func (m *circuitBreakerMW) Stream(ctx context.Context, prompt string, onChunk func(string) error) (string, error) {
	return m.StreamWithOptions(ctx, prompt, RequestOptions{}, onChunk)
}

func (m *circuitBreakerMW) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(string) error) (string, error) {
	if err := m.checkState(); err != nil {
		return "", err
	}
	resp, err := StreamWithOptions(ctx, m.next, prompt, opts, onChunk)
	m.recordResult(err)
	return resp, err
}

func (m *circuitBreakerMW) CallCached(ctx context.Context, system, user string) (string, error) {
	return m.CallCachedWithOptions(ctx, system, user, RequestOptions{})
}

func (m *circuitBreakerMW) CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (string, error) {
	if err := m.checkState(); err != nil {
		return "", err
	}
	resp, err := CallCachedWithOptions(ctx, m.next, system, user, opts)
	m.recordResult(err)
	return resp, err
}

func (m *circuitBreakerMW) CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (Message, error) {
	return m.CallWithToolsOptions(ctx, system, messages, tools, RequestOptions{})
}

func (m *circuitBreakerMW) CallWithToolsOptions(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error) {
	if err := m.checkState(); err != nil {
		return Message{}, err
	}
	resp, err := CallWithToolsOptions(ctx, m.next, system, messages, tools, opts)
	m.recordResult(err)
	return resp, err
}

func (m *circuitBreakerMW) Unwrap() Client { return m.next }

// State returns the current circuit breaker state (for observability).
func (m *circuitBreakerMW) State() CircuitState {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.state
}
