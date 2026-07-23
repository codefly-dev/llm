package llm

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/benbjohnson/clock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ─── Helpers ────────────────────────────────────────────────

// alwaysFailClient returns an error on every call (all three methods).
type alwaysFailClient struct {
	err   error
	calls int
	mu    sync.Mutex
}

func (c *alwaysFailClient) Call(_ context.Context, _ string) (string, error) {
	c.mu.Lock()
	c.calls++
	c.mu.Unlock()
	return "", c.err
}

func (c *alwaysFailClient) Stream(_ context.Context, _ string, _ func(string) error) (string, error) {
	c.mu.Lock()
	c.calls++
	c.mu.Unlock()
	return "", c.err
}

func (c *alwaysFailClient) CallWithTools(_ context.Context, _ string, _ []Message, _ []ToolDef) (Message, error) {
	c.mu.Lock()
	c.calls++
	c.mu.Unlock()
	return Message{}, c.err
}

// failThenSucceedClient fails for the first N calls, then succeeds.
type failThenSucceedClient struct {
	failCount int
	failErr   error
	response  string
	calls     int
	mu        sync.Mutex
}

func (c *failThenSucceedClient) Call(_ context.Context, _ string) (string, error) {
	c.mu.Lock()
	c.calls++
	n := c.calls
	c.mu.Unlock()
	if n <= c.failCount {
		return "", c.failErr
	}
	return c.response, nil
}

func (c *failThenSucceedClient) Stream(_ context.Context, _ string, _ func(string) error) (string, error) {
	return c.Call(context.Background(), "")
}

func (c *failThenSucceedClient) CallWithTools(_ context.Context, _ string, _ []Message, _ []ToolDef) (Message, error) {
	c.mu.Lock()
	c.calls++
	n := c.calls
	c.mu.Unlock()
	if n <= c.failCount {
		return Message{}, c.failErr
	}
	return Message{Role: "assistant", Content: c.response}, nil
}

// cbState extracts the circuit state from a wrapped client.
func cbState(c Client) CircuitState {
	type stater interface{ State() CircuitState }
	if s, ok := c.(stater); ok {
		return s.State()
	}
	// Walk through Unwrap chain.
	if u, ok := c.(interface{ Unwrap() Client }); ok {
		return cbState(u.Unwrap())
	}
	return -1
}

// ─── Default Config ─────────────────────────────────────────

func TestCircuitBreakerConfig_Defaults(t *testing.T) {
	cfg := CircuitBreakerConfig{}
	cfg.defaults()

	assert.Equal(t, 5, cfg.FailureThreshold)
	assert.Equal(t, 30*time.Second, cfg.ResetTimeout)
	assert.Equal(t, 1, cfg.HalfOpenMax)
}

func TestCircuitBreakerConfig_DoesNotOverrideExplicit(t *testing.T) {
	cfg := CircuitBreakerConfig{
		FailureThreshold: 3,
		ResetTimeout:     10 * time.Second,
		HalfOpenMax:      2,
	}
	cfg.defaults()

	assert.Equal(t, 3, cfg.FailureThreshold)
	assert.Equal(t, 10*time.Second, cfg.ResetTimeout)
	assert.Equal(t, 2, cfg.HalfOpenMax)
}

// ─── CircuitState String ────────────────────────────────────

func TestCircuitState_String(t *testing.T) {
	assert.Equal(t, "closed", CircuitClosed.String())
	assert.Equal(t, "open", CircuitOpen.String())
	assert.Equal(t, "half-open", CircuitHalfOpen.String())
	assert.Equal(t, "unknown", CircuitState(99).String())
}

// ─── Normal operation (closed) ──────────────────────────────

func TestCircuitBreaker_ClosedPassesThrough(t *testing.T) {
	inner := newTransportProbe("ok")
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{FailureThreshold: 3}, clock.NewMock())(inner)
	ctx := context.Background()

	out, err := client.Call(ctx, "hello")
	require.NoError(t, err)
	assert.Equal(t, "ok", out)
	assert.Equal(t, CircuitClosed, cbState(client))
}

func TestCircuitBreaker_ClosedStream(t *testing.T) {
	inner := newTransportProbe("streamed")
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{FailureThreshold: 3}, clock.NewMock())(inner)
	ctx := context.Background()

	var chunks []string
	out, err := client.Stream(ctx, "hello", func(text string) error {
		chunks = append(chunks, text)
		return nil
	})
	require.NoError(t, err)
	assert.Equal(t, "streamed", out)
	assert.Equal(t, []string{"streamed"}, chunks)
}

// ─── Opens after threshold ──────────────────────────────────

func TestCircuitBreaker_OpensAfterThresholdFailures(t *testing.T) {
	threshold := 3
	inner := &alwaysFailClient{err: errors.New("service down")}
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: threshold,
		ResetTimeout:     time.Hour,
	}, clock.NewMock())(inner)
	ctx := context.Background()

	// First threshold calls all fail but circuit stays closed until threshold.
	for i := 0; i < threshold; i++ {
		_, err := client.Call(ctx, "test")
		assert.Error(t, err)
		assert.Equal(t, "service down", err.Error())
	}
	// Circuit should now be open.
	assert.Equal(t, CircuitOpen, cbState(client))

	// Next call should be rejected immediately with CircuitOpenError.
	_, err := client.Call(ctx, "test")
	require.Error(t, err)
	var coe *CircuitOpenError
	assert.True(t, errors.As(err, &coe))
	assert.Equal(t, threshold, coe.Failures)

	// The inner client should not have received the last call.
	inner.mu.Lock()
	assert.Equal(t, threshold, inner.calls)
	inner.mu.Unlock()
}

// ─── Rejects when open ──────────────────────────────────────

func TestCircuitBreaker_RejectsCallWhenOpen(t *testing.T) {
	inner := &alwaysFailClient{err: errors.New("fail")}
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 1,
		ResetTimeout:     time.Hour,
	}, clock.NewMock())(inner)
	ctx := context.Background()

	// One failure opens the circuit.
	_, _ = client.Call(ctx, "test")
	assert.Equal(t, CircuitOpen, cbState(client))

	// All methods should be rejected.
	_, err := client.Call(ctx, "test")
	var coe *CircuitOpenError
	assert.True(t, errors.As(err, &coe))

	_, err = client.Stream(ctx, "test", nil)
	assert.True(t, errors.As(err, &coe))

	tc, ok := client.(ToolClient)
	if ok {
		_, err = tc.CallWithTools(ctx, "", nil, nil)
		assert.True(t, errors.As(err, &coe))
	}
}

// ─── Transitions to half-open ───────────────────────────────

func TestCircuitBreaker_TransitionsToHalfOpenAfterTimeout(t *testing.T) {
	inner := &failThenSucceedClient{
		failCount: 1,
		failErr:   errors.New("fail"),
		response:  "recovered",
	}
	resetTimeout := 50 * time.Millisecond
	runtimeClock := clock.NewMock()
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 1,
		ResetTimeout:     resetTimeout,
		HalfOpenMax:      1,
	}, runtimeClock)(inner)
	ctx := context.Background()

	// Trigger open.
	_, err := client.Call(ctx, "test")
	require.Error(t, err)
	assert.Equal(t, CircuitOpen, cbState(client))

	runtimeClock.Add(resetTimeout)

	// Next call should be allowed (half-open probe) and succeed, closing the circuit.
	out, err := client.Call(ctx, "test")
	require.NoError(t, err)
	assert.Equal(t, "recovered", out)
	assert.Equal(t, CircuitClosed, cbState(client))
}

// ─── Closes on successful half-open probe ───────────────────

func TestCircuitBreaker_ClosesOnSuccessfulHalfOpenProbe(t *testing.T) {
	inner := &failThenSucceedClient{
		failCount: 1,
		failErr:   errors.New("transient"),
		response:  "ok",
	}

	resetTimeout := 30 * time.Millisecond
	runtimeClock := clock.NewMock()
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 1,
		ResetTimeout:     resetTimeout,
		HalfOpenMax:      1,
	}, runtimeClock)(inner)
	ctx := context.Background()

	// Open the circuit.
	_, _ = client.Call(ctx, "test")
	assert.Equal(t, CircuitOpen, cbState(client))

	runtimeClock.Add(resetTimeout)

	// Probe succeeds.
	out, err := client.Call(ctx, "test")
	require.NoError(t, err)
	assert.Equal(t, "ok", out)
	assert.Equal(t, CircuitClosed, cbState(client))

	// Subsequent calls should pass through normally.
	out, err = client.Call(ctx, "test")
	require.NoError(t, err)
	assert.Equal(t, "ok", out)
}

// ─── Re-opens on failed half-open probe ─────────────────────

func TestCircuitBreaker_ReopensOnFailedHalfOpenProbe(t *testing.T) {
	inner := &alwaysFailClient{err: errors.New("still broken")}
	resetTimeout := 30 * time.Millisecond
	runtimeClock := clock.NewMock()
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 1,
		ResetTimeout:     resetTimeout,
		HalfOpenMax:      1,
	}, runtimeClock)(inner)
	ctx := context.Background()

	// Open the circuit.
	_, _ = client.Call(ctx, "test")
	assert.Equal(t, CircuitOpen, cbState(client))

	runtimeClock.Add(resetTimeout)

	// Probe fails -> should re-open.
	_, err := client.Call(ctx, "test")
	require.Error(t, err)
	assert.Equal(t, "still broken", err.Error())
	assert.Equal(t, CircuitOpen, cbState(client))

	// Immediately, next call should be rejected.
	_, err = client.Call(ctx, "test")
	var coe *CircuitOpenError
	assert.True(t, errors.As(err, &coe))
}

// ─── Half-open rejects beyond HalfOpenMax ───────────────────

func TestCircuitBreaker_HalfOpenRejectsBeyondMax(t *testing.T) {
	failInner := &alwaysFailClient{err: errors.New("fail")}
	resetTimeout := 30 * time.Millisecond
	runtimeClock := clock.NewMock()
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 1,
		ResetTimeout:     resetTimeout,
		HalfOpenMax:      1,
	}, runtimeClock)(failInner)
	ctx := context.Background()

	// Open the circuit.
	_, _ = client.Call(ctx, "test")
	assert.Equal(t, CircuitOpen, cbState(client))

	runtimeClock.Add(resetTimeout)

	// First call in half-open is allowed (increments halfOpenCount to 1).
	// But it will fail since inner always fails, re-opening the circuit.
	_, err := client.Call(ctx, "test")
	require.Error(t, err)
	assert.Equal(t, "fail", err.Error())

	// After failed probe, circuit re-opens. Second call should get CircuitOpenError.
	_, err = client.Call(ctx, "test")
	var coe *CircuitOpenError
	assert.True(t, errors.As(err, &coe))
}

// ─── Resets failure count on success ────────────────────────

func TestCircuitBreaker_ResetsFailureCountOnSuccess(t *testing.T) {
	threshold := 3
	// Custom client: fails twice, succeeds once, then fails twice more, succeeds once.
	// Circuit should never open because success resets the counter.
	inner := &patternClient{
		pattern: []bool{true, true, false, true, true, false},
		err:     errors.New("transient"),
	}

	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: threshold,
		ResetTimeout:     time.Hour,
	}, clock.NewMock())(inner)
	ctx := context.Background()

	for i := 0; i < 6; i++ {
		_, _ = client.Call(ctx, "test")
	}

	// Circuit should still be closed because success resets consecutive failure count.
	assert.Equal(t, CircuitClosed, cbState(client))
}

// patternClient succeeds or fails based on a pattern (true=fail, false=succeed... wait, let's be clear).
type patternClient struct {
	pattern []bool // true = failure, false = success
	err     error
	idx     int
	mu      sync.Mutex
}

func (c *patternClient) Call(_ context.Context, _ string) (string, error) {
	c.mu.Lock()
	i := c.idx
	c.idx++
	c.mu.Unlock()
	if i < len(c.pattern) && c.pattern[i] {
		return "", c.err
	}
	return "ok", nil
}

func (c *patternClient) Stream(_ context.Context, _ string, _ func(string) error) (string, error) {
	return c.Call(context.Background(), "")
}

// ─── Stream follows circuit breaking ────────────────────────

func TestCircuitBreaker_StreamOpensCircuit(t *testing.T) {
	inner := &alwaysFailClient{err: errors.New("stream fail")}
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 2,
		ResetTimeout:     time.Hour,
	}, clock.NewMock())(inner)
	ctx := context.Background()

	// Two stream failures should open the circuit.
	_, _ = client.Stream(ctx, "test", nil)
	_, _ = client.Stream(ctx, "test", nil)
	assert.Equal(t, CircuitOpen, cbState(client))

	// Next stream call should be rejected.
	_, err := client.Stream(ctx, "test", nil)
	var coe *CircuitOpenError
	assert.True(t, errors.As(err, &coe))
}

func TestCircuitBreaker_StreamSuccessResetsCount(t *testing.T) {
	inner := &failThenSucceedClient{
		failCount: 1,
		failErr:   errors.New("fail"),
		response:  "ok",
	}
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 3,
		ResetTimeout:     time.Hour,
	}, clock.NewMock())(inner)
	ctx := context.Background()

	// One failure via Stream.
	_, _ = client.Stream(ctx, "test", nil)
	// One success via Stream — should reset counter.
	_, err := client.Stream(ctx, "test", nil)
	require.NoError(t, err)
	assert.Equal(t, CircuitClosed, cbState(client))
}

// ─── CallWithTools follows circuit breaking ─────────────────

func TestCircuitBreaker_CallWithToolsOpensCircuit(t *testing.T) {
	inner := &alwaysFailClient{err: errors.New("tool fail")}
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 2,
		ResetTimeout:     time.Hour,
	}, clock.NewMock())(inner)
	ctx := context.Background()

	tc, ok := client.(ToolClient)
	require.True(t, ok)

	_, _ = tc.CallWithTools(ctx, "", nil, nil)
	_, _ = tc.CallWithTools(ctx, "", nil, nil)
	assert.Equal(t, CircuitOpen, cbState(client))

	_, err := tc.CallWithTools(ctx, "", nil, nil)
	var coe *CircuitOpenError
	assert.True(t, errors.As(err, &coe))
}

func TestCircuitBreaker_CallWithToolsSuccessCloses(t *testing.T) {
	inner := &failThenSucceedClient{
		failCount: 1,
		failErr:   errors.New("fail"),
		response:  "ok",
	}
	resetTimeout := 30 * time.Millisecond
	runtimeClock := clock.NewMock()
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 1,
		ResetTimeout:     resetTimeout,
	}, runtimeClock)(inner)
	ctx := context.Background()

	tc, ok := client.(ToolClient)
	require.True(t, ok)

	// Open via CallWithTools.
	_, _ = tc.CallWithTools(ctx, "", nil, nil)
	assert.Equal(t, CircuitOpen, cbState(client))

	runtimeClock.Add(resetTimeout)
	msg, err := tc.CallWithTools(ctx, "", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, "ok", msg.Content)
	assert.Equal(t, CircuitClosed, cbState(client))
}

// ─── CallWithTools on plain Client inner ──────────────────

func TestCircuitBreaker_CallWithToolsNonToolClient(t *testing.T) {
	// A client without typed tool support must fail closed. Turning a tool
	// protocol into prompt text makes schema validation and replay ambiguous.
	inner := &plainClient{response: "ok"}
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 3,
	}, clock.NewMock())(inner)
	ctx := context.Background()

	tc, ok := client.(ToolClient)
	require.True(t, ok) // The middleware itself implements ToolClient.

	_, err := tc.CallWithTools(ctx, "", nil, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "does not implement native tool calling")
}

// plainClient implements only Client (no CallWithTools).
type plainClient struct {
	response string
}

func (c *plainClient) Call(_ context.Context, _ string) (string, error) {
	return c.response, nil
}

func (c *plainClient) Stream(_ context.Context, _ string, _ func(string) error) (string, error) {
	return c.response, nil
}

// ─── Mixed method failures accumulate ───────────────────────

func TestCircuitBreaker_MixedMethodFailuresAccumulate(t *testing.T) {
	inner := &alwaysFailClient{err: errors.New("fail")}
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 3,
		ResetTimeout:     time.Hour,
	}, clock.NewMock())(inner)
	ctx := context.Background()

	// One Call failure, one Stream failure, one CallWithTools failure = 3 total.
	_, _ = client.Call(ctx, "test")
	_, _ = client.Stream(ctx, "test", nil)
	tc, ok := client.(ToolClient)
	require.True(t, ok)
	_, _ = tc.CallWithTools(ctx, "", nil, nil)

	assert.Equal(t, CircuitOpen, cbState(client))
}

// ─── CircuitOpenError format ────────────────────────────────

func TestCircuitOpenError_Format(t *testing.T) {
	err := &CircuitOpenError{
		Failures:     5,
		OpenSince:    time.Date(2026, 4, 6, 12, 0, 0, 0, time.UTC),
		ResetTimeout: 30 * time.Second,
	}
	msg := err.Error()
	assert.Contains(t, msg, "circuit breaker open")
	assert.Contains(t, msg, "5 consecutive failures")
	assert.Contains(t, msg, "30s")
}

// ─── Concurrent safety ─────────────────────────────────────

func TestCircuitBreaker_ConcurrentSafety(t *testing.T) {
	inner := newTransportProbe("ok")
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 100,
		ResetTimeout:     time.Hour,
	}, clock.NewMock())(inner)
	ctx := context.Background()

	var wg sync.WaitGroup
	var errCount atomic.Int64

	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, err := client.Call(ctx, "test")
			if err != nil {
				errCount.Add(1)
			}
		}()
	}

	wg.Wait()
	assert.Equal(t, int64(0), errCount.Load())
	assert.Equal(t, CircuitClosed, cbState(client))
}

func TestCircuitBreaker_ConcurrentFailures(t *testing.T) {
	inner := &alwaysFailClient{err: errors.New("fail")}
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 5,
		ResetTimeout:     time.Hour,
	}, clock.NewMock())(inner)
	ctx := context.Background()

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = client.Call(ctx, "test")
		}()
	}
	wg.Wait()

	// Circuit must be open after enough concurrent failures.
	assert.Equal(t, CircuitOpen, cbState(client))
}

func TestCircuitBreaker_ConcurrentMixedSuccessAndFailure(t *testing.T) {
	// Ensure no panics or data races under concurrent mixed load.
	inner := &failThenSucceedClient{
		failCount: 10,
		failErr:   errors.New("fail"),
		response:  "ok",
	}
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 5,
		ResetTimeout:     10 * time.Millisecond,
	}, clock.NewMock())(inner)
	ctx := context.Background()

	var wg sync.WaitGroup
	for i := 0; i < 30; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = client.Call(ctx, "test")
		}()
	}
	wg.Wait()
	// No assertion on state; the point is no panics/races.
}

// ─── Unwrap ─────────────────────────────────────────────────

func TestCircuitBreaker_Unwrap(t *testing.T) {
	inner := newTransportProbe("ok")
	client := CircuitBreakerMiddleware(CircuitBreakerConfig{}, clock.NewMock())(inner)

	type unwrapper interface{ Unwrap() Client }
	u, ok := client.(unwrapper)
	require.True(t, ok)
	assert.Equal(t, inner, u.Unwrap())
}

// ─── Chain integration ──────────────────────────────────────

func TestCircuitBreaker_WorksWithChain(t *testing.T) {
	inner := &failThenSucceedClient{
		failCount: 2,
		failErr:   errors.New("fail"),
		response:  "ok",
	}
	runtimeClock := clock.NewMock()
	client := Chain(inner, CircuitBreakerMiddleware(CircuitBreakerConfig{
		FailureThreshold: 2,
		ResetTimeout:     30 * time.Millisecond,
	}, runtimeClock))
	ctx := context.Background()

	// Two failures open the circuit.
	_, _ = client.Call(ctx, "test")
	_, _ = client.Call(ctx, "test")

	// Rejected immediately.
	_, err := client.Call(ctx, "test")
	var coe *CircuitOpenError
	assert.True(t, errors.As(err, &coe))

	runtimeClock.Add(30 * time.Millisecond)
	out, err := client.Call(ctx, "test")
	require.NoError(t, err)
	assert.Equal(t, "ok", out)
}
