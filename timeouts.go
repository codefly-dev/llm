package llm

// ARCHITECTURE: per-attempt timeout + streaming stall watchdog config.
//
// Every live provider attempt must be BOUNDED. A real trace showed one
// Anthropic streaming call running 39 minutes past a 12-minute task deadline:
// the SDK's HTTP client has no request timeout, and a stream that keeps the
// TCP connection alive but trickles (or stops emitting) events never
// terminates on its own. Two independent brakes fix that class of failure:
//
//   - PerAttempt: a hard ceiling on one request attempt (non-streaming call
//     or an entire stream). Derived as a child context of the caller's ctx,
//     so the caller's deadline still wins when it is sooner.
//   - StreamStall: an inter-event watchdog for streams. If no stream event
//     arrives within the window, the attempt is aborted with a typed,
//     RETRYABLE stall error (ErrStreamStalled wrapped in a ProviderError
//     with CodeTimeout) so the retry middleware can re-attempt cleanly.
//
// The knobs live on the provider client (see Anthropic.SetTimeouts and the
// WithTimeouts ClientOption in provider.go) — NOT on env vars — following the
// existing pkg/llm config pattern (CircuitBreakerConfig, retry options).
//
// Cassette replay is unaffected: the recorder middleware short-circuits
// replayed calls before they ever reach a provider client, so neither brake
// can fire on instant replayed responses.

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/benbjohnson/clock"
	"github.com/codefly-dev/core/wool"

	"github.com/codefly-dev/llm/apierror"
)

const (
	// DefaultPerAttemptTimeout bounds one live request attempt (a full
	// non-streaming call, or an entire stream from request to last event).
	// Generous enough for large tool-call responses at normal token rates;
	// small enough that a wedged attempt is recycled well inside any
	// realistic task budget. Tune per client via TimeoutConfig.
	DefaultPerAttemptTimeout = 240 * time.Second

	// DefaultStreamStallTimeout is the maximum gap between two consecutive
	// stream EVENTS before the attempt is declared stalled. Anthropic emits
	// content/thinking deltas continuously while generating; the SDK swallows
	// keepalive `ping` events inside Next(), so this must stay comfortably
	// above the provider's inter-delta gap but far below "wedged forever".
	DefaultStreamStallTimeout = 60 * time.Second
)

// ErrStreamStalled is the typed sentinel for a stream aborted by the
// inter-event watchdog. It is always delivered wrapped inside an
// apierror.ProviderError with CodeTimeout, so it is BOTH matchable
// (errors.Is(err, llm.ErrStreamStalled)) and retryable
// (llm.IsRetryable(err) == true).
var ErrStreamStalled = errors.New("llm: stream stalled")

// TimeoutConfig holds the per-attempt discipline knobs for a provider client.
//
// Zero values select the defaults above; a NEGATIVE value disables that
// brake explicitly (e.g. StreamStall: -1 for a provider whose streams
// legitimately go silent).
type TimeoutConfig struct {
	// PerAttempt bounds one request attempt end-to-end.
	// 0 = DefaultPerAttemptTimeout, <0 = disabled.
	PerAttempt time.Duration
	// StreamStall bounds the gap between consecutive stream events.
	// 0 = DefaultStreamStallTimeout, <0 = disabled. Streaming only.
	StreamStall time.Duration
}

// withDefaults normalizes the config: zero fields get the package defaults,
// negative fields collapse to 0 meaning "disabled" at the call sites.
func (c TimeoutConfig) withDefaults() TimeoutConfig {
	out := c
	switch {
	case out.PerAttempt == 0:
		out.PerAttempt = DefaultPerAttemptTimeout
	case out.PerAttempt < 0:
		out.PerAttempt = 0
	}
	switch {
	case out.StreamStall == 0:
		out.StreamStall = DefaultStreamStallTimeout
	case out.StreamStall < 0:
		out.StreamStall = 0
	}
	return out
}

// attemptContext derives the bounded per-attempt context from the caller's
// ctx. The caller's own deadline still applies (child context); the returned
// cancel MUST be deferred. With PerAttempt disabled it returns ctx unchanged.
func (c TimeoutConfig) attemptContext(ctx context.Context) (context.Context, context.CancelFunc) {
	if c.PerAttempt <= 0 {
		return ctx, func() {}
	}
	return context.WithTimeout(ctx, c.PerAttempt)
}

// TimeoutConfigurable is implemented by provider clients whose per-attempt
// timeout / stall watchdog knobs can be tuned after construction. NewClient's
// WithTimeouts option uses it so the knobs flow through the standard
// ClientOption pattern without changing every provider factory signature.
type TimeoutConfigurable interface {
	SetTimeouts(TimeoutConfig)
}

type RuntimeClockConfigurable interface {
	SetRuntimeClock(clock.Clock)
}

// streamGuard bounds ONE streaming attempt with the two brakes from this
// file: the per-attempt budget (a child context of the caller's ctx) and the
// inter-event stall watchdog. It exists so every streaming provider shares one
// implementation of the retryability-critical error attribution below instead
// of hand-syncing a copy per provider.
//
// Lifecycle: build with newStreamGuard, hand ctx() to the SDK's streaming
// constructor, call reset() after every received event to re-arm the watchdog,
// and defer close(). Route the terminal stream error through attribute(), and
// guard clean completion with callerExpired().
type streamGuard struct {
	caller        context.Context
	attemptCtx    context.Context
	streamCtx     context.Context
	provider      string
	perAttempt    time.Duration
	stall         time.Duration
	watchdog      *time.Timer
	cancelAttempt context.CancelFunc
	cancelStream  context.CancelCauseFunc
}

// newStreamGuard normalizes cfg, emits the near-deadline observability warning,
// and derives the per-attempt + stall-watchdog contexts. The caller MUST defer
// close() to release both contexts and stop the watchdog.
func newStreamGuard(ctx context.Context, cfg TimeoutConfig, provider, model string, runtimeClock clock.Clock) *streamGuard {
	cfg = cfg.withDefaults()
	warnNearDeadline(ctx, provider, model, cfg, runtimeClock)

	attemptCtx, cancelAttempt := cfg.attemptContext(ctx)
	streamCtx, cancelStream := context.WithCancelCause(attemptCtx)
	g := &streamGuard{
		caller:        ctx,
		attemptCtx:    attemptCtx,
		streamCtx:     streamCtx,
		provider:      provider,
		perAttempt:    cfg.PerAttempt,
		stall:         cfg.StreamStall,
		cancelAttempt: cancelAttempt,
		cancelStream:  cancelStream,
	}
	if cfg.StreamStall > 0 {
		stall := cfg.StreamStall
		g.watchdog = time.AfterFunc(stall, func() { cancelStream(stallError(provider, stall)) })
	}
	return g
}

// ctx is the context to hand to the SDK's streaming constructor.
func (g *streamGuard) ctx() context.Context { return g.streamCtx }

// reset re-arms the stall watchdog; call it once per received stream event.
// No-op when the watchdog is disabled.
func (g *streamGuard) reset() {
	if g.watchdog != nil {
		g.watchdog.Reset(g.stall)
	}
}

// close stops the watchdog and releases the derived contexts. Deferring it
// preserves the original stop→cancelStream→cancelAttempt ordering.
func (g *streamGuard) close() {
	if g.watchdog != nil {
		g.watchdog.Stop()
	}
	g.cancelStream(nil)
	g.cancelAttempt()
}

// attribute maps a failed streaming attempt's terminal error to the error to
// surface, in priority order: watchdog stall (typed, retryable, returned as-is
// so the ProviderError survives) → dead CALLER ctx (its deadline/cancel must
// surface, not an opaque transport error) → our per-attempt budget (retryable
// timeout, DeadlineExceeded wrapped explicitly because the transport error is
// not guaranteed to carry it) → the raw transport error. classify is the
// provider's own classifier.
func (g *streamGuard) attribute(streamErr error, classify func(error) error) error {
	if cause := context.Cause(g.streamCtx); cause != nil && errors.Is(cause, ErrStreamStalled) {
		return cause
	}
	if ctxErr := g.caller.Err(); ctxErr != nil {
		return classify(fmt.Errorf("%w (stream aborted: %v)", ctxErr, streamErr))
	}
	if g.attemptCtx.Err() != nil {
		return classify(fmt.Errorf("%s: per-attempt timeout %s exceeded: %w (stream error: %v)", g.provider, g.perAttempt, context.DeadlineExceeded, streamErr))
	}
	return classify(streamErr)
}

// callerExpired returns classify(ctx.Err()) when the caller ctx is already dead
// even though the stream ended without error, else nil. A stream must never
// report clean success on a dead caller ctx — callers rely on the error to stop
// looping.
func (g *streamGuard) callerExpired(classify func(error) error) error {
	if ctxErr := g.caller.Err(); ctxErr != nil {
		return classify(ctxErr)
	}
	return nil
}

// stallError builds the typed, retryable error the watchdog aborts with.
// ProviderError with CodeTimeout → IsRetryable(err) is true; the Wrapped
// chain carries ErrStreamStalled for precise matching.
func stallError(provider string, window time.Duration) error {
	return &apierror.ProviderError{
		Provider: provider,
		Code:     apierror.CodeTimeout,
		Message:  fmt.Sprintf("stream stalled: no events for %s", window),
		Wrapped:  fmt.Errorf("%w: no stream events within %s", ErrStreamStalled, window),
	}
}

// deadlineHeadroom reports how much of the caller's deadline remains and
// whether the attempt is starting DANGEROUSLY late — with less headroom than
// 20% of the per-attempt budget. Pure function so it is unit-testable; the
// providers log the warning via warnNearDeadline.
func deadlineHeadroom(ctx context.Context, cfg TimeoutConfig, now time.Time) (remaining time.Duration, near bool) {
	deadline, ok := ctx.Deadline()
	if !ok {
		return 0, false
	}
	remaining = deadline.Sub(now)
	budget := cfg.PerAttempt
	if budget <= 0 {
		budget = DefaultPerAttemptTimeout
	}
	return remaining, remaining < budget/5
}

// warnNearDeadline emits the deadline-observability warning: an LLM attempt
// is starting with < 20% of a per-attempt budget of ctx headroom left, so it
// will almost certainly be cut off by the caller's deadline. Logged (wool)
// rather than returned — the attempt still proceeds; the log field is the
// trace-side breadcrumb for post-hoc "why did this call die at deadline".
func warnNearDeadline(ctx context.Context, provider, model string, cfg TimeoutConfig, runtimeClock clock.Clock) {
	if runtimeClock == nil {
		return
	}
	remaining, near := deadlineHeadroom(ctx, cfg, runtimeClock.Now())
	if !near {
		return
	}
	wool.Get(ctx).In("llm.timeouts").Warn("llm attempt starting near caller deadline",
		wool.Field("provider", provider),
		wool.Field("model", model),
		wool.Field("remaining_ms", remaining.Milliseconds()),
		wool.Field("per_attempt_ms", cfg.PerAttempt.Milliseconds()),
	)
}
