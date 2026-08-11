package llm

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/benbjohnson/clock"
)

// Provider is the interface for custom LLM backend providers.
// Register with WithProvider() instead of using a built-in provider name.
// The client only needs to implement Client; if it also implements
// UsageProvider, real token counts are used for cost tracking.
type Provider interface {
	Client
}

// ClientOption configures the middleware stack on a new client.
type ClientOption func(*clientBuilder)

type clientBuilder struct {
	retry    bool
	retryMax int
	retryBO  time.Duration

	recorder         bool
	recorderDir      string
	recorderDebugDir string
	recorderRunID    string
	recorderClock    clock.Clock
	runtimeClock     clock.Clock
	recordMode       RecordMode

	costTracker    *CostTracker
	rateLimit      float64
	budgetUSD      float64
	provider       Provider
	providerInfo   ModelInfo
	creds          APIKeyStore
	transport      *ProviderTransportProfile
	circuitBreaker *CircuitBreakerConfig
	timeouts       *TimeoutConfig
}

// WithRetry adds retry middleware with exponential backoff.
func WithRetry(maxRetries int, baseBackoff time.Duration) ClientOption {
	return func(b *clientBuilder) {
		b.retry = true
		b.retryMax = maxRetries
		b.retryBO = baseBackoff
	}
}

// WithRecorder adds the recorder middleware for deterministic test replays.
func WithRecorder(dir string, mode RecordMode) ClientOption {
	return func(b *clientBuilder) {
		b.recorder = true
		b.recorderDir = dir
		b.recordMode = mode
	}
}

// WithRecorderDebugDir selects an artifact directory for opt-in recorder key
// diagnostics. It must be separate from the cassette directory: replay must
// never mutate committed cassette state, even when MIND_RECORDER_DEBUG=1.
func WithRecorderDebugDir(dir string) ClientOption {
	return func(b *clientBuilder) {
		b.recorderDebugDir = dir
	}
}

// WithRecorderRunID selects a cassette sample namespace for WithRecorder.
// The default run id is "0", which preserves the legacy flat cassette layout.
// Non-zero ids write under <record-dir>/runs/<id>/ so repeated live samples of
// identical prompts can coexist and replay deterministically.
func WithRecorderRunID(runID string) ClientOption {
	return func(b *clientBuilder) {
		b.recorderRunID = runID
	}
}

// WithRecorderClock injects the timestamp source used in newly written
// cassette metadata. Replay does not consult it.
func WithRecorderClock(recorderClock clock.Clock) ClientOption {
	return func(b *clientBuilder) {
		b.recorderClock = recorderClock
	}
}

// WithRuntimeClock injects the time source for cost/timing middleware. It is
// distinct from cassette metadata configuration so every provider boundary can
// be deterministic even when recording is disabled.
func WithRuntimeClock(runtimeClock clock.Clock) ClientOption {
	return func(b *clientBuilder) {
		b.runtimeClock = runtimeClock
	}
}

// WithCostTracker adds cost tracking middleware.
func WithCostTracker(tracker *CostTracker) ClientOption {
	return func(b *clientBuilder) {
		b.costTracker = tracker
	}
}

// WithRateLimit overrides the default rate limit (requests/sec).
func WithRateLimit(rps float64) ClientOption {
	return func(b *clientBuilder) {
		b.rateLimit = rps
	}
}

// WithProvider uses a custom transport implementation instead of a built-in
// provider. Model metadata is mandatory so test/integration transports cannot
// silently fabricate pricing or context limits for an unknown model.
func WithProvider(p Provider, info ModelInfo) ClientOption {
	return func(b *clientBuilder) {
		b.provider = p
		b.providerInfo = info
	}
}

// WithBudget adds a cost ceiling (in USD). The middleware checks the running
// total before each LLM call and returns BudgetExceededError when the limit
// is reached. Requires a cost tracker (added automatically if not supplied).
func WithBudget(limitUSD float64) ClientOption {
	return func(b *clientBuilder) {
		b.budgetUSD = limitUSD
	}
}

// WithCircuitBreaker adds circuit breaker protection. After consecutive failures
// exceed the threshold, calls are rejected immediately until the reset timeout.
func WithCircuitBreaker(cfg CircuitBreakerConfig) ClientOption {
	return func(b *clientBuilder) {
		b.circuitBreaker = &cfg
	}
}

// WithTimeouts tunes the provider's per-attempt timeout + streaming stall
// watchdog (timeouts.go). Providers that support the knobs implement
// TimeoutConfigurable (Anthropic does); for providers that don't, the option
// is a no-op — they keep whatever bounding they implement natively. Without
// this option providers run with the package defaults, NOT unbounded.
func WithTimeouts(cfg TimeoutConfig) ClientOption {
	return func(b *clientBuilder) {
		b.timeouts = &cfg
	}
}

// WithCredentialStore resolves API keys via the credential store. Key lookup:
//   - "anthropic_api_key" for Anthropic
//   - "openai_api_key" for OpenAI
//   - the profile's credential kind when WithTransportProfile is supplied
func WithCredentialStore(cs APIKeyStore) ClientOption {
	return func(b *clientBuilder) {
		b.creds = cs
	}
}

// WithTransportProfile routes a built-in provider through a compatible
// gateway. Its credential is resolved through WithCredentialStore.
func WithTransportProfile(profile ProviderTransportProfile) ClientOption {
	staticHeaders := make(map[string]string, len(profile.StaticHeaders))
	for name, value := range profile.StaticHeaders {
		staticHeaders[name] = value
	}
	profile.StaticHeaders = staticHeaders
	return func(b *clientBuilder) {
		copy := profile
		b.transport = &copy
	}
}

// NewClient returns a Client for the given model identifier.
// Model can be "anthropic", "openai", or "provider/model" (e.g. "anthropic/claude-sonnet-5").
// Use WithProvider() to supply a custom backend.
// The client is wrapped so cassette replay can return before live-call
// concerns such as budget, cost accounting, retry, circuit breaking, and rate
// limiting run.
func NewClient(model Model, opts Options, clientOpts ...ClientOption) (Client, error) {
	providerName, modelSuffix, _ := strings.Cut(string(model), "/")
	providerName = strings.ToLower(providerName)

	// Collect options first so we can check for a custom provider.
	b := &clientBuilder{
		rateLimit: opts.RateLimit,
	}
	for _, o := range clientOpts {
		o(b)
	}

	var inner Client
	var modelName string
	var info ModelInfo
	var transport *providerTransport
	var key string
	transportIdentity := TransportIdentityDirectProvider

	// In replay-only mode the real provider is never called, so we can
	// skip API key validation and use a no-op inner client.
	replayOnly := b.recorder && b.recordMode == RecordReplayOnly

	if b.provider != nil {
		if b.transport != nil {
			return nil, fmt.Errorf("transport profiles cannot be combined with custom providers")
		}
		inner = b.provider
		transportIdentity = ClientTransportIdentity(inner)
		modelName = opts.Model
		if modelSuffix != "" {
			modelName = modelSuffix
		}
		if modelName == "" {
			modelName = "default"
		}
		info = b.providerInfo
		if info.Provider != providerName || info.Model != modelName {
			return nil, fmt.Errorf("custom provider metadata %q does not match requested model %q", info.Provider+"/"+info.Model, providerName+"/"+modelName)
		}
	} else {
		spec, ok := resolveProviderSpec(providerName)
		if !ok {
			return nil, fmt.Errorf("unknown LLM provider: %q (registered: %v; use WithProvider() for custom backends)", providerName, RegisteredProviders())
		}
		modelName = opts.Model
		if modelName == "" {
			modelName = spec.DefaultModel
		}
		if modelSuffix != "" {
			modelName = modelSuffix
		}
		var modelErr error
		info, modelErr = RequireModel(providerName, modelName)
		if modelErr != nil {
			return nil, modelErr
		}
		if b.transport != nil {
			transportIdentity = b.transport.Identity
			if replayOnly {
				if err := validateProviderTransportProfile(*b.transport); err != nil {
					return nil, err
				}
			} else {
				var transportErr error
				transport, transportErr = resolveProviderTransport(*b.transport, b.creds)
				if transportErr != nil {
					return nil, transportErr
				}
			}
		} else {
			key = resolveAPIKey(b.creds, explicitKeyFor(providerName, opts), spec.CredKind, spec.EnvKey)
			if key == "" && !replayOnly {
				return nil, fmt.Errorf("%s: API key not found. Run `mind setup` to configure, or ensure codefly injects %s into the credential store", providerName, spec.EnvKey)
			}
		}
		if !replayOnly {
			if transport != nil {
				inner = spec.transportFactory(modelName, *transport)
			} else {
				inner = spec.Factory(key, modelName)
			}
		}
	}
	if replayOnly {
		inner = StubError("replay-only: provider disabled; cassette miss is fatal")
	}

	// Apply explicit timeout knobs to providers that expose them. Providers
	// default to the package timeout config on construction, so absence of
	// this option means "defaults", never "unbounded".
	if b.timeouts != nil {
		if tc, ok := inner.(TimeoutConfigurable); ok {
			tc.SetTimeouts(*b.timeouts)
		}
	}

	// Build middleware chain in Chain order: first entry is outermost.
	var mws []Middleware

	fullModelName := providerName + "/" + modelName

	// OTEL is outermost so both replay and live calls have a span. The
	// recorder sets llm.recorder on that span.
	mws = append(mws, otelMiddleware(fullModelName, transportIdentity))

	// Recorder sits outside live-call middleware. A replay hit must not spend
	// budget, record usage, wait on rate limits, trip the circuit, or retry.
	if b.recorder {
		mws = append(mws, RecorderMiddlewareWithConfig(RecorderConfig{
			Dir:               b.recorderDir,
			DebugDir:          b.recorderDebugDir,
			Model:             fullModelName,
			TransportIdentity: transportIdentity,
			Mode:              b.recordMode,
			RunID:             b.recorderRunID,
			Clock:             b.recorderClock,
		}))
	}

	runtimeClock := b.runtimeClock
	if runtimeClock == nil && b.costTracker != nil {
		runtimeClock = b.costTracker.Clock()
	}
	if runtimeClock == nil {
		runtimeClock = clock.New()
	}
	if configurable, ok := inner.(RuntimeClockConfigurable); ok {
		configurable.SetRuntimeClock(runtimeClock)
	}
	inner = protectProviderErrors(inner, key, transport)
	var tracker *CostTracker
	if b.costTracker != nil {
		b.costTracker.SetModelInfo(info)
		tracker = b.costTracker
	} else {
		tracker = NewCostTracker(info, runtimeClock)
	}

	// Budget enforcement happens before cost tracking so a rejected call is not
	// counted as LLM usage.
	if b.budgetUSD > 0 {
		mws = append(mws, BudgetMiddleware(tracker, b.budgetUSD))
	}

	// Cost tracking records one logical LLM call. Retry is inside it, so
	// transient retry attempts do not create separate usage records.
	mws = append(mws, CostMiddleware(tracker))

	// Retry wraps the transport protections below it. Each attempt still goes
	// through circuit breaking and rate limiting.
	retryMax := b.retryMax
	retryBO := b.retryBO
	if !b.retry {
		retryMax = 3
		retryBO = 15 * time.Second
	}
	mws = append(mws, RetryMiddleware(retryMax, retryBO))

	// Circuit breaker rejects calls before consuming a rate-limit token when
	// the provider has been failing.
	if b.circuitBreaker != nil {
		mws = append(mws, CircuitBreakerMiddleware(*b.circuitBreaker, runtimeClock))
	} else {
		// Default circuit breaker: 5 consecutive failures, 30s reset.
		mws = append(mws, CircuitBreakerMiddleware(CircuitBreakerConfig{}, runtimeClock))
	}

	// Rate limit is closest to the provider, so it gates each live attempt.
	rps := b.rateLimit
	if rps <= 0 {
		rps = DefaultRate
	}
	mws = append(mws, RateLimitMiddlewareRPS(rps))

	return Chain(inner, mws...), nil
}

// protectProviderErrors installs the same credential-redaction boundary for
// direct providers and compatible gateway transports. Recorder, tracing, cost,
// and retry middleware all observe errors outside this boundary, so leaving the
// direct API key unbound here would let an echoed credential reach every
// durable diagnostic sink.
func protectProviderErrors(inner Client, directCredential string, transport *providerTransport) Client {
	credential := directCredential
	if transport != nil {
		credential = transport.credential
	}
	if credential == "" {
		return inner
	}
	return redactTransportErrors(inner, credential)
}

// resolveAPIKey returns a provider API key from, in order:
//  1. explicit Options field (for tests that inject a specific key)
//  2. APIKeyStore.Get (production: CodeflyStore; tests: MemoryStore)
//
// There is NO os.Getenv fallback. Mind's credentials policy is that every
// application secret flows through APIKeyStore — the only permitted
// env-var reads are inside CodeflyStore's SDK call, which reads the
// CODEFLY__* variables codefly CLI injects. Mixing direct env reads here
// would hide codefly-injection bugs (works locally, breaks remote/CI).
// See feedback_no_env_fallback_for_codefly and
// feedback_codefly_everything_infra in memory.
//
// The envVar parameter is preserved for use in error messages so callers
// get a hint of what secret they need codefly to inject.
func resolveAPIKey(creds APIKeyStore, explicit string, kind CredKind, _ string) string {
	if explicit != "" {
		return explicit
	}
	if creds != nil {
		if v, err := creds.APIKey(context.Background(), kind); err == nil {
			return v
		}
	}
	return ""
}
