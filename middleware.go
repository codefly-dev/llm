package llm

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/benbjohnson/clock"

	"log/slog"
)

// Chain applies middleware in order: the first middleware is outermost.
// Chain(inner, A, B, C) produces A(B(C(inner))).
func Chain(inner Client, mws ...Middleware) Client {
	for i := len(mws) - 1; i >= 0; i-- {
		inner = mws[i](inner)
	}
	return inner
}

// ─── Rate Limit Middleware ───────────────────────────────────

// RateLimitMiddleware returns a middleware that gates every LLM call
// through the supplied Limiter. The Limiter is the swap point between
// in-process (NewInMemoryLimiter) and shared-state backends (Redis,
// future). The middleware itself is backend-agnostic.
//
// Use RateLimitMiddlewareRPS for the common "I just want N requests per
// second in-process" case.
func RateLimitMiddleware(lim Limiter) Middleware {
	if lim == nil {
		lim = NoopLimiter{}
	}
	return func(next Client) Client {
		return &rateLimitMW{next: next, limiter: lim}
	}
}

// RateLimitMiddlewareRPS is RateLimitMiddleware with an in-memory
// token-bucket limiter sized to requestsPerSec. Equivalent to
// RateLimitMiddleware(NewInMemoryLimiter(requestsPerSec)).
//
// Most callers want this. Reach for RateLimitMiddleware(custom) when
// you need a Redis-backed limiter (multi-replica deployment) or any
// other shared-state backend.
func RateLimitMiddlewareRPS(requestsPerSec float64) Middleware {
	return RateLimitMiddleware(NewInMemoryLimiter(requestsPerSec))
}

type rateLimitMW struct {
	next    Client
	limiter Limiter
}

func (m *rateLimitMW) Call(ctx context.Context, prompt string) (string, error) {
	if err := m.limiter.Allow(ctx); err != nil {
		return "", err
	}
	return m.next.Call(ctx, prompt)
}

func (m *rateLimitMW) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error) {
	if err := m.limiter.Allow(ctx); err != nil {
		return "", err
	}
	return CallWithOptions(ctx, m.next, prompt, opts)
}

func (m *rateLimitMW) CallCached(ctx context.Context, system, user string) (string, error) {
	if err := m.limiter.Allow(ctx); err != nil {
		return "", err
	}
	if cc, ok := m.next.(CacheableClient); ok {
		return cc.CallCached(ctx, system, user)
	}
	return m.next.Call(ctx, system+"\n\n"+user)
}

func (m *rateLimitMW) CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (string, error) {
	if err := m.limiter.Allow(ctx); err != nil {
		return "", err
	}
	return CallCachedWithOptions(ctx, m.next, system, user, opts)
}

func (m *rateLimitMW) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (string, error) {
	if err := m.limiter.Allow(ctx); err != nil {
		return "", err
	}
	return m.next.Stream(ctx, prompt, onChunk)
}

func (m *rateLimitMW) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(text string) error) (string, error) {
	if err := m.limiter.Allow(ctx); err != nil {
		return "", err
	}
	return StreamWithOptions(ctx, m.next, prompt, opts, onChunk)
}

func (m *rateLimitMW) CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (Message, error) {
	return m.CallWithToolsOptions(ctx, system, messages, tools, RequestOptions{})
}

func (m *rateLimitMW) CallWithToolsOptions(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error) {
	if err := m.limiter.Allow(ctx); err != nil {
		return Message{}, err
	}
	return CallWithToolsOptions(ctx, m.next, system, messages, tools, opts)
}

func (m *rateLimitMW) Unwrap() Client { return m.next }

// ─── Cost Tracking Middleware ────────────────────────────────

// CallLog accumulates CallRecords for debug inspection. Thread-safe.
type CallLog struct {
	mu      sync.Mutex
	records []CallRecord
	subs    []chan<- CallRecord
}

func NewCallLog() *CallLog { return &CallLog{} }

// Append adds a record and notifies subscribers.
func (l *CallLog) Append(r CallRecord) {
	l.mu.Lock()
	l.records = append(l.records, r)
	subs := append([]chan<- CallRecord(nil), l.subs...)
	l.mu.Unlock()
	for _, ch := range subs {
		select {
		case ch <- r:
		default:
		}
	}
}

// Records returns a snapshot of all recorded calls.
func (l *CallLog) Records() []CallRecord {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]CallRecord(nil), l.records...)
}

// Subscribe returns a channel that receives new CallRecords in real time.
// Keep the returned channel to pass to Unsubscribe.
func (l *CallLog) Subscribe() chan CallRecord {
	ch := make(chan CallRecord, 64)
	l.mu.Lock()
	l.subs = append(l.subs, ch)
	l.mu.Unlock()
	return ch
}

// Unsubscribe removes a subscription channel.
func (l *CallLog) Unsubscribe(ch chan CallRecord) {
	l.mu.Lock()
	defer l.mu.Unlock()
	for i, s := range l.subs {
		if s == (chan<- CallRecord)(ch) {
			l.subs = append(l.subs[:i], l.subs[i+1:]...)
			close(s)
			return
		}
	}
}

// Persister is an optional sink that receives every Usage record for durable storage.
// Implemented by store.Store to flush LLM calls to Postgres.
type Persister interface {
	Persist(ctx context.Context, u Usage) error
}

// CostTracker accumulates token usage, cost, and timing across multiple calls.
// Trackers form a tree: session -> objective -> sub-objective.
// Each node tracks its own direct calls. Total() rolls up children.
//
// Timing: each node records Started/Ended. For concurrent objectives,
// WallClock < WorkTime (parallelism helped). Parallelism() gives the ratio.
type CostTracker struct {
	mu        sync.Mutex
	Name      string // "session", "objective: add health endpoint", etc.
	calls     []Usage
	info      ModelInfo
	children  []*CostTracker
	parent    *CostTracker
	persister Persister // optional: writes to Postgres when set
	callLog   *CallLog  // optional: debug UI call log
	clock     clock.Clock

	Started time.Time `json:"started,omitempty"`
	Ended   time.Time `json:"ended,omitempty"`
}

// NewCostTracker creates a cost tracker for the given model.
func NewCostTracker(info ModelInfo, runtimeClock clock.Clock) *CostTracker {
	return NewCostTrackerNamed("session", info, runtimeClock)
}

// NewCostTrackerNamed creates a named cost tracker (for objectives, sub-objectives).
func NewCostTrackerNamed(name string, info ModelInfo, runtimeClock clock.Clock) *CostTracker {
	if runtimeClock == nil {
		panic("llm.NewCostTrackerNamed: runtime clock is required")
	}
	return &CostTracker{info: info, Name: name, clock: runtimeClock, Started: runtimeClock.Now().UTC()}
}

// SetPersister attaches a durable storage sink (e.g. Postgres store).
func (t *CostTracker) SetPersister(p Persister) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.persister = p
}

// SetCallLog attaches a debug call log. Propagated to children.
func (t *CostTracker) SetCallLog(l *CallLog) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.callLog = l
}

// Clock returns the tracker's runtime clock so builders can reuse the same
// (test-injected) clock when constructing sibling components.
func (t *CostTracker) Clock() clock.Clock { return t.clock }

// SetModelInfo updates the pricing/context model info used to cost calls.
func (t *CostTracker) SetModelInfo(info ModelInfo) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.info = info
}

// GetCallLog returns the attached call log (may be nil).
func (t *CostTracker) GetCallLog() *CallLog {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.callLog
}

// Child creates a child tracker whose costs roll up to this parent.
// Children inherit the parent's persister and call log. Started is set automatically.
func (t *CostTracker) Child(name string) *CostTracker {
	t.mu.Lock()
	defer t.mu.Unlock()
	child := &CostTracker{
		info: t.info, Name: name, parent: t,
		persister: t.persister, callLog: t.callLog,
		clock: t.clock, Started: t.clock.Now().UTC(),
	}
	t.children = append(t.children, child)
	return child
}

// Record adds a usage entry to this tracker and flushes to the persister if set.
func (t *CostTracker) Record(ctx context.Context, u Usage) error {
	if u.Provider == "" {
		u.Provider = t.info.Provider
	}
	if u.Model == "" {
		u.Model = t.info.Model
	}
	if u.CostUSD == 0 && hasBillableUsage(u) {
		fillUsageCost(t.info, &u)
	}

	t.mu.Lock()
	t.calls = append(t.calls, u)
	p := t.persister
	t.mu.Unlock()

	if p != nil {
		_ = p.Persist(ctx, u)
	}
	return nil
}

func hasBillableUsage(u Usage) bool {
	return u.Requests > 0 ||
		u.InputTokens > 0 ||
		u.OutputTokens > 0 ||
		u.ReasoningTokens > 0 ||
		u.CacheReadInputTokens > 0 ||
		u.CacheWriteInputTokens > 0 ||
		u.HostedToolCalls > 0 ||
		u.WebSearchRequests > 0 ||
		u.WebFetchRequests > 0
}

// DirectTotal returns usage from this tracker's own calls only (no children).
func (t *CostTracker) DirectTotal() Usage {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.directTotalLocked()
}

// Total returns aggregate usage including all children (recursive).
func (t *CostTracker) Total() Usage {
	direct := t.DirectTotal()
	t.mu.Lock()
	children := append([]*CostTracker(nil), t.children...)
	t.mu.Unlock()

	for _, child := range children {
		ct := child.Total()
		direct.Requests += ct.Requests
		direct.InputTokens += ct.InputTokens
		direct.OutputTokens += ct.OutputTokens
		direct.ReasoningTokens += ct.ReasoningTokens
		direct.CachedTokens += ct.CachedTokens
		direct.CachedInputTokens += ct.CachedInputTokens
		direct.CachedOutputTokens += ct.CachedOutputTokens
		direct.CacheReadInputTokens += ct.CacheReadInputTokens
		direct.CacheWriteInputTokens += ct.CacheWriteInputTokens
		direct.ToolCalls += ct.ToolCalls
		direct.HostedToolCalls += ct.HostedToolCalls
		direct.WebSearchRequests += ct.WebSearchRequests
		direct.WebFetchRequests += ct.WebFetchRequests
		direct.InputCostUSD += ct.InputCostUSD
		direct.OutputCostUSD += ct.OutputCostUSD
		direct.ReasoningCostUSD += ct.ReasoningCostUSD
		direct.CacheReadCostUSD += ct.CacheReadCostUSD
		direct.CacheWriteCostUSD += ct.CacheWriteCostUSD
		direct.RequestCostUSD += ct.RequestCostUSD
		direct.ToolCostUSD += ct.ToolCostUSD
		direct.WebSearchCostUSD += ct.WebSearchCostUSD
		direct.WebFetchCostUSD += ct.WebFetchCostUSD
		direct.CostUSD += ct.CostUSD
	}
	return direct
}

// CallCount returns the total number of LLM calls (including children).
func (t *CostTracker) CallCount() int {
	t.mu.Lock()
	n := len(t.calls)
	children := append([]*CostTracker(nil), t.children...)
	t.mu.Unlock()
	for _, child := range children {
		n += child.CallCount()
	}
	return n
}

// Children returns the child trackers.
func (t *CostTracker) Children() []*CostTracker {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]*CostTracker(nil), t.children...)
}

// Start marks the beginning of this tracker's scope.
func (t *CostTracker) Start() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.Started = t.clock.Now().UTC()
}

// End marks the completion of this tracker's scope.
func (t *CostTracker) End() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.Ended = t.clock.Now().UTC()
}

// WallClock returns the elapsed time for this tracker's scope.
// If not yet ended, returns time since start.
func (t *CostTracker) WallClock() time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.Started.IsZero() {
		return 0
	}
	if !t.Ended.IsZero() {
		return t.Ended.Sub(t.Started)
	}
	return t.clock.Since(t.Started)
}

// WorkTime returns total work effort across all leaves.
// For leaf trackers (no children): same as WallClock.
// For parent trackers: sum of children's WorkTime.
// This captures total compute effort regardless of parallelism.
func (t *CostTracker) WorkTime() time.Duration {
	t.mu.Lock()
	children := append([]*CostTracker(nil), t.children...)
	t.mu.Unlock()

	if len(children) == 0 {
		return t.WallClock()
	}
	var total time.Duration
	for _, child := range children {
		total += child.WorkTime()
	}
	return total
}

// Parallelism returns the ratio of WorkTime to WallClock.
// >1.0 means concurrent objectives saved time.
// Returns 1.0 if no meaningful data.
func (t *CostTracker) Parallelism() float64 {
	wc := t.WallClock()
	if wc == 0 {
		return 1.0
	}
	return float64(t.WorkTime()) / float64(wc)
}

// Report returns a human-readable cost summary with breakdown.
func (t *CostTracker) Report() string {
	return t.reportIndent(0)
}

func (t *CostTracker) reportIndent(depth int) string {
	indent := strings.Repeat("  ", depth)
	total := t.Total()
	direct := t.DirectTotal()

	var b strings.Builder
	fmt.Fprintf(&b, "%s%s\n", indent, t.Name)
	fmt.Fprintf(&b, "%s  Model:   %s/%s\n", indent, t.info.Provider, t.info.Model)
	fmt.Fprintf(&b, "%s  Calls:   %d\n", indent, t.CallCount())
	promptCost := total.InputCostUSD + total.CacheReadCostUSD + total.CacheWriteCostUSD
	completionCost := total.OutputCostUSD + total.ReasoningCostUSD
	toolCost := total.ToolCostUSD + total.WebSearchCostUSD + total.WebFetchCostUSD
	fmt.Fprintf(&b, "%s  Input:   %d tokens ($%.4f)\n", indent, total.InputTokens, promptCost)
	fmt.Fprintf(&b, "%s  Output:  %d tokens ($%.4f)\n", indent, total.OutputTokens, completionCost)
	if total.ReasoningTokens > 0 {
		fmt.Fprintf(&b, "%s  Reason:  %d tokens ($%.4f)\n", indent, total.ReasoningTokens, total.ReasoningCostUSD)
	}
	if total.WebSearchRequests > 0 || total.WebFetchRequests > 0 || total.HostedToolCalls > 0 {
		fmt.Fprintf(&b, "%s  Tools:   %d hosted / %d search / %d fetch ($%.4f)\n",
			indent, total.HostedToolCalls, total.WebSearchRequests, total.WebFetchRequests, toolCost)
	}
	if total.RequestCostUSD > 0 {
		fmt.Fprintf(&b, "%s  Request: $%.4f\n", indent, total.RequestCostUSD)
	}
	fmt.Fprintf(&b, "%s  Total:   $%.4f\n", indent, total.CostUSD)

	wc := t.WallClock()
	wt := t.WorkTime()
	if wc > 0 {
		fmt.Fprintf(&b, "%s  Time:    %s wall", indent, fmtDur(wc))
		if wt > wc+time.Second {
			fmt.Fprintf(&b, " / %s work (%.1fx parallelism)", fmtDur(wt), t.Parallelism())
		}
		b.WriteString("\n")
	}

	children := t.Children()
	if len(children) > 0 && direct.CostUSD > 0 {
		fmt.Fprintf(&b, "%s  (direct: $%.4f, children: $%.4f)\n", indent, direct.CostUSD, total.CostUSD-direct.CostUSD)
	}

	for _, child := range children {
		b.WriteString(child.reportIndent(depth + 1))
	}
	return b.String()
}

// CostSummary is a flattened cost record for display in breakdown tables.
type CostSummary struct {
	Name         string
	Calls        int
	InputTokens  int
	OutputTokens int
	CachedTokens int
	CostUSD      float64
	WallClock    time.Duration
	WorkTime     time.Duration
}

// ByAgent returns a flat list of cost summaries, one per child tracker.
// The first entry is this tracker's direct costs. Children follow recursively.
func (t *CostTracker) ByAgent() []CostSummary {
	var out []CostSummary

	t.mu.Lock()
	direct := t.directTotalLocked()
	directCalls := len(t.calls)
	children := append([]*CostTracker(nil), t.children...)
	name := t.Name
	t.mu.Unlock()

	out = append(out, CostSummary{
		Name:         name,
		Calls:        directCalls,
		InputTokens:  direct.InputTokens,
		OutputTokens: direct.OutputTokens,
		CachedTokens: direct.CachedTokens,
		CostUSD:      direct.CostUSD,
		WallClock:    t.WallClock(),
		WorkTime:     t.WorkTime(),
	})

	for _, child := range children {
		out = append(out, child.byAgentRecursive()...)
	}
	return out
}

func (t *CostTracker) byAgentRecursive() []CostSummary {
	total := t.Total()
	return []CostSummary{{
		Name:         t.Name,
		Calls:        t.CallCount(),
		InputTokens:  total.InputTokens,
		OutputTokens: total.OutputTokens,
		CachedTokens: total.CachedTokens,
		CostUSD:      total.CostUSD,
		WallClock:    t.WallClock(),
		WorkTime:     t.WorkTime(),
	}}
}

func (t *CostTracker) directTotalLocked() Usage {
	var total Usage
	total.Provider = t.info.Provider
	total.Model = t.info.Model
	for _, u := range t.calls {
		total.Requests += u.Requests
		total.InputTokens += u.InputTokens
		total.OutputTokens += u.OutputTokens
		total.ReasoningTokens += u.ReasoningTokens
		total.CachedTokens += u.CachedTokens
		total.CachedInputTokens += u.CachedInputTokens
		total.CachedOutputTokens += u.CachedOutputTokens
		total.CacheReadInputTokens += u.CacheReadInputTokens
		total.CacheWriteInputTokens += u.CacheWriteInputTokens
		total.ToolCalls += u.ToolCalls
		total.HostedToolCalls += u.HostedToolCalls
		total.WebSearchRequests += u.WebSearchRequests
		total.WebFetchRequests += u.WebFetchRequests
		total.InputCostUSD += u.InputCostUSD
		total.OutputCostUSD += u.OutputCostUSD
		total.ReasoningCostUSD += u.ReasoningCostUSD
		total.CacheReadCostUSD += u.CacheReadCostUSD
		total.CacheWriteCostUSD += u.CacheWriteCostUSD
		total.RequestCostUSD += u.RequestCostUSD
		total.ToolCostUSD += u.ToolCostUSD
		total.WebSearchCostUSD += u.WebSearchCostUSD
		total.WebFetchCostUSD += u.WebFetchCostUSD
		total.CostUSD += u.CostUSD
	}
	return total
}

// CacheHitRate returns the fraction of input tokens served from cache (0.0-1.0).
func (t *CostTracker) CacheHitRate() float64 {
	total := t.Total()
	if total.InputTokens <= 0 {
		return 0
	}
	return float64(total.CachedTokens) / float64(total.InputTokens)
}

// --- Context-scoped tracker ---

type ctxKeyTracker struct{}

// WithTracker stores a CostTracker in the context. Used by agent.RunChild
// so that LLM calls inside child agents record to the child's tracker.
func WithTracker(ctx context.Context, t *CostTracker) context.Context {
	return context.WithValue(ctx, ctxKeyTracker{}, t)
}

// TrackerFromContext returns the CostTracker stored in ctx, or nil.
func TrackerFromContext(ctx context.Context) *CostTracker {
	if t, ok := ctx.Value(ctxKeyTracker{}).(*CostTracker); ok {
		return t
	}
	return nil
}

// CostMiddleware returns a middleware that tracks token usage and cost.
func CostMiddleware(tracker *CostTracker) Middleware {
	return func(next Client) Client {
		return &costMW{next: next, tracker: tracker}
	}
}

type costMW struct {
	next     Client
	tracker  *CostTracker
	lastMu   sync.Mutex
	lastCall *Usage
}

// activeTracker returns the context-scoped tracker if present, otherwise the
// explicitly configured tracker. Missing deterministic runtime services fail
// before the provider boundary is invoked.
func (m *costMW) activeTracker(ctx context.Context) *CostTracker {
	if t := TrackerFromContext(ctx); t != nil {
		if t.clock != nil {
			return t
		}
		return nil
	}
	if m.tracker != nil && m.tracker.clock != nil {
		return m.tracker
	}
	return nil
}

func (m *costMW) Call(ctx context.Context, prompt string) (string, error) {
	t := m.activeTracker(ctx)
	if t == nil {
		return "", fmt.Errorf("llm cost middleware: runtime tracker with clock is required")
	}
	start := t.clock.Now()
	out, err := m.next.Call(ctx, prompt)
	dur := t.clock.Since(start)
	u := m.buildUsage(t.info, prompt, out, err == nil)
	u.DurationMS = dur.Milliseconds()
	m.setLastUsage(u)
	_ = t.Record(ctx, u)
	m.recordCallLogTo(t, u, prompt, out, 0)
	m.logCall(u, dur, err)
	return out, err
}

func (m *costMW) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error) {
	t := m.activeTracker(ctx)
	if t == nil {
		return "", fmt.Errorf("llm cost middleware: runtime tracker with clock is required")
	}
	start := t.clock.Now()
	out, err := CallWithOptions(ctx, m.next, prompt, opts)
	dur := t.clock.Since(start)
	u := m.buildUsage(t.info, prompt, out, err == nil)
	if opts.PriceLevel != "" && u.PriceLevel == "" {
		u.PriceLevel = opts.PriceLevel
		fillUsageCost(t.info, &u)
	}
	u.DurationMS = dur.Milliseconds()
	m.setLastUsage(u)
	_ = t.Record(ctx, u)
	m.recordCallLogTo(t, u, prompt, out, 0)
	m.logCall(u, dur, err)
	return out, err
}

func (m *costMW) CallCached(ctx context.Context, system, user string) (string, error) {
	t := m.activeTracker(ctx)
	if t == nil {
		return "", fmt.Errorf("llm cost middleware: runtime tracker with clock is required")
	}
	start := t.clock.Now()
	var out string
	var err error
	if cc, ok := m.next.(CacheableClient); ok {
		out, err = cc.CallCached(ctx, system, user)
	} else {
		out, err = m.next.Call(ctx, system+"\n\n"+user)
	}
	dur := t.clock.Since(start)
	prompt := system + "\n\n" + user
	u := m.buildUsage(t.info, prompt, out, err == nil)
	u.DurationMS = dur.Milliseconds()
	m.setLastUsage(u)
	_ = t.Record(ctx, u)
	m.recordCallLogTo(t, u, prompt, out, 0)
	m.logCall(u, dur, err)
	return out, err
}

func (m *costMW) CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (string, error) {
	t := m.activeTracker(ctx)
	if t == nil {
		return "", fmt.Errorf("llm cost middleware: runtime tracker with clock is required")
	}
	start := t.clock.Now()
	out, err := CallCachedWithOptions(ctx, m.next, system, user, opts)
	dur := t.clock.Since(start)
	prompt := system + "\n\n" + user
	u := m.buildUsage(t.info, prompt, out, err == nil)
	if opts.PriceLevel != "" && u.PriceLevel == "" {
		u.PriceLevel = opts.PriceLevel
		fillUsageCost(t.info, &u)
	}
	u.DurationMS = dur.Milliseconds()
	m.setLastUsage(u)
	_ = t.Record(ctx, u)
	m.recordCallLogTo(t, u, prompt, out, 0)
	m.logCall(u, dur, err)
	return out, err
}

func (m *costMW) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (string, error) {
	t := m.activeTracker(ctx)
	if t == nil {
		return "", fmt.Errorf("llm cost middleware: runtime tracker with clock is required")
	}
	start := t.clock.Now()
	full, err := m.next.Stream(ctx, prompt, onChunk)
	dur := t.clock.Since(start)
	u := m.buildUsage(t.info, prompt, full, err == nil)
	u.DurationMS = dur.Milliseconds()
	m.setLastUsage(u)
	_ = t.Record(ctx, u)
	m.recordCallLogTo(t, u, prompt, full, 0)
	m.logCall(u, dur, err)
	return full, err
}

func (m *costMW) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(text string) error) (string, error) {
	t := m.activeTracker(ctx)
	if t == nil {
		return "", fmt.Errorf("llm cost middleware: runtime tracker with clock is required")
	}
	start := t.clock.Now()
	full, err := StreamWithOptions(ctx, m.next, prompt, opts, onChunk)
	dur := t.clock.Since(start)
	u := m.buildUsage(t.info, prompt, full, err == nil)
	if opts.PriceLevel != "" && u.PriceLevel == "" {
		u.PriceLevel = opts.PriceLevel
		fillUsageCost(t.info, &u)
	}
	u.DurationMS = dur.Milliseconds()
	m.setLastUsage(u)
	_ = t.Record(ctx, u)
	m.recordCallLogTo(t, u, prompt, full, 0)
	m.logCall(u, dur, err)
	return full, err
}

func (m *costMW) CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (Message, error) {
	return m.CallWithToolsOptions(ctx, system, messages, tools, RequestOptions{})
}

func (m *costMW) CallWithToolsOptions(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error) {
	t := m.activeTracker(ctx)
	if t == nil {
		return Message{}, fmt.Errorf("llm cost middleware: runtime tracker with clock is required")
	}
	start := t.clock.Now()
	resp, err := CallWithToolsOptions(ctx, m.next, system, messages, tools, opts)
	dur := t.clock.Since(start)

	prompt := system
	for _, msg := range messages {
		prompt += "\n" + msg.Content
	}
	u := m.buildUsage(t.info, prompt, resp.Content, err == nil)
	if u.ToolCalls == 0 && len(resp.ToolCalls) > 0 {
		u.ToolCalls = len(resp.ToolCalls)
		fillUsageCost(t.info, &u)
	}
	if opts.PriceLevel != "" && u.PriceLevel == "" {
		u.PriceLevel = opts.PriceLevel
		fillUsageCost(t.info, &u)
	}
	u.DurationMS = dur.Milliseconds()
	m.setLastUsage(u)
	_ = t.Record(ctx, u)
	m.recordCallLogTo(t, u, prompt, resp.Content, len(resp.ToolCalls))
	m.logCall(u, dur, err)
	return resp, err
}

func (m *costMW) Unwrap() Client { return m.next }

func (m *costMW) LastCallUsage() *Usage {
	m.lastMu.Lock()
	defer m.lastMu.Unlock()
	if m.lastCall == nil {
		return nil
	}
	u := *m.lastCall
	return &u
}

func (m *costMW) setLastUsage(u Usage) {
	m.lastMu.Lock()
	defer m.lastMu.Unlock()
	copy := u
	m.lastCall = &copy
}

func (m *costMW) recordCallLogTo(t *CostTracker, u Usage, prompt, response string, toolCalls int) {
	log := t.GetCallLog()
	if log == nil {
		return
	}
	log.Append(CallRecord{
		Timestamp:       t.clock.Now().UTC(),
		Model:           u.Model,
		Agent:           t.Name,
		PromptHash:      shortHash(u.Model + "\x00" + prompt),
		PromptPreview:   trunc(prompt, 200),
		ResponsePreview: trunc(response, 200),
		FullPrompt:      prompt,
		FullResponse:    response,
		InputTokens:     u.InputTokens,
		OutputTokens:    u.OutputTokens,
		CachedTokens:    u.CachedTokens,
		CostUSD:         u.CostUSD,
		DurationMS:      u.DurationMS,
		CacheHit:        u.CachedTokens > 0,
		ToolCallCount:   toolCalls,
	})
}

func (m *costMW) logCall(u Usage, dur time.Duration, err error) {
	errStr := ""
	if err != nil {
		errStr = err.Error()
	}
	slog.Debug("llm call",
		"provider", u.Provider,
		"model", u.Model,
		"duration_ms", dur.Milliseconds(),
		"input_tokens", u.InputTokens,
		"output_tokens", u.OutputTokens,
		"cached_tokens", u.CachedTokens,
		"cost_usd", fmt.Sprintf("%.6f", u.CostUSD),
		"error", errStr,
	)
}

// buildUsage constructs a Usage record. Prefers real API token counts from any
// provider implementing UsageProvider; falls back to heuristic estimation.
func (m *costMW) buildUsage(info ModelInfo, prompt, output string, useProviderUsage bool) Usage {
	u := Usage{
		Provider: info.Provider,
		Model:    info.Model,
		Requests: 1,
	}

	if useProviderUsage {
		provider := findUsageProvider(m.next)
		if provider != nil {
			if lu := provider.LastCallUsage(); lu != nil {
				u.Requests = firstPositive(lu.Requests, 1)
				u.InputTokens = lu.InputTokens
				u.OutputTokens = lu.OutputTokens
				u.CachedTokens = lu.CachedTokens
				u.CachedInputTokens = lu.CachedInputTokens
				u.CachedOutputTokens = lu.CachedOutputTokens
				u.CacheReadInputTokens = lu.CacheReadInputTokens
				u.CacheWriteInputTokens = lu.CacheWriteInputTokens
				u.ReasoningTokens = lu.ReasoningTokens
				u.ReasoningOutputExtra = lu.ReasoningOutputExtra
				u.PriceLevel = lu.PriceLevel
				u.ToolCalls = lu.ToolCalls
				u.HostedToolCalls = lu.HostedToolCalls
				u.WebSearchRequests = lu.WebSearchRequests
				u.WebFetchRequests = lu.WebFetchRequests

				fillUsageCost(info, &u)
				return u
			}
		}
	}

	u.InputTokens = estTokens(prompt)
	u.OutputTokens = estTokens(output)
	fillUsageCost(info, &u)
	u.Estimated = true
	return u
}

func fillUsageCost(info ModelInfo, u *Usage) {
	cost := PriceUsage(info, *u)
	u.InputCostUSD = cost.InputUSD
	u.OutputCostUSD = cost.OutputUSD
	u.ReasoningCostUSD = cost.ReasoningUSD
	u.CacheReadCostUSD = cost.CacheReadUSD
	u.CacheWriteCostUSD = cost.CacheWriteUSD
	u.RequestCostUSD = cost.RequestUSD
	u.ToolCostUSD = cost.ToolUSD
	u.WebSearchCostUSD = cost.WebSearchUSD
	u.WebFetchCostUSD = cost.WebFetchUSD
	u.CostUSD = cost.TotalUSD
}

func firstPositive(values ...int) int {
	for _, value := range values {
		if value > 0 {
			return value
		}
	}
	return 0
}

// findUsageProvider walks through middleware wrappers to find any client
// implementing UsageProvider (Anthropic, OpenAI, or future providers).
func findUsageProvider(c Client) UsageProvider {
	if up, ok := c.(UsageProvider); ok {
		return up
	}
	if u, ok := c.(interface{ Unwrap() Client }); ok {
		return findUsageProvider(u.Unwrap())
	}
	return nil
}
