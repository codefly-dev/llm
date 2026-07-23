package llm

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/benbjohnson/clock"
)

// ─── Chain ───────────────────────────────────────────────────

func TestChain_Order(t *testing.T) {
	var order []string
	a := func(next Client) Client {
		return &orderMW{next: next, name: "A", order: &order}
	}
	b := func(next Client) Client {
		return &orderMW{next: next, name: "B", order: &order}
	}
	inner := newTransportProbe("ok")
	c := Chain(inner, a, b)

	_, err := c.Call(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	// A is outermost, B is inner. A should fire first.
	if len(order) != 2 || order[0] != "A" || order[1] != "B" {
		t.Errorf("expected [A B], got %v", order)
	}
}

type orderMW struct {
	next  Client
	name  string
	order *[]string
}

type advancingCostClient struct {
	inner   *transportProbeClient
	clock   *clock.Mock
	advance time.Duration
}

func (c *advancingCostClient) Call(ctx context.Context, prompt string) (string, error) {
	c.clock.Add(c.advance)
	return c.inner.Call(ctx, prompt)
}

func (c *advancingCostClient) Stream(ctx context.Context, prompt string, onChunk func(string) error) (string, error) {
	c.clock.Add(c.advance)
	return c.inner.Stream(ctx, prompt, onChunk)
}

func (m *orderMW) Call(ctx context.Context, prompt string) (string, error) {
	*m.order = append(*m.order, m.name)
	return m.next.Call(ctx, prompt)
}

func (m *orderMW) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (string, error) {
	*m.order = append(*m.order, m.name)
	return m.next.Stream(ctx, prompt, onChunk)
}

// ─── CostTracker ─────────────────────────────────────────────

func TestCostTracker_Accumulates(t *testing.T) {
	info := MustLookupModel("anthropic", AnthropicModelSonnet)
	tracker := NewCostTracker(info, clock.NewMock())
	client := Chain(newTransportProbe("response"), CostMiddleware(tracker))

	ctx := context.Background()
	_, _ = client.Call(ctx, "hello")
	_, _ = client.Call(ctx, "world")

	if tracker.CallCount() != 2 {
		t.Errorf("expected 2 calls, got %d", tracker.CallCount())
	}

	total := tracker.Total()
	if total.InputTokens <= 0 {
		t.Errorf("expected positive input tokens, got %d", total.InputTokens)
	}
	if total.OutputTokens <= 0 {
		t.Errorf("expected positive output tokens, got %d", total.OutputTokens)
	}
	if total.CostUSD <= 0 {
		t.Errorf("expected positive cost, got %f", total.CostUSD)
	}

	report := tracker.Report()
	if report == "" {
		t.Error("expected non-empty report")
	}
}

func TestCostMiddlewareUsesInjectedClockForUsageAndCallLog(t *testing.T) {
	runtimeClock := clock.NewMock()
	startedAt := time.Date(2026, time.July, 13, 20, 21, 22, 0, time.UTC)
	runtimeClock.Set(startedAt)
	tracker := NewCostTracker(ModelInfo{Provider: "test", Model: "timed"}, runtimeClock)
	callLog := NewCallLog()
	tracker.SetCallLog(callLog)
	inner := newTransportProbe("done")
	client := CostMiddleware(tracker)(&advancingCostClient{inner: inner, clock: runtimeClock, advance: 275 * time.Millisecond})

	if _, err := client.Call(context.Background(), "prompt"); err != nil {
		t.Fatal(err)
	}
	tracker.mu.Lock()
	usage := append([]Usage(nil), tracker.calls...)
	tracker.mu.Unlock()
	if len(usage) != 1 || usage[0].DurationMS != 275 {
		t.Fatalf("usage = %#v", usage)
	}
	records := callLog.Records()
	if len(records) != 1 || !records[0].Timestamp.Equal(startedAt.Add(275*time.Millisecond)) || records[0].DurationMS != 275 {
		t.Fatalf("call log = %#v", records)
	}
}

func TestCostMiddlewareFailsBeforeTransportWithoutRuntimeTracker(t *testing.T) {
	inner := newTransportProbe("must not run")
	client := CostMiddleware(nil)(inner)
	if _, err := client.Call(context.Background(), "prompt"); err == nil {
		t.Fatal("cost middleware without runtime tracker succeeded")
	}
	inner.mu.Lock()
	calls := inner.calls
	inner.mu.Unlock()
	if calls != 0 {
		t.Fatalf("transport called %d times", calls)
	}
}

func TestCostTracker_Hierarchical(t *testing.T) {
	info := MustLookupModel("anthropic", AnthropicModelSonnet)
	session := NewCostTrackerNamed("session", info, clock.NewMock())
	obj1 := session.Child("objective: add health endpoint")
	obj2 := session.Child("objective: add version flag")

	// Record calls on children.
	ctx := context.Background()
	c1 := Chain(newTransportProbe("response"), CostMiddleware(obj1))
	_, _ = c1.Call(ctx, "hello")
	_, _ = c1.Call(ctx, "world")

	c2 := Chain(newTransportProbe("done"), CostMiddleware(obj2))
	_, _ = c2.Call(ctx, "fix it")

	// Direct calls on session.
	cs := Chain(newTransportProbe("session-call"), CostMiddleware(session))
	_, _ = cs.Call(ctx, "general question")

	// Children should have their own counts.
	if obj1.CallCount() != 2 {
		t.Errorf("obj1 calls: got %d, want 2", obj1.CallCount())
	}
	if obj2.CallCount() != 1 {
		t.Errorf("obj2 calls: got %d, want 1", obj2.CallCount())
	}

	// Session total should include all children + its own direct call.
	if session.CallCount() != 4 {
		t.Errorf("session total calls: got %d, want 4", session.CallCount())
	}

	// Session direct should only have 1 call.
	direct := session.DirectTotal()
	if direct.InputTokens <= 0 {
		t.Error("expected positive direct input tokens")
	}

	// Total should be greater than direct (children contributed).
	total := session.Total()
	if total.CostUSD <= direct.CostUSD {
		t.Errorf("total ($%.4f) should exceed direct ($%.4f)", total.CostUSD, direct.CostUSD)
	}

	// Children list.
	children := session.Children()
	if len(children) != 2 {
		t.Fatalf("expected 2 children, got %d", len(children))
	}
	if children[0].Name != "objective: add health endpoint" {
		t.Errorf("child 0 name: %s", children[0].Name)
	}

	// Report should contain child names.
	report := session.Report()
	if !containsStr(report, "objective: add health endpoint") {
		t.Error("report missing child objective name")
	}
	if !containsStr(report, "objective: add version flag") {
		t.Error("report missing child objective name")
	}
}

func TestCostTrackerReportUsesItemizedUsageCosts(t *testing.T) {
	tracker := NewCostTracker(ModelInfo{Provider: "test", Model: "metered"}, clock.NewMock())
	err := tracker.Record(context.Background(), Usage{
		InputTokens:          100,
		OutputTokens:         20,
		WebSearchRequests:    1,
		InputCostUSD:         0.01,
		OutputCostUSD:        0.02,
		WebSearchCostUSD:     2,
		HostedToolCalls:      1,
		ToolCostUSD:          3,
		RequestCostUSD:       0.25,
		CostUSD:              5.90,
		ReasoningTokens:      5,
		ReasoningCostUSD:     0.03,
		WebFetchRequests:     1,
		WebFetchCostUSD:      0.5,
		CacheReadCostUSD:     0.04,
		CacheWriteCostUSD:    0.05,
		CacheReadInputTokens: 7,
	})
	if err != nil {
		t.Fatal(err)
	}

	report := tracker.Report()
	if !containsStr(report, "Tools:") {
		t.Fatalf("report should include tool costs, got:\n%s", report)
	}
	if !containsStr(report, "Request: $0.2500") {
		t.Fatalf("report should include request costs, got:\n%s", report)
	}
	if !containsStr(report, "Total:   $5.9000") {
		t.Fatalf("report should use recorded itemized total, got:\n%s", report)
	}
}

func TestCostMiddlewareDoesNotReuseProviderUsageAfterError(t *testing.T) {
	info := ModelInfo{Provider: "test", Model: "stale", InputPricePer1M: 1, OutputPricePer1M: 1}
	tracker := NewCostTracker(info, clock.NewMock())
	inner := &staleUsageErrorClient{
		usage: &Usage{Provider: "test", Model: "stale", InputTokens: 100, OutputTokens: 50},
	}
	client := Chain(inner, CostMiddleware(tracker))

	if _, err := client.Call(context.Background(), "success"); err != nil {
		t.Fatalf("first Call: %v", err)
	}
	if _, err := client.Call(context.Background(), "fail"); err == nil {
		t.Fatal("second Call should fail")
	}

	tracker.mu.Lock()
	calls := append([]Usage(nil), tracker.calls...)
	tracker.mu.Unlock()
	if len(calls) != 2 {
		t.Fatalf("recorded calls = %d, want 2", len(calls))
	}
	if !calls[1].Estimated {
		t.Fatalf("failed call should use estimated usage, got %+v", calls[1])
	}
	if calls[1].InputTokens == calls[0].InputTokens && calls[1].OutputTokens == calls[0].OutputTokens {
		t.Fatalf("failed call reused stale provider usage: first=%+v second=%+v", calls[0], calls[1])
	}
}

type staleUsageErrorClient struct {
	calls int
	usage *Usage
}

func (c *staleUsageErrorClient) Call(context.Context, string) (string, error) {
	c.calls++
	if c.calls == 1 {
		return "ok", nil
	}
	return "", errors.New("provider failed")
}

func (c *staleUsageErrorClient) Stream(ctx context.Context, prompt string, onChunk func(string) error) (string, error) {
	return c.Call(ctx, prompt)
}

func (c *staleUsageErrorClient) LastCallUsage() *Usage {
	if c.usage == nil {
		return nil
	}
	u := *c.usage
	return &u
}

func containsStr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// ─── Recorder ────────────────────────────────────────────────

func TestRecorder_RecordAndReplay(t *testing.T) {
	dir := t.TempDir()
	model := "test/recorder-model"

	// First call: records.
	inner := newTransportProbe("recorded-response")
	client := Chain(inner, RecorderMiddleware(dir, model, RecordAlways))

	ctx := context.Background()
	out, err := client.Call(ctx, "test-prompt")
	if err != nil {
		t.Fatal(err)
	}
	if out != "recorded-response" {
		t.Errorf("expected recorded-response, got %q", out)
	}

	// Verify file was written.
	hash := hashKey(model, "test-prompt")
	path := filepath.Join(dir, hash+".json")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatalf("recording file not created: %s", path)
	}

	// Second call: replays from disk (different inner to prove it doesn't call API).
	inner2 := newTransportProbe("should-not-see-this")
	client2 := Chain(inner2, RecorderMiddleware(dir, model, RecordReplayOnly))
	out2, err := client2.Call(ctx, "test-prompt")
	if err != nil {
		t.Fatal(err)
	}
	if out2 != "recorded-response" {
		t.Errorf("expected replay of recorded-response, got %q", out2)
	}
}

func TestRecorder_ReplayOnly_FailsWhenMissing(t *testing.T) {
	dir := t.TempDir()
	client := Chain(newTransportProbe("x"), RecorderMiddleware(dir, "test/model", RecordReplayOnly))

	_, err := client.Call(context.Background(), "no-such-recording")
	if err == nil {
		t.Fatal("expected error for missing recording in ReplayOnly mode")
	}
}

func TestRecorder_AlwaysMode_Overwrites(t *testing.T) {
	dir := t.TempDir()
	model := "test/always"

	// Record first response.
	client1 := Chain(newTransportProbe("first"), RecorderMiddleware(dir, model, RecordAlways))
	out1, _ := client1.Call(context.Background(), "prompt")
	if out1 != "first" {
		t.Errorf("got %q", out1)
	}

	// Record second response (overwrites).
	client2 := Chain(newTransportProbe("second"), RecorderMiddleware(dir, model, RecordAlways))
	out2, _ := client2.Call(context.Background(), "prompt")
	if out2 != "second" {
		t.Errorf("got %q", out2)
	}

	// Replay should return "second".
	client3 := Chain(newTransportProbe("ignored"), RecorderMiddleware(dir, model, RecordReplayOnly))
	out3, _ := client3.Call(context.Background(), "prompt")
	if out3 != "second" {
		t.Errorf("expected second, got %q", out3)
	}
}

// ─── Retry ───────────────────────────────────────────────────

func TestRetry_SucceedsOnFirstTry(t *testing.T) {
	client := Chain(newTransportProbe("ok"), RetryMiddleware(3, time.Millisecond))
	out, err := client.Call(context.Background(), "test")
	if err != nil {
		t.Fatal(err)
	}
	if out != "ok" {
		t.Errorf("got %q", out)
	}
}

func TestRetry_RetriesTransientError(t *testing.T) {
	fc := &failNClient{failCount: 2, failErr: &ProviderError{Provider: "test", Code: CodeServer, Status: 503}, response: "recovered"}
	client := Chain(fc, RetryMiddleware(3, time.Millisecond))

	out, err := client.Call(context.Background(), "test")
	if err != nil {
		t.Fatalf("expected success after retries, got: %v", err)
	}
	if out != "recovered" {
		t.Errorf("got %q", out)
	}
	if fc.calls != 3 {
		t.Errorf("expected 3 calls (2 failures + 1 success), got %d", fc.calls)
	}
}

func TestRetry_DoesNotRetry4xx(t *testing.T) {
	fc := &failNClient{failCount: 10, failErr: &ProviderError{Provider: "test", Code: CodeBadRequest, Status: 400}, response: "never"}
	client := Chain(fc, RetryMiddleware(3, time.Millisecond))

	_, err := client.Call(context.Background(), "test")
	if err == nil {
		t.Fatal("expected error for non-retryable 400")
	}
	if fc.calls != 1 {
		t.Errorf("expected 1 call (no retries for 4xx), got %d", fc.calls)
	}
}

func TestRetry_StreamRetriesOnlyBeforeProgress(t *testing.T) {
	t.Run("retries when no chunks were emitted", func(t *testing.T) {
		inner := &scriptedStreamClient{
			errs: []error{&ProviderError{Provider: "test", Code: CodeServer, Status: 503}, nil},
			outs: []string{"", "ok"},
		}
		client := Chain(inner, RetryMiddleware(3, time.Millisecond))
		var chunks []string
		out, err := client.Stream(context.Background(), "test", func(text string) error {
			chunks = append(chunks, text)
			return nil
		})
		if err != nil {
			t.Fatalf("Stream: %v", err)
		}
		if out != "ok" || len(chunks) != 1 || chunks[0] != "ok" {
			t.Fatalf("out=%q chunks=%v", out, chunks)
		}
		if inner.calls != 2 {
			t.Fatalf("calls=%d, want retry", inner.calls)
		}
	})

	t.Run("does not retry after partial progress", func(t *testing.T) {
		inner := &scriptedStreamClient{
			errs: []error{&ProviderError{Provider: "test", Code: CodeServer, Status: 503}, nil},
			outs: []string{"partial", "duplicate"},
		}
		client := Chain(inner, RetryMiddleware(3, time.Millisecond))
		var chunks []string
		out, err := client.Stream(context.Background(), "test", func(text string) error {
			chunks = append(chunks, text)
			return nil
		})
		if err == nil {
			t.Fatal("expected stream error")
		}
		if out != "partial" || len(chunks) != 1 || chunks[0] != "partial" {
			t.Fatalf("out=%q chunks=%v", out, chunks)
		}
		if inner.calls != 1 {
			t.Fatalf("calls=%d, want no retry after partial output", inner.calls)
		}
	})
}

// failNClient fails for the first N calls, then succeeds.
type failNClient struct {
	failCount int
	failErr   error
	response  string
	calls     int
}

func (f *failNClient) Call(ctx context.Context, prompt string) (string, error) {
	f.calls++
	if f.calls <= f.failCount {
		return "", f.failErr
	}
	return f.response, nil
}

func (f *failNClient) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (string, error) {
	return f.Call(ctx, prompt)
}

func (f *failNClient) CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (Message, error) {
	f.calls++
	if f.calls <= f.failCount {
		return Message{}, f.failErr
	}
	return Message{Role: "assistant", Content: f.response}, nil
}

type scriptedStreamClient struct {
	errs  []error
	outs  []string
	calls int
}

func (s *scriptedStreamClient) Call(ctx context.Context, prompt string) (string, error) {
	return "", errors.New("not used")
}

func (s *scriptedStreamClient) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (string, error) {
	i := s.calls
	s.calls++
	out := s.outs[i]
	if out != "" && onChunk != nil {
		if err := onChunk(out); err != nil {
			return out, err
		}
	}
	return out, s.errs[i]
}

func TestRetry_CallWithTools_RetriesRateLimit(t *testing.T) {
	fc := &failNClient{failCount: 1, failErr: &ProviderError{Provider: "test", Code: CodeRateLimited, Status: 429}, response: "ok"}
	client := Chain(fc, RetryMiddleware(3, time.Millisecond))
	tc, ok := AsToolClient(client)
	if !ok {
		t.Fatal("retry middleware must preserve native tool capability")
	}

	msg, err := tc.CallWithTools(context.Background(), "sys", []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("expected success after retry, got: %v", err)
	}
	if msg.Content != "ok" {
		t.Errorf("got %q", msg.Content)
	}
	if fc.calls != 2 {
		t.Errorf("expected 2 calls (1 failure + 1 success), got %d", fc.calls)
	}
}

func TestRateLimit_CallWithToolsGatesOuterWrapper(t *testing.T) {
	lim := &countingLimiter{}
	client := Chain(newTransportProbe("ok"), RateLimitMiddleware(lim))
	tc, ok := AsToolClient(client)
	if !ok {
		t.Fatal("rate-limit middleware must preserve native tool capability")
	}

	msg, err := tc.CallWithTools(context.Background(), "sys", []Message{{Role: "user", Content: "hi"}}, nil)
	if err != nil {
		t.Fatalf("CallWithTools: %v", err)
	}
	if msg.Content != "ok" {
		t.Fatalf("content=%q", msg.Content)
	}
	if lim.calls != 1 {
		t.Fatalf("limiter calls=%d, want 1", lim.calls)
	}
}

type countingLimiter struct {
	calls int
}

func (l *countingLimiter) Allow(context.Context) error {
	l.calls++
	return nil
}

// ─── ModelInfo ───────────────────────────────────────────────

func TestLookupModel_Known(t *testing.T) {
	info := MustLookupModel("anthropic", AnthropicModelSonnet)
	if info.ContextWindow != 1_000_000 {
		t.Errorf("context window: got %d", info.ContextWindow)
	}
	if info.InputPricePer1M != 3.0 {
		t.Errorf("input price: got %f", info.InputPricePer1M)
	}
}

func TestLookupModel_Unknown(t *testing.T) {
	if _, ok := LookupModel("unknown", "model-x"); ok {
		t.Fatal("unknown model must not receive invented metadata")
	}
}

func TestModelInfo_Cost(t *testing.T) {
	info := MustLookupModel("anthropic", AnthropicModelSonnet)
	cost := TotalCost(info, 1_000_000, 1_000_000)
	expected := 3.0 + 15.0
	if cost != expected {
		t.Errorf("cost for 1M/1M: got %f, want %f", cost, expected)
	}
}

// ─── CostTracker Timing ─────────────────────────────────────

func TestCostTracker_Timing_Sequential(t *testing.T) {
	info := MustLookupModel("anthropic", AnthropicModelSonnet)
	runtimeClock := clock.NewMock()
	parent := NewCostTracker(info, runtimeClock)

	// Two sequential children, exactly 50ms each.
	c1 := parent.Child("obj-1")
	runtimeClock.Add(50 * time.Millisecond)
	c1.End()

	c2 := parent.Child("obj-2")
	runtimeClock.Add(50 * time.Millisecond)
	c2.End()

	parent.End()

	wc := parent.WallClock()
	wt := parent.WorkTime()
	if wc != 100*time.Millisecond {
		t.Errorf("parent wall clock = %v, want 100ms", wc)
	}
	if wt != 100*time.Millisecond {
		t.Errorf("work time = %v, want 100ms", wt)
	}
	if p := parent.Parallelism(); p != 1 {
		t.Errorf("parallelism = %.2f, want 1", p)
	}
}

func TestCostTracker_Timing_Concurrent(t *testing.T) {
	info := MustLookupModel("anthropic", AnthropicModelSonnet)
	runtimeClock := clock.NewMock()
	parent := NewCostTracker(info, runtimeClock)

	// Two concurrent children, each ~100ms.
	c1 := parent.Child("obj-1")
	c2 := parent.Child("obj-2")

	runtimeClock.Add(100 * time.Millisecond)
	c1.End()
	c2.End()

	parent.End()

	wc := parent.WallClock()
	wt := parent.WorkTime()

	if wc != 100*time.Millisecond {
		t.Errorf("wall clock = %v, want 100ms", wc)
	}
	if wt != 200*time.Millisecond {
		t.Errorf("work time = %v, want 200ms", wt)
	}

	// Parallelism should be ~2.0.
	p := parent.Parallelism()
	if p != 2 {
		t.Errorf("parallelism = %.2f, want 2", p)
	}
}

func TestCostTracker_WallClock_NotEnded(t *testing.T) {
	info := MustLookupModel("anthropic", AnthropicModelSonnet)
	runtimeClock := clock.NewMock()
	tracker := NewCostTracker(info, runtimeClock)
	runtimeClock.Add(20 * time.Millisecond)

	wc := tracker.WallClock()
	if wc != 20*time.Millisecond {
		t.Errorf("wall clock = %v, want 20ms", wc)
	}
}
