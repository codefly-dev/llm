package llm

// Regression tests for the per-attempt timeout + stream stall watchdog
// (timeouts.go) and for runStream's context discipline.
//
// These use a fake slow/stalling SSE server (httptest) — the sanctioned
// error-injection exception in CLAUDE.md: a provider that stalls mid-stream
// or never answers is exactly the failure real infrastructure cannot produce
// on demand. The Anthropic SDK client is pointed at the fake via BaseURL, so
// the FULL production path (SDK request construction, SSE decode, Accumulate,
// classify) runs — nothing is mocked inside Mind.

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/benbjohnson/clock"
)

// newAnthropicAgainst builds a real Anthropic client whose SDK talks to the
// fake server. SDK retries are disabled so timing assertions stay tight.
func newAnthropicAgainst(url string, cfg TimeoutConfig) *Anthropic {
	sdk := anthropic.NewClient(
		option.WithAPIKey("test-key"),
		option.WithBaseURL(url),
		option.WithMaxRetries(0),
	)
	return &Anthropic{sdk: sdk, model: "claude-test", clock: clock.NewMock(), timeouts: cfg}
}

func sseHeaders(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

func writeSSE(w http.ResponseWriter, event, data string) {
	fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}
}

const sseMessageStart = `{"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","model":"claude-test","content":[],"stop_reason":null,"usage":{"input_tokens":3,"output_tokens":1}}}`

func sseTextDelta(text string) string {
	return fmt.Sprintf(`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"%s"}}`, text)
}

// TestAnthropicStreamStallWatchdogAborts: a stream that goes silent must be
// aborted by the inter-event watchdog with the typed, RETRYABLE stall error —
// not hang until some outer deadline (the 39-minute-call regression).
func TestAnthropicStreamStallWatchdogAborts(t *testing.T) {
	// done releases the stalled handler at test end: httptest's Close waits
	// for in-flight handlers, and the server does not always observe a
	// client-side abort, so blocking on r.Context() alone can deadlock Close.
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sseHeaders(w)
		writeSSE(w, "message_start", sseMessageStart)
		writeSSE(w, "content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
		writeSSE(w, "content_block_delta", sseTextDelta("hello"))
		// Stall forever: no more events until the client gives up.
		select {
		case <-r.Context().Done():
		case <-done:
		}
	}))
	defer srv.Close()
	defer close(done) // LIFO: release the handler, THEN srv.Close()

	c := newAnthropicAgainst(srv.URL, TimeoutConfig{
		PerAttempt:  30 * time.Second,
		StreamStall: 150 * time.Millisecond,
	})

	start := time.Now()
	out, err := c.Stream(context.Background(), "stall please", nil)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatalf("expected stall error, got clean success with %q", out)
	}
	if !errors.Is(err, ErrStreamStalled) {
		t.Fatalf("err = %v, want ErrStreamStalled in chain", err)
	}
	if !IsRetryable(err) {
		t.Fatalf("stall error must be retryable, got %v", err)
	}
	if out != "hello" {
		t.Errorf("partial text = %q, want %q (deltas before the stall are kept)", out, "hello")
	}
	if elapsed > 5*time.Second {
		t.Fatalf("stall abort took %s, want well under 5s", elapsed)
	}
}

// TestAnthropicStreamCancelledCtxKillsInFlight: cancelling the CALLER's ctx
// must kill an in-flight stream promptly and surface the ctx error — the
// regression where a call outlived its deadline by 35+ minutes.
func TestAnthropicStreamCancelledCtxKillsInFlight(t *testing.T) {
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sseHeaders(w)
		writeSSE(w, "message_start", sseMessageStart)
		select {
		case <-r.Context().Done():
		case <-done:
		}
	}))
	defer srv.Close()
	defer close(done)

	// Both brakes far away: the caller's cancel must be what kills the call.
	c := newAnthropicAgainst(srv.URL, TimeoutConfig{
		PerAttempt:  30 * time.Second,
		StreamStall: 30 * time.Second,
	})

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(100 * time.Millisecond)
		cancel()
	}()

	start := time.Now()
	_, err := c.Stream(ctx, "cancel me", nil)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected error from cancelled ctx, got clean success")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want context.Canceled in chain", err)
	}
	if elapsed > 2*time.Second {
		t.Fatalf("cancel took %s to kill the stream, want prompt (<2s)", elapsed)
	}
}

// TestAnthropicStreamPerAttemptTimeout: a stream that keeps trickling events
// (so the stall watchdog never fires) must still be cut off by the hard
// per-attempt budget with a retryable timeout.
func TestAnthropicStreamPerAttemptTimeout(t *testing.T) {
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sseHeaders(w)
		writeSSE(w, "message_start", sseMessageStart)
		writeSSE(w, "content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
		for {
			select {
			case <-r.Context().Done():
				return
			case <-done:
				return
			case <-time.After(50 * time.Millisecond):
			}
			writeSSE(w, "content_block_delta", sseTextDelta("x"))
		}
	}))
	defer srv.Close()
	defer close(done)

	c := newAnthropicAgainst(srv.URL, TimeoutConfig{
		PerAttempt:  300 * time.Millisecond,
		StreamStall: -1, // disabled: isolate the attempt budget
	})

	start := time.Now()
	_, err := c.Stream(context.Background(), "trickle forever", nil)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected per-attempt timeout error, got clean success")
	}
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("err = %v, want CodeTimeout ProviderError", err)
	}
	if !IsRetryable(err) {
		t.Fatalf("per-attempt timeout must be retryable, got %v", err)
	}
	if elapsed > 3*time.Second {
		t.Fatalf("attempt budget took %s to fire, want <3s", elapsed)
	}
}

// TestAnthropicPerAttemptTimeoutNonStreaming: the bounded attempt applies to
// the non-streaming tool-call path too — a server that never answers cannot
// wedge CallWithTools past the attempt budget.
func TestAnthropicPerAttemptTimeoutNonStreaming(t *testing.T) {
	done := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Never answer. The done channel (not r.Context()) is the release
		// valve: with an unread POST body and no response written, the
		// server may not observe the client abort, and httptest's Close
		// would wait on this handler forever.
		<-done
	}))
	defer srv.Close()
	defer close(done)

	c := newAnthropicAgainst(srv.URL, TimeoutConfig{PerAttempt: 150 * time.Millisecond})

	start := time.Now()
	_, err := c.CallWithTools(context.Background(), "sys", []Message{{Role: "user", Content: "hi"}}, nil)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected timeout error, got clean success")
	}
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("err = %v, want CodeTimeout ProviderError", err)
	}
	if elapsed > 3*time.Second {
		t.Fatalf("attempt budget took %s to fire, want <3s", elapsed)
	}
}

// TestAnthropicStreamCompletesUnderWatchdog: the happy path — a normal
// fast stream must complete cleanly with both brakes armed at defaults-ish
// values (the watchdog must NOT fire on instant events; cassette replay
// never even reaches this layer, but a fast live stream must not regress).
func TestAnthropicStreamCompletesUnderWatchdog(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sseHeaders(w)
		writeSSE(w, "message_start", sseMessageStart)
		writeSSE(w, "content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
		writeSSE(w, "content_block_delta", sseTextDelta("fast "))
		writeSSE(w, "content_block_delta", sseTextDelta("reply"))
		writeSSE(w, "content_block_stop", `{"type":"content_block_stop","index":0}`)
		writeSSE(w, "message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}`)
		writeSSE(w, "message_stop", `{"type":"message_stop"}`)
	}))
	defer srv.Close()

	c := newAnthropicAgainst(srv.URL, TimeoutConfig{}) // package defaults

	out, err := c.Stream(context.Background(), "hi", nil)
	if err != nil {
		t.Fatalf("fast stream must pass under the watchdog: %v", err)
	}
	if out != "fast reply" {
		t.Fatalf("out = %q, want %q", out, "fast reply")
	}
}

// ── Config plumbing + pure helpers ─────────────────────────────

func TestTimeoutConfigWithDefaults(t *testing.T) {
	got := TimeoutConfig{}.withDefaults()
	if got.PerAttempt != DefaultPerAttemptTimeout || got.StreamStall != DefaultStreamStallTimeout {
		t.Fatalf("zero config = %+v, want package defaults", got)
	}
	got = TimeoutConfig{PerAttempt: -1, StreamStall: -1}.withDefaults()
	if got.PerAttempt != 0 || got.StreamStall != 0 {
		t.Fatalf("negative config = %+v, want disabled (0)", got)
	}
	got = TimeoutConfig{PerAttempt: time.Minute, StreamStall: time.Second}.withDefaults()
	if got.PerAttempt != time.Minute || got.StreamStall != time.Second {
		t.Fatalf("explicit config = %+v, want passthrough", got)
	}
}

func TestDeadlineHeadroom(t *testing.T) {
	cfg := TimeoutConfig{PerAttempt: 240 * time.Second}
	now := time.Date(2099, time.January, 1, 0, 0, 0, 0, time.UTC)

	if _, near := deadlineHeadroom(context.Background(), cfg, now); near {
		t.Fatal("no deadline must never be 'near'")
	}

	ctx, cancel := context.WithDeadline(context.Background(), now.Add(10*time.Second))
	defer cancel()
	if remaining, near := deadlineHeadroom(ctx, cfg, now); !near {
		t.Fatalf("10s remaining of a 240s attempt budget must warn (remaining=%s)", remaining)
	}

	ctx2, cancel2 := context.WithDeadline(context.Background(), now.Add(10*time.Minute))
	defer cancel2()
	if _, near := deadlineHeadroom(ctx2, cfg, now); near {
		t.Fatal("10m remaining must not warn")
	}
}

// timeoutRecordingProvider is a minimal custom provider that records the
// TimeoutConfig NewClient hands it — verifies the WithTimeouts ClientOption
// plumbing without needing an API key.
type timeoutRecordingProvider struct {
	Client
	got *TimeoutConfig
}

func (p *timeoutRecordingProvider) SetTimeouts(cfg TimeoutConfig) { *p.got = cfg }

func TestWithTimeoutsOptionReachesProvider(t *testing.T) {
	var got TimeoutConfig
	p := &timeoutRecordingProvider{Client: newTransportProbe("ok"), got: &got}
	want := TimeoutConfig{PerAttempt: 90 * time.Second, StreamStall: 15 * time.Second}

	if _, err := NewClient("custom/model", Options{}, WithProvider(p, ModelInfo{Provider: "custom", Model: "model", ContextWindow: 4096, MaxOutput: 1024}), WithTimeouts(want)); err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	if got != want {
		t.Fatalf("provider received %+v, want %+v", got, want)
	}
}
