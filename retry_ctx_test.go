package llm

// Regression tests for the retry middleware's caller-context discipline:
// a dead caller ctx must never be retried (a real trace showed 4 attempts
// fired after the caller's deadline had already expired). Error-injection
// clients are the sanctioned exception for failure paths real infra cannot
// produce on demand (CLAUDE.md).

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// cancellingClient fails with a retryable error AND kills the caller's ctx
// during the call — simulating "the deadline fired while the provider call
// was in flight".
type cancellingClient struct {
	cancel context.CancelFunc
	calls  int
}

func (c *cancellingClient) Call(ctx context.Context, prompt string) (string, error) {
	c.calls++
	c.cancel()
	return "", &ProviderError{Provider: "test", Code: CodeServer, Status: 503, Message: "unavailable"}
}

func (c *cancellingClient) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (string, error) {
	return c.Call(ctx, prompt)
}

func TestRetry_NoAttemptWhenCallerCtxAlreadyDead(t *testing.T) {
	fc := &failNClient{failCount: 10, failErr: &ProviderError{Provider: "test", Code: CodeServer, Status: 503}, response: "never"}
	client := Chain(fc, RetryMiddleware(3, time.Millisecond))

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := client.Call(ctx, "prompt")
	if err == nil {
		t.Fatal("expected error on dead ctx")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want wrapped context.Canceled", err)
	}
	if fc.calls != 0 {
		t.Fatalf("provider calls = %d, want 0 (never attempt on a dead caller ctx)", fc.calls)
	}
}

func TestRetry_CallWithTools_NoAttemptWhenCallerCtxExpired(t *testing.T) {
	fc := &failNClient{failCount: 10, failErr: &ProviderError{Provider: "test", Code: CodeServer, Status: 503}, response: "never"}
	client := Chain(fc, RetryMiddleware(3, time.Millisecond))

	ctx, cancel := context.WithDeadline(context.Background(), time.Unix(1, 0))
	defer cancel()

	tc, ok := AsToolClient(client)
	if !ok {
		t.Fatal("retry middleware must preserve native tool capability")
	}
	_, err := tc.CallWithTools(ctx, "sys", []Message{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Fatal("expected error on expired ctx")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v, want wrapped context.DeadlineExceeded", err)
	}
	if fc.calls != 0 {
		t.Fatalf("provider calls = %d, want 0 (never attempt on an expired caller ctx)", fc.calls)
	}
}

func TestRetry_StopsWhenCtxDiesMidRun(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	cc := &cancellingClient{cancel: cancel}
	client := Chain(cc, RetryMiddleware(3, time.Millisecond))

	_, err := client.Call(ctx, "prompt")
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("err = %v, want wrapped context.Canceled", err)
	}
	// The last provider error must survive as text so the failure trace
	// shows WHAT was being retried when the clock ran out.
	if !strings.Contains(err.Error(), "503") {
		t.Fatalf("err = %v, want last provider error preserved in message", err)
	}
	if cc.calls != 1 {
		t.Fatalf("provider calls = %d, want exactly 1 (no retry after ctx died mid-run)", cc.calls)
	}
}
