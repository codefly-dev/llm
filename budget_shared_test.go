package llm

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/benbjohnson/clock"
)

type blockingSpendClient struct {
	*transportProbeClient
	entered chan struct{}
	release chan struct{}
}

func (c *blockingSpendClient) Call(ctx context.Context, _ string) (string, error) {
	close(c.entered)
	select {
	case <-ctx.Done():
		return "", ctx.Err()
	case <-c.release:
		return c.response, nil
	}
}

func TestNewSharedSpendBudgetRejectsInvalidLimits(t *testing.T) {
	for _, limit := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := NewSharedSpendBudget(limit); err == nil {
			t.Fatalf("NewSharedSpendBudget(%v) succeeded", limit)
		}
	}
	client := SharedSpendBudgetMiddleware(&SharedSpendBudget{})(newTransportProbe("must not execute"))
	if _, err := client.Call(context.Background(), "invalid zero-value budget"); err == nil {
		t.Fatal("zero-value shared budget crossed the provider boundary")
	}
}

func TestSharedSpendBudgetStopsAcrossModelClients(t *testing.T) {
	budget, err := NewSharedSpendBudget(0.000000001)
	if err != nil {
		t.Fatal(err)
	}
	firstTransport := newTransportProbe("first response")
	secondTransport := newTransportProbe("second response")
	first := Chain(
		firstTransport,
		SharedSpendBudgetMiddleware(budget),
		CostMiddleware(NewCostTracker(MustLookupModel("openai", OpenAIModelGPT56Luna), clock.NewMock())),
	)
	second := Chain(
		secondTransport,
		SharedSpendBudgetMiddleware(budget),
		CostMiddleware(NewCostTracker(MustLookupModel("openai", OpenAIModelGPT56Sol), clock.NewMock())),
	)

	if _, err := first.Call(context.Background(), "charge the shared account"); err != nil {
		t.Fatal(err)
	}
	snapshot := budget.Snapshot()
	if snapshot.Calls != 1 || snapshot.SpentUSD <= snapshot.LimitUSD {
		t.Fatalf("snapshot after first model = %+v", snapshot)
	}
	if _, err := second.Call(context.Background(), "must not cross the provider boundary"); err == nil {
		t.Fatal("second model call succeeded after shared budget was exhausted")
	} else {
		var exceeded *BudgetExceededError
		if !errors.As(err, &exceeded) {
			t.Fatalf("second model error = %T %v", err, err)
		}
	}
	secondTransport.mu.Lock()
	secondCalls := secondTransport.calls
	secondTransport.mu.Unlock()
	if secondCalls != 0 {
		t.Fatalf("second model crossed the provider boundary %d times", secondCalls)
	}
}

func TestSharedSpendBudgetWaitHonorsCallerCancellation(t *testing.T) {
	budget, err := NewSharedSpendBudget(10)
	if err != nil {
		t.Fatal(err)
	}
	info := MustLookupModel("openai", OpenAIModelGPT56Luna)
	blocking := &blockingSpendClient{
		transportProbeClient: newTransportProbe("first response"),
		entered:              make(chan struct{}),
		release:              make(chan struct{}),
	}
	first := Chain(blocking, SharedSpendBudgetMiddleware(budget), CostMiddleware(NewCostTracker(info, clock.NewMock())))
	secondTransport := newTransportProbe("must not execute")
	second := Chain(secondTransport, SharedSpendBudgetMiddleware(budget), CostMiddleware(NewCostTracker(info, clock.NewMock())))

	firstDone := make(chan error, 1)
	go func() {
		_, callErr := first.Call(context.Background(), "hold the shared live-call permit")
		firstDone <- callErr
	}()
	<-blocking.entered

	canceled, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := second.Call(canceled, "do not wait beyond caller cancellation"); !errors.Is(err, context.Canceled) {
		t.Fatalf("waiting call error = %v, want context cancellation", err)
	}
	close(blocking.release)
	if err := <-firstDone; err != nil {
		t.Fatal(err)
	}
	secondTransport.mu.Lock()
	secondCalls := secondTransport.calls
	secondTransport.mu.Unlock()
	if secondCalls != 0 {
		t.Fatalf("canceled waiting call crossed the provider boundary %d times", secondCalls)
	}
}

func TestRecorderReplayBypassesExhaustedSharedSpendBudget(t *testing.T) {
	budget, err := NewSharedSpendBudget(0.000000001)
	if err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	model := "openai/" + OpenAIModelGPT56Luna
	prompt := "record once and replay without live spend"
	info := MustLookupModel("openai", OpenAIModelGPT56Luna)

	recordTransport := newTransportProbe("durable response")
	record := Chain(
		recordTransport,
		RecorderMiddlewareWithConfig(RecorderConfig{Dir: dir, Model: model, Mode: RecordAlways}),
		SharedSpendBudgetMiddleware(budget),
		CostMiddleware(NewCostTracker(info, clock.NewMock())),
	)
	if _, err := record.Call(context.Background(), prompt); err != nil {
		t.Fatal(err)
	}
	exhausted := budget.Snapshot()
	if exhausted.Calls != 1 || exhausted.SpentUSD <= exhausted.LimitUSD {
		t.Fatalf("recording did not exhaust shared budget: %+v", exhausted)
	}

	replayTransport := newTransportProbe("must not execute")
	replay := Chain(
		replayTransport,
		RecorderMiddlewareWithConfig(RecorderConfig{Dir: dir, Model: model, Mode: RecordReplayOnly}),
		SharedSpendBudgetMiddleware(budget),
		CostMiddleware(NewCostTracker(info, clock.NewMock())),
	)
	response, err := replay.Call(context.Background(), prompt)
	if err != nil || response != "durable response" {
		t.Fatalf("replay = %q, %v", response, err)
	}
	if got := budget.Snapshot(); got != exhausted {
		t.Fatalf("replay changed live spend: before=%+v after=%+v", exhausted, got)
	}
	replayTransport.mu.Lock()
	replayCalls := replayTransport.calls
	replayTransport.mu.Unlock()
	if replayCalls != 0 {
		t.Fatalf("replay crossed the provider boundary %d times", replayCalls)
	}
}
