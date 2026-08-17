package llm

import (
	"context"
	"fmt"
	"math"
)

// BudgetExceededError is returned when a budget limit is reached.
type BudgetExceededError struct {
	CostSoFar float64
	Limit     float64
}

func (e *BudgetExceededError) Error() string {
	return fmt.Sprintf("budget exceeded: $%.4f spent, limit $%.2f", e.CostSoFar, e.Limit)
}

// SharedSpendBudget is one operator-owned live-provider spending brake shared
// by every model client participating in a run. It deliberately serializes
// live calls: otherwise several routed clients could all pass the preflight
// concurrently and multiply the admitted overshoot. Recorder middleware must
// remain outside this boundary so cassette hits neither consume nor block on a
// live-spend budget.
//
// The limit is checked before each call. The final admitted call can cross the
// limit because providers report exact usage only after completing it; no
// subsequent live call begins once that usage has been charged.
type SharedSpendBudget struct {
	gate     chan struct{}
	limitUSD float64
	spentUSD float64
	calls    int
}

// SharedSpendBudgetSnapshot is an immutable operator-facing view of the live
// calls charged to a shared budget.
type SharedSpendBudgetSnapshot struct {
	LimitUSD float64
	SpentUSD float64
	Calls    int
}

// NewSharedSpendBudget constructs a positive, finite live-provider budget.
func NewSharedSpendBudget(limitUSD float64) (*SharedSpendBudget, error) {
	if limitUSD <= 0 || math.IsNaN(limitUSD) || math.IsInf(limitUSD, 0) {
		return nil, fmt.Errorf("shared spend budget limit must be positive and finite, got %v", limitUSD)
	}
	budget := &SharedSpendBudget{limitUSD: limitUSD, gate: make(chan struct{}, 1)}
	budget.gate <- struct{}{}
	return budget, nil
}

// Snapshot returns the exact completed live spend observed by this process.
func (b *SharedSpendBudget) Snapshot() SharedSpendBudgetSnapshot {
	if b == nil || b.gate == nil {
		return SharedSpendBudgetSnapshot{}
	}
	<-b.gate
	defer func() { b.gate <- struct{}{} }()
	return SharedSpendBudgetSnapshot{LimitUSD: b.limitUSD, SpentUSD: b.spentUSD, Calls: b.calls}
}

func (b *SharedSpendBudget) begin(ctx context.Context) error {
	if b == nil {
		return nil
	}
	if b.gate == nil {
		return fmt.Errorf("shared spend budget is not initialized; use NewSharedSpendBudget")
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-b.gate:
	}
	if err := ctx.Err(); err != nil {
		b.gate <- struct{}{}
		return err
	}
	if b.spentUSD >= b.limitUSD {
		spent := b.spentUSD
		limit := b.limitUSD
		b.gate <- struct{}{}
		return &BudgetExceededError{CostSoFar: spent, Limit: limit}
	}
	return nil
}

func (b *SharedSpendBudget) finish(client Client) {
	if b == nil {
		return
	}
	if usage := latestClientUsage(client); usage != nil {
		if usage.CostUSD > 0 && !math.IsNaN(usage.CostUSD) && !math.IsInf(usage.CostUSD, 0) {
			b.spentUSD += usage.CostUSD
		}
		b.calls++
	}
	b.gate <- struct{}{}
}

// SharedSpendBudgetMiddleware binds a client to one cross-model live-spend
// account. Install it inside RecorderMiddleware and outside CostMiddleware.
func SharedSpendBudgetMiddleware(budget *SharedSpendBudget) Middleware {
	return func(next Client) Client {
		return &sharedSpendBudgetMW{next: next, budget: budget}
	}
}

type sharedSpendBudgetMW struct {
	next   Client
	budget *SharedSpendBudget
}

func (m *sharedSpendBudgetMW) Call(ctx context.Context, prompt string) (out string, err error) {
	if err := m.budget.begin(ctx); err != nil {
		return "", err
	}
	defer m.budget.finish(m.next)
	return m.next.Call(ctx, prompt)
}

func (m *sharedSpendBudgetMW) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (out string, err error) {
	if err := m.budget.begin(ctx); err != nil {
		return "", err
	}
	defer m.budget.finish(m.next)
	return CallWithOptions(ctx, m.next, prompt, opts)
}

func (m *sharedSpendBudgetMW) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (out string, err error) {
	return m.StreamWithOptions(ctx, prompt, RequestOptions{}, onChunk)
}

func (m *sharedSpendBudgetMW) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(text string) error) (out string, err error) {
	if err := m.budget.begin(ctx); err != nil {
		return "", err
	}
	defer m.budget.finish(m.next)
	return StreamWithOptions(ctx, m.next, prompt, opts, onChunk)
}

func (m *sharedSpendBudgetMW) CallCached(ctx context.Context, system, user string) (out string, err error) {
	return m.CallCachedWithOptions(ctx, system, user, RequestOptions{})
}

func (m *sharedSpendBudgetMW) CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (out string, err error) {
	if err := m.budget.begin(ctx); err != nil {
		return "", err
	}
	defer m.budget.finish(m.next)
	return CallCachedWithOptions(ctx, m.next, system, user, opts)
}

func (m *sharedSpendBudgetMW) CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (out Message, err error) {
	return m.CallWithToolsOptions(ctx, system, messages, tools, RequestOptions{})
}

func (m *sharedSpendBudgetMW) CallWithToolsOptions(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (out Message, err error) {
	if err := m.budget.begin(ctx); err != nil {
		return Message{}, err
	}
	defer m.budget.finish(m.next)
	return CallWithToolsOptions(ctx, m.next, system, messages, tools, opts)
}

func (m *sharedSpendBudgetMW) Unwrap() Client { return m.next }

// BudgetMiddleware returns a middleware that stops LLM calls once a cost ceiling
// is reached. This provides per-objective cost safety: if the iteration loop
// is spending too much, the middleware stops it before the next LLM call.
//
// The tracker is used to read the running total. The limit is in USD.
func BudgetMiddleware(tracker *CostTracker, limitUSD float64) Middleware {
	return func(next Client) Client {
		return &budgetMW{next: next, tracker: tracker, limit: limitUSD}
	}
}

type budgetMW struct {
	next    Client
	tracker *CostTracker
	limit   float64
}

func (m *budgetMW) Call(ctx context.Context, prompt string) (string, error) {
	return m.CallWithOptions(ctx, prompt, RequestOptions{})
}

func (m *budgetMW) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error) {
	if err := m.checkBudget(); err != nil {
		return "", err
	}
	return CallWithOptions(ctx, m.next, prompt, opts)
}

func (m *budgetMW) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (string, error) {
	return m.StreamWithOptions(ctx, prompt, RequestOptions{}, onChunk)
}

func (m *budgetMW) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(text string) error) (string, error) {
	if err := m.checkBudget(); err != nil {
		return "", err
	}
	return StreamWithOptions(ctx, m.next, prompt, opts, onChunk)
}

func (m *budgetMW) CallCached(ctx context.Context, system, user string) (string, error) {
	return m.CallCachedWithOptions(ctx, system, user, RequestOptions{})
}

func (m *budgetMW) CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (string, error) {
	if err := m.checkBudget(); err != nil {
		return "", err
	}
	return CallCachedWithOptions(ctx, m.next, system, user, opts)
}

func (m *budgetMW) CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (Message, error) {
	return m.CallWithToolsOptions(ctx, system, messages, tools, RequestOptions{})
}

func (m *budgetMW) CallWithToolsOptions(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error) {
	if err := m.checkBudget(); err != nil {
		return Message{}, err
	}
	return CallWithToolsOptions(ctx, m.next, system, messages, tools, opts)
}

func (m *budgetMW) Unwrap() Client { return m.next }

func (m *budgetMW) checkBudget() error {
	if m.limit <= 0 {
		return nil
	}
	if m.tracker == nil {
		return nil
	}
	total := m.tracker.Total()
	if total.CostUSD >= m.limit {
		return &BudgetExceededError{
			CostSoFar: total.CostUSD,
			Limit:     m.limit,
		}
	}
	return nil
}
