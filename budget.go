package llm

import (
	"context"
	"fmt"
)

// BudgetExceededError is returned when a budget limit is reached.
type BudgetExceededError struct {
	CostSoFar float64
	Limit     float64
}

func (e *BudgetExceededError) Error() string {
	return fmt.Sprintf("budget exceeded: $%.4f spent, limit $%.2f", e.CostSoFar, e.Limit)
}

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
