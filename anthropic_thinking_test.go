package llm

import (
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
)

// applyAnthropicOptions must translate reasoning effort into budgeted
// extended thinking for extended-thinking models, and must NOT set a
// budget for adaptive models (Sonnet 5 and Opus 4.8), unknown models,
// or ReasoningNone.
func TestApplyAnthropicOptionsBudgetedThinking(t *testing.T) {
	cases := []struct {
		name       string
		model      string
		effort     ReasoningEffort
		wantBudget int64 // 0 = thinking must be unset
	}{
		{"haiku45 low", "claude-haiku-4-5-20251001", ReasoningLow, 2048},
		{"haiku45 high", "claude-haiku-4-5-20251001", ReasoningHigh, 12288},
		{"haiku45 none", "claude-haiku-4-5-20251001", ReasoningNone, 0},
		{"sonnet5 high → no budget", "claude-sonnet-5", ReasoningHigh, 0},
		{"opus48 high → no budget", "claude-opus-4-8", ReasoningHigh, 0},
		{"unknown high → no budget", "claude-future-6", ReasoningHigh, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			params := anthropic.MessageNewParams{Model: anthropic.Model(tc.model), MaxTokens: 8192}
			applyAnthropicOptions(&params, RequestOptions{ReasoningEffort: tc.effort})

			if tc.wantBudget == 0 {
				if params.Thinking.OfEnabled != nil {
					t.Fatalf("expected no budgeted thinking, got budget=%d", params.Thinking.OfEnabled.BudgetTokens)
				}
				return
			}
			if params.Thinking.OfEnabled == nil {
				t.Fatal("expected budgeted thinking to be set")
			}
			if params.Thinking.OfEnabled.BudgetTokens != tc.wantBudget {
				t.Errorf("budget = %d, want %d", params.Thinking.OfEnabled.BudgetTokens, tc.wantBudget)
			}
			// max_tokens must exceed the thinking budget (API requirement).
			if params.MaxTokens <= tc.wantBudget {
				t.Errorf("max_tokens %d must exceed budget %d", params.MaxTokens, tc.wantBudget)
			}
		})
	}
}

func TestAnthropicSupportsBudgetedThinking(t *testing.T) {
	yes := []string{AnthropicModelHaiku}
	no := []string{AnthropicModelSonnet, AnthropicModelOpus, "claude-future-6"}
	for _, m := range yes {
		if !anthropicSupportsBudgetedThinking(m) {
			t.Errorf("%s should support budgeted thinking", m)
		}
	}
	for _, m := range no {
		if anthropicSupportsBudgetedThinking(m) {
			t.Errorf("%s should NOT support budgeted thinking (effort/adaptive)", m)
		}
	}
}
