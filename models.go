package llm

import "fmt"

// ARCHITECTURE: ModelInfo lives in pkg/spec (the type). This file holds
// the impl-side catalog — pricing snapshots for every executable LLM Mind supports,
// plus the LookupModel helper that translates "provider/model-id" into a
// ModelInfo. Pricing is a moving target; keep edits to this file small and
// cross-check against the PROVIDER docs (the authoritative source):
//   - Anthropic: https://platform.claude.com/docs/en/about-claude/models/overview
//   - OpenAI:    https://developers.openai.com/api/docs/models
// This registry is the executable allowlist, not merely a pricing table.

// Canonical bare model identifiers (the "model-id" portion of
// "provider/model-id"). Use these in provider config (DefaultModel),
// test fixtures, and bench wiring instead of duplicating literal
// strings. The full "provider/model-id" form lives in
// pkg/spec/llm.go::ModelAnthropic*; the registry below keys on that
// full form.
const (
	// Anthropic current executable lineup (July 2026). These are canonical
	// pinned API IDs, not convenience aliases.
	AnthropicModelHaiku  = "claude-haiku-4-5-20251001"
	AnthropicModelSonnet = "claude-sonnet-5"
	AnthropicModelOpus   = "claude-opus-4-8"

	OpenAIModelGPT56Sol   = "gpt-5.6-sol"
	OpenAIModelGPT56Terra = "gpt-5.6-terra"
	OpenAIModelGPT56Luna  = "gpt-5.6-luna"
)

// modelRegistry is the built-in table of known models.
// Pricing is cross-checked against the Anthropic and OpenAI provider docs.
var modelRegistry = map[string]ModelInfo{
	// ── Anthropic (July 2026) ────────────────────────────────
	"anthropic/claude-opus-4-8": {
		Provider: "anthropic", Model: "claude-opus-4-8",
		ContextWindow: 1_000_000, MaxOutput: 128_000,
		InputPricePer1M: 5.00, OutputPricePer1M: 25.00,
	},
	"anthropic/claude-sonnet-5": {
		Provider: "anthropic", Model: "claude-sonnet-5",
		ContextWindow: 1_000_000, MaxOutput: 128_000,
		InputPricePer1M: 3.00, OutputPricePer1M: 15.00,
	},
	"anthropic/claude-haiku-4-5-20251001": {
		Provider: "anthropic", Model: "claude-haiku-4-5-20251001",
		ContextWindow: 200_000, MaxOutput: 64_000,
		InputPricePer1M: 1.00, OutputPricePer1M: 5.00,
	},
	// ── OpenAI (July 2026) ────────────────────────────────────
	"openai/gpt-5.6-sol": {
		Provider: "openai", Model: "gpt-5.6-sol",
		ContextWindow: 1_050_000, MaxOutput: 128_000,
		InputPricePer1M: 5.00, OutputPricePer1M: 30.00,
	},
	"openai/gpt-5.6-terra": {
		Provider: "openai", Model: "gpt-5.6-terra",
		ContextWindow: 1_050_000, MaxOutput: 128_000,
		InputPricePer1M: 2.50, OutputPricePer1M: 15.00,
	},
	"openai/gpt-5.6-luna": {
		Provider: "openai", Model: "gpt-5.6-luna",
		ContextWindow: 1_050_000, MaxOutput: 128_000,
		InputPricePer1M: 1.00, OutputPricePer1M: 6.00,
	},
}

// LookupModel is a closed-catalog lookup. Unknown models return false; callers
// must never invent pricing, token limits, or execution support for a stale ID.
func LookupModel(provider, model string) (ModelInfo, bool) {
	info, ok := modelRegistry[provider+"/"+model]
	return info, ok
}

// RequireModel is the error-returning production boundary for dynamic model
// identifiers. Its diagnostics include the exact rejected provider/model.
func RequireModel(provider, model string) (ModelInfo, error) {
	info, ok := LookupModel(provider, model)
	if !ok {
		return ModelInfo{}, fmt.Errorf("unsupported executable model %q", provider+"/"+model)
	}
	return info, nil
}

// MustLookupModel is for source-pinned constants and tests. Dynamic input must
// use RequireModel so an invalid ID becomes a normal configuration error.
func MustLookupModel(provider, model string) ModelInfo {
	info, err := RequireModel(provider, model)
	if err != nil {
		panic(err)
	}
	return info
}
