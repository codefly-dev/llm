package llm

import (
	"math"
	"sort"
	"testing"
)

func TestModelInfo_InputCost(t *testing.T) {
	info := ModelInfo{InputPricePer1M: 3.0}
	cost := InputCost(info, 1_000_000)
	if math.Abs(cost-3.0) > 1e-9 {
		t.Errorf("InputCost(1M) = %f, want 3.0", cost)
	}
}

func TestModelInfo_InputCost_Small(t *testing.T) {
	info := ModelInfo{InputPricePer1M: 3.0}
	cost := InputCost(info, 1000)
	expected := 3.0 / 1000.0 // 0.003
	if math.Abs(cost-expected) > 1e-9 {
		t.Errorf("InputCost(1000) = %f, want %f", cost, expected)
	}
}

func TestModelInfo_OutputCost(t *testing.T) {
	info := ModelInfo{OutputPricePer1M: 15.0}
	cost := OutputCost(info, 1_000_000)
	if math.Abs(cost-15.0) > 1e-9 {
		t.Errorf("OutputCost(1M) = %f, want 15.0", cost)
	}
}

func TestModelInfo_TotalCost_ZeroTokens(t *testing.T) {
	info := ModelInfo{InputPricePer1M: 3.0, OutputPricePer1M: 15.0}
	cost := TotalCost(info, 0, 0)
	if cost != 0 {
		t.Errorf("TotalCost(0, 0) = %f, want 0", cost)
	}
}

func TestModelInfo_TotalCost_Additive(t *testing.T) {
	info := ModelInfo{InputPricePer1M: 3.0, OutputPricePer1M: 15.0}
	total := TotalCost(info, 500_000, 500_000)
	expected := 1.5 + 7.5
	if math.Abs(total-expected) > 1e-9 {
		t.Errorf("TotalCost(500K, 500K) = %f, want %f", total, expected)
	}
}

func TestPricingProfile_PriceUsageByLevelAndComponent(t *testing.T) {
	info := ModelInfo{Pricing: PricingProfile{
		DefaultLevel: PriceLevelStandard,
		Rules: []PriceRule{
			{Component: PriceComponentInputTokens, Level: PriceLevelStandard, USD: 1, UnitSize: 1},
			{Component: PriceComponentOutputTokens, Level: PriceLevelStandard, USD: 2, UnitSize: 1},
			{Component: PriceComponentReasoningTokens, Level: PriceLevelStandard, USD: 3, UnitSize: 1},
			{Component: PriceComponentCacheReadInputTokens, Level: PriceLevelStandard, USD: 0.1, UnitSize: 1},
			{Component: PriceComponentCacheWriteInputTokens, Level: PriceLevelStandard, USD: 0.5, UnitSize: 1},
			{Component: PriceComponentRequest, Level: PriceLevelStandard, USD: 0.25, UnitSize: 1},
			{Component: PriceComponentToolCall, Level: PriceLevelStandard, USD: 1.5, UnitSize: 1},
			{Component: PriceComponentWebSearchRequest, Level: PriceLevelStandard, USD: 2, UnitSize: 1},
			{Component: PriceComponentWebFetchRequest, Level: PriceLevelStandard, USD: 0.25, UnitSize: 1},
			{Component: PriceComponentInputTokens, Level: PriceLevelPriority, USD: 10, UnitSize: 1},
		},
	}}

	cost := PriceUsage(info, Usage{
		InputTokens:           10,
		CachedInputTokens:     4,
		CacheReadInputTokens:  4,
		CacheWriteInputTokens: 2,
		OutputTokens:          8,
		ReasoningTokens:       3,
		ToolCalls:             9,
		HostedToolCalls:       1,
		WebSearchRequests:     1,
		WebFetchRequests:      2,
	})

	if cost.InputUSD != 6 {
		t.Fatalf("InputUSD = %f, want 6", cost.InputUSD)
	}
	if cost.OutputUSD != 10 {
		t.Fatalf("OutputUSD = %f, want 10", cost.OutputUSD)
	}
	if math.Abs(cost.TotalUSD-30.65) > 1e-9 {
		t.Fatalf("TotalUSD = %f, want 30.65", cost.TotalUSD)
	}
	if cost.RequestUSD != 0.25 || cost.ToolUSD != 1.5 || cost.WebSearchUSD != 2 || cost.WebFetchUSD != 0.5 {
		t.Fatalf("hosted/tool costs = %+v, want request=.25 tool=1.5 web_search=2 web_fetch=.5", cost)
	}
	if got := PriceFor(info, PriceComponentInputTokens, PriceLevelPriority, 2); got != 20 {
		t.Fatalf("priority input = %f, want 20", got)
	}
}

func TestFlatTokenPricing_DoesNotDoubleChargeReasoningTokens(t *testing.T) {
	info := ModelInfo{InputPricePer1M: 1, OutputPricePer1M: 10}
	cost := PriceUsage(info, Usage{OutputTokens: 100, ReasoningTokens: 40})
	want := 100.0 * 10 / 1_000_000
	if math.Abs(cost.TotalUSD-want) > 1e-12 {
		t.Fatalf("TotalUSD = %.12f, want %.12f", cost.TotalUSD, want)
	}
	if cost.OutputUSD <= 0 || cost.ReasoningUSD <= 0 {
		t.Fatalf("expected output and reasoning split, got %+v", cost)
	}
}

func TestPricingProfile_ReasoningCanBeOutputExtra(t *testing.T) {
	info := ModelInfo{InputPricePer1M: 1, OutputPricePer1M: 10}
	cost := PriceUsage(info, Usage{OutputTokens: 100, ReasoningTokens: 40, ReasoningOutputExtra: true})
	want := 140.0 * 10 / 1_000_000
	if math.Abs(cost.TotalUSD-want) > 1e-12 {
		t.Fatalf("TotalUSD = %.12f, want %.12f", cost.TotalUSD, want)
	}
}

func TestPricingProfile_UsesProfileDefaultLevel(t *testing.T) {
	info := ModelInfo{Pricing: PricingProfile{
		DefaultLevel: PriceLevelBatch,
		Rules: []PriceRule{
			{Component: PriceComponentInputTokens, Level: PriceLevelBatch, USD: 2, UnitSize: 1},
			{Component: PriceComponentInputTokens, Level: PriceLevelStandard, USD: 10, UnitSize: 1},
		},
	}}
	cost := PriceUsage(info, Usage{InputTokens: 3})
	if cost.InputUSD != 6 {
		t.Fatalf("InputUSD = %f, want batch default cost 6", cost.InputUSD)
	}
}

func TestLookupModel_AllKnownProviders(t *testing.T) {
	cases := []struct {
		provider string
		model    string
	}{
		{"anthropic", AnthropicModelHaiku},
		{"anthropic", AnthropicModelSonnet},
		{"anthropic", AnthropicModelOpus},
		{"openai", "gpt-5.6-sol"},
		{"openai", "gpt-5.6-terra"},
		{"openai", "gpt-5.6-luna"},
	}

	for _, tc := range cases {
		t.Run(tc.provider+"/"+tc.model, func(t *testing.T) {
			info, ok := LookupModel(tc.provider, tc.model)
			if !ok {
				t.Fatal("current model missing from closed catalog")
			}
			if info.Provider != tc.provider {
				t.Errorf("Provider = %q, want %q", info.Provider, tc.provider)
			}
			if info.ContextWindow <= 0 {
				t.Error("ContextWindow should be positive")
			}
			if info.MaxOutput <= 0 {
				t.Error("MaxOutput should be positive")
			}
			if info.InputPricePer1M <= 0 {
				t.Error("InputPricePer1M should be positive")
			}
			if info.OutputPricePer1M <= 0 {
				t.Error("OutputPricePer1M should be positive")
			}
		})
	}
}

func TestExecutableAnthropicRosterIsExactlyHaiku45Sonnet5Opus48(t *testing.T) {
	var got []string
	for _, info := range modelRegistry {
		if info.Provider == "anthropic" {
			got = append(got, info.Model)
		}
	}
	sort.Strings(got)
	want := []string{AnthropicModelHaiku, AnthropicModelOpus, AnthropicModelSonnet}
	sort.Strings(want)
	if len(got) != len(want) {
		t.Fatalf("Anthropic executable roster = %v, want exactly %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Anthropic executable roster = %v, want exactly %v", got, want)
		}
	}
}

func TestLookupModel_UnknownFailsClosed(t *testing.T) {
	if info, ok := LookupModel("fake-provider", "fake-model"); ok || info.Provider != "" || info.Model != "" || info.ContextWindow != 0 {
		t.Fatalf("unknown model resolved to %+v", info)
	}
	if _, err := RequireModel("anthropic", "claude-sonnet-4-6"); err == nil {
		t.Fatal("stale Anthropic model must fail closed")
	}
}

func TestLookupModel_Sonnet5(t *testing.T) {
	info := MustLookupModel("anthropic", AnthropicModelSonnet)
	if info.Model != AnthropicModelSonnet {
		t.Errorf("Model = %q, want %s", info.Model, AnthropicModelSonnet)
	}
	if info.InputPricePer1M != 3.00 || info.OutputPricePer1M != 15.00 {
		t.Errorf("pricing = $%.2f/$%.2f, want $3.00/$15.00", info.InputPricePer1M, info.OutputPricePer1M)
	}
	if info.ContextWindow != 1_000_000 || info.MaxOutput != 128_000 {
		t.Errorf("limits = %d/%d, want 1000000/128000", info.ContextWindow, info.MaxOutput)
	}
}

func TestLookupModel_Opus48(t *testing.T) {
	info := MustLookupModel("anthropic", AnthropicModelOpus)
	if info.Model != AnthropicModelOpus {
		t.Errorf("Model = %q, want %s", info.Model, AnthropicModelOpus)
	}
	if info.InputPricePer1M != 5.00 || info.OutputPricePer1M != 25.00 || info.MaxOutput != 128_000 {
		t.Errorf("Opus catalog entry = %+v", info)
	}
}

func TestUsage_CacheHitRate_NoInput(t *testing.T) {
	u := Usage{InputTokens: 0, CachedTokens: 0}
	if CacheHitRate(u) != 0 {
		t.Errorf("CacheHitRate with zero input = %f", CacheHitRate(u))
	}
}

func TestUsage_CacheHitRate_AllCached(t *testing.T) {
	u := Usage{InputTokens: 1000, CachedTokens: 1000}
	rate := CacheHitRate(u)
	if math.Abs(rate-1.0) > 1e-9 {
		t.Errorf("CacheHitRate all cached = %f, want 1.0", rate)
	}
}

func TestUsage_CacheHitRate_Partial(t *testing.T) {
	u := Usage{InputTokens: 1000, CachedTokens: 250}
	rate := CacheHitRate(u)
	if math.Abs(rate-0.25) > 1e-9 {
		t.Errorf("CacheHitRate partial = %f, want 0.25", rate)
	}
}

func TestUsage_CacheHitRate_NegativeInput(t *testing.T) {
	u := Usage{InputTokens: -1, CachedTokens: 0}
	if CacheHitRate(u) != 0 {
		t.Errorf("CacheHitRate with negative input = %f", CacheHitRate(u))
	}
}

func TestModelInfo_OpusHigherPriceThanSonnet(t *testing.T) {
	opus := MustLookupModel("anthropic", AnthropicModelOpus)
	sonnet := MustLookupModel("anthropic", AnthropicModelSonnet)

	if opus.OutputPricePer1M <= sonnet.OutputPricePer1M {
		t.Errorf("Opus output ($%.2f) should be more expensive than Sonnet ($%.2f)",
			opus.OutputPricePer1M, sonnet.OutputPricePer1M)
	}
}

func TestModelInfo_CurrentFrontierContextWindows(t *testing.T) {
	anthropic := MustLookupModel("anthropic", AnthropicModelSonnet)
	openai := MustLookupModel("openai", OpenAIModelGPT56Sol)
	if openai.ContextWindow != 1_050_000 || anthropic.ContextWindow != 1_000_000 {
		t.Errorf("current context windows = openai:%d anthropic:%d, want 1050000/1000000", openai.ContextWindow, anthropic.ContextWindow)
	}
}
