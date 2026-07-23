package llm

func InputCost(info ModelInfo, tokens int) float64 {
	return PriceFor(info, PriceComponentInputTokens, PriceLevelStandard, tokens)
}

func OutputCost(info ModelInfo, tokens int) float64 {
	return PriceFor(info, PriceComponentOutputTokens, PriceLevelStandard, tokens)
}

func TotalCost(info ModelInfo, inputTokens, outputTokens int) float64 {
	return InputCost(info, inputTokens) + OutputCost(info, outputTokens)
}

func PriceFor(info ModelInfo, component PriceComponent, level PriceLevel, quantity int) float64 {
	return pricingProfilePriceFor(effectivePricing(info), component, level, quantity)
}

func PriceUsage(info ModelInfo, u Usage) UsageCost {
	return pricingProfilePriceUsage(effectivePricing(info), u, PriceLevel(u.PriceLevel))
}

func CacheHitRate(u Usage) float64 {
	if u.InputTokens <= 0 {
		return 0
	}
	return float64(u.CachedTokens) / float64(u.InputTokens)
}

func effectivePricing(info ModelInfo) PricingProfile {
	if len(info.Pricing.Rules) > 0 {
		return info.Pricing
	}
	return FlatTokenPricing(info.InputPricePer1M, info.OutputPricePer1M)
}

func FlatTokenPricing(inputPer1M, outputPer1M float64) PricingProfile {
	return PricingProfile{
		Currency:     "USD",
		DefaultLevel: PriceLevelStandard,
		Rules: []PriceRule{
			{Component: PriceComponentInputTokens, Level: PriceLevelStandard, USD: inputPer1M, UnitSize: 1_000_000},
			{Component: PriceComponentOutputTokens, Level: PriceLevelStandard, USD: outputPer1M, UnitSize: 1_000_000},
			{Component: PriceComponentReasoningTokens, Level: PriceLevelStandard, USD: outputPer1M, UnitSize: 1_000_000},
			{Component: PriceComponentCacheReadInputTokens, Level: PriceLevelStandard, USD: inputPer1M * 0.1, UnitSize: 1_000_000},
			{Component: PriceComponentCacheWriteInputTokens, Level: PriceLevelStandard, USD: inputPer1M, UnitSize: 1_000_000},
		},
	}
}

func pricingProfilePriceFor(p PricingProfile, component PriceComponent, level PriceLevel, quantity int) float64 {
	if quantity <= 0 {
		return 0
	}
	rule, ok := pricingProfileFindRule(p, component, level)
	if !ok {
		return 0
	}
	unit := rule.UnitSize
	if unit <= 0 {
		unit = 1
	}
	return float64(quantity) * rule.USD / float64(unit)
}

func pricingProfilePriceUsage(p PricingProfile, u Usage, level PriceLevel) UsageCost {
	freshInput := u.InputTokens - u.CachedInputTokens
	if freshInput < 0 {
		freshInput = 0
	}
	if freshInput == u.InputTokens && u.CacheReadInputTokens > 0 {
		freshInput = u.InputTokens - u.CacheReadInputTokens
		if freshInput < 0 {
			freshInput = 0
		}
	}
	visibleOutput := u.OutputTokens
	if !u.ReasoningOutputExtra {
		visibleOutput -= u.ReasoningTokens
		if visibleOutput < 0 {
			visibleOutput = 0
		}
	}
	cost := UsageCost{
		InputUSD:      pricingProfilePriceFor(p, PriceComponentInputTokens, level, freshInput),
		OutputUSD:     pricingProfilePriceFor(p, PriceComponentOutputTokens, level, visibleOutput),
		ReasoningUSD:  pricingProfilePriceFor(p, PriceComponentReasoningTokens, level, u.ReasoningTokens),
		CacheReadUSD:  pricingProfilePriceFor(p, PriceComponentCacheReadInputTokens, level, u.CacheReadInputTokens),
		CacheWriteUSD: pricingProfilePriceFor(p, PriceComponentCacheWriteInputTokens, level, u.CacheWriteInputTokens),
		RequestUSD:    pricingProfilePriceFor(p, PriceComponentRequest, level, requestCount(u)),
		ToolUSD:       pricingProfilePriceFor(p, PriceComponentToolCall, level, u.HostedToolCalls),
		WebSearchUSD:  pricingProfilePriceFor(p, PriceComponentWebSearchRequest, level, u.WebSearchRequests),
		WebFetchUSD:   pricingProfilePriceFor(p, PriceComponentWebFetchRequest, level, u.WebFetchRequests),
	}
	cost.TotalUSD = cost.InputUSD + cost.OutputUSD + cost.ReasoningUSD +
		cost.CacheReadUSD + cost.CacheWriteUSD +
		cost.RequestUSD + cost.ToolUSD + cost.WebSearchUSD + cost.WebFetchUSD
	return cost
}

func requestCount(u Usage) int {
	if u.Requests > 0 {
		return u.Requests
	}
	if u.InputTokens > 0 || u.OutputTokens > 0 || u.ToolCalls > 0 || u.HostedToolCalls > 0 {
		return 1
	}
	return 0
}

func pricingProfileFindRule(p PricingProfile, component PriceComponent, level PriceLevel) (PriceRule, bool) {
	if level == "" {
		level = p.DefaultLevel
	}
	if level == "" {
		level = PriceLevelStandard
	}
	for _, r := range p.Rules {
		if r.Component == component && r.Level == level {
			return r, true
		}
	}
	for _, r := range p.Rules {
		if r.Component == component && (r.Level == "" || r.Level == p.DefaultLevel || r.Level == PriceLevelStandard) {
			return r, true
		}
	}
	return PriceRule{}, false
}
