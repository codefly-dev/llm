package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	openai "github.com/openai/openai-go"
)

func TestClassifyStatus_Buckets(t *testing.T) {
	cases := map[int]ErrorCode{
		200: CodeUnknown,
		400: CodeBadRequest,
		401: CodeUnauthorized,
		403: CodeUnauthorized,
		404: CodeModelNotFound,
		408: CodeTimeout,
		413: CodeContextLength,
		422: CodeBadRequest,
		429: CodeRateLimited,
		500: CodeServer,
		502: CodeServer,
		503: CodeServer,
		504: CodeTimeout,
		529: CodeOverloaded,
	}
	for status, want := range cases {
		if got := ClassifyStatus(status); got != want {
			t.Errorf("ClassifyStatus(%d) = %q, want %q", status, got, want)
		}
	}
}

func TestProviderError_IsMatching(t *testing.T) {
	err := &ProviderError{Provider: "anthropic", Code: CodeRateLimited, Status: 429}

	if !errors.Is(err, ErrRateLimited) {
		t.Error("ProviderError{rate_limited} did not match ErrRateLimited sentinel")
	}
	if errors.Is(err, ErrServer) {
		t.Error("ProviderError{rate_limited} incorrectly matched ErrServer")
	}
}

func TestProviderError_UnwrapPreservesOriginal(t *testing.T) {
	inner := fmt.Errorf("raw sdk failure")
	wrapped := &ProviderError{
		Provider: "openai",
		Code:     CodeServer,
		Status:   503,
		Message:  inner.Error(),
		Wrapped:  inner,
	}
	if got := errors.Unwrap(wrapped); got != inner {
		t.Errorf("Unwrap = %v, want %v", got, inner)
	}
}

func TestClassifyOpenAI_UsesExactProviderName(t *testing.T) {
	err := classifyOpenAI(context.DeadlineExceeded, time.Unix(0, 0).UTC())
	var pe *ProviderError
	if !errors.As(err, &pe) {
		t.Fatalf("classified error = %T, want ProviderError", err)
	}
	if pe.Provider != "openai" {
		t.Fatalf("Provider = %q, want openai", pe.Provider)
	}
	if pe.Code != CodeTimeout {
		t.Fatalf("Code = %q, want %q", pe.Code, CodeTimeout)
	}
}

func TestClassifyAnthropic_CreditBalanceExhaustionIsTerminal(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/messages", nil)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	apiErr := &anthropic.Error{
		StatusCode: http.StatusBadRequest,
		Request:    req,
		Response: &http.Response{
			StatusCode: http.StatusBadRequest,
			Header:     make(http.Header),
		},
	}
	payload := []byte(`{"type":"error","error":{"type":"invalid_request_error","message":"Your credit balance is too low to access the Anthropic API. Please go to Plans & Billing to upgrade or purchase credits."}}`)
	if err := json.Unmarshal(payload, apiErr); err != nil {
		t.Fatalf("decode Anthropic error: %v", err)
	}

	classified := classifyAnthropic(apiErr, time.Unix(0, 0).UTC())
	if !errors.Is(classified, ErrQuotaExhausted) {
		t.Fatalf("classified error = %v, want quota exhausted", classified)
	}
	if IsRetryable(classified) {
		t.Fatalf("credit exhaustion must fail without retry, got %v", classified)
	}
	var providerErr *ProviderError
	if !errors.As(classified, &providerErr) || providerErr.Status != http.StatusBadRequest {
		t.Fatalf("classified error = %+v, want original HTTP 400 evidence", classified)
	}
}

func TestClassifyOpenAI_QuotaExhaustionIsTerminal(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/responses", nil)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	apiErr := &openai.Error{
		Code:       "insufficient_quota",
		Message:    "You exceeded your current quota.",
		Type:       "insufficient_quota",
		StatusCode: http.StatusTooManyRequests,
		Request:    req,
		Response: &http.Response{
			StatusCode: http.StatusTooManyRequests,
			Header:     make(http.Header),
		},
	}

	classified := classifyOpenAI(apiErr, time.Unix(0, 0).UTC())
	if !errors.Is(classified, ErrQuotaExhausted) {
		t.Fatalf("classified error = %v, want quota exhausted", classified)
	}
	if IsRetryable(classified) {
		t.Fatalf("quota exhaustion must fail without retry, got %v", classified)
	}
}

// TestClassifyOpenAI_GatewayQuotaExhaustionIsTerminal covers OpenAI-compatible
// gateways that forward the upstream billing message while stripping the
// structured Code/Type fields. The billing-exhaustion phrase alone must still
// classify as terminal so a proxied quota error does not get retried.
func TestClassifyOpenAI_GatewayQuotaExhaustionIsTerminal(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://gateway.example.com/v1/responses", nil)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	apiErr := &openai.Error{
		Message:    "You exceeded your current quota, please check your plan and billing details.",
		StatusCode: http.StatusTooManyRequests,
		Request:    req,
		Response: &http.Response{
			StatusCode: http.StatusTooManyRequests,
			Header:     make(http.Header),
		},
	}

	classified := classifyOpenAI(apiErr, time.Unix(0, 0).UTC())
	if !errors.Is(classified, ErrQuotaExhausted) {
		t.Fatalf("classified error = %v, want quota exhausted", classified)
	}
	if IsRetryable(classified) {
		t.Fatalf("proxied quota exhaustion must fail without retry, got %v", classified)
	}
}

// TestClassifyOpenAI_TransientRateLimitStaysRetryable guards against
// over-matching the quota phrases: a genuine per-minute throttle carries no
// billing-exhaustion text and must stay a retryable rate limit.
func TestClassifyOpenAI_TransientRateLimitStaysRetryable(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/responses", nil)
	if err != nil {
		t.Fatalf("build request: %v", err)
	}
	apiErr := &openai.Error{
		Code:       "rate_limit_exceeded",
		Type:       "requests",
		Message:    "Rate limit reached for gpt-4o in organization org-123 on requests per min. Please try again in 20s.",
		StatusCode: http.StatusTooManyRequests,
		Request:    req,
		Response: &http.Response{
			StatusCode: http.StatusTooManyRequests,
			Header:     make(http.Header),
		},
	}

	classified := classifyOpenAI(apiErr, time.Unix(0, 0).UTC())
	if errors.Is(classified, ErrQuotaExhausted) {
		t.Fatalf("transient rate limit must not classify as quota exhausted, got %v", classified)
	}
	if !errors.Is(classified, ErrRateLimited) {
		t.Fatalf("classified error = %v, want rate limited", classified)
	}
	if !IsRetryable(classified) {
		t.Fatalf("transient rate limit must be retryable, got %v", classified)
	}
}

func TestIsRetryable_Sentinels(t *testing.T) {
	// The retryable set is rate-limited, overloaded, server, timeout.
	retryable := []*ProviderError{
		{Code: CodeRateLimited},
		{Code: CodeOverloaded},
		{Code: CodeServer},
		{Code: CodeTimeout},
	}
	for _, e := range retryable {
		if !IsRetryable(e) {
			t.Errorf("IsRetryable(%v) = false, want true", e.Code)
		}
	}

	nonRetryable := []*ProviderError{
		{Code: CodeUnauthorized},
		{Code: CodeQuotaExhausted},
		{Code: CodeBadRequest},
		{Code: CodeUnknown},
	}
	for _, e := range nonRetryable {
		if IsRetryable(e) {
			t.Errorf("IsRetryable(%v) = true, want false", e.Code)
		}
	}
}

func TestIsRetryableRequiresTypedError(t *testing.T) {
	if isRetryable(fmt.Errorf("openai api: 429 Too Many Requests")) {
		t.Error("arbitrary status text must not drive retry control flow")
	}
	if !isRetryable(context.DeadlineExceeded) {
		t.Error("typed transport timeout should be retryable")
	}
	if isRetryable(nil) {
		t.Error("nil err should not be retryable")
	}
}
