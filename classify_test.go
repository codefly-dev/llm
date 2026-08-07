package llm

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"testing"
	"time"

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
