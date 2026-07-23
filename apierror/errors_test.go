package apierror

import (
	"errors"
	"fmt"
	"testing"
)

func TestProviderErrorMatchesSentinelByCode(t *testing.T) {
	err := &ProviderError{
		Provider: "openai",
		Code:     CodeRateLimited,
		Status:   429,
		Message:  "slow down",
	}

	if !errors.Is(err, ErrRateLimited) {
		t.Fatal("rate limited provider error did not match sentinel")
	}
	if errors.Is(err, ErrServer) {
		t.Fatal("rate limited provider error matched server sentinel")
	}
}

func TestProviderErrorUnwrap(t *testing.T) {
	cause := fmt.Errorf("sdk failed")
	err := &ProviderError{
		Provider: "anthropic",
		Code:     CodeServer,
		Status:   500,
		Message:  cause.Error(),
		Wrapped:  cause,
	}

	if !errors.Is(err, cause) {
		t.Fatal("provider error did not unwrap original error")
	}
}

func TestIsRetryable(t *testing.T) {
	retryable := []error{
		ErrRateLimited,
		ErrOverloaded,
		ErrServer,
		ErrStream,
		ErrTimeout,
		&ProviderError{Code: CodeRateLimited},
	}
	for _, err := range retryable {
		if !IsRetryable(err) {
			t.Fatalf("IsRetryable(%v) = false, want true", err)
		}
	}

	nonRetryable := []error{
		ErrUnauthorized,
		ErrBadRequest,
		ErrContextLength,
		ErrContentFiltered,
		ErrModelNotFound,
		&ProviderError{Code: CodeUnknown},
	}
	for _, err := range nonRetryable {
		if IsRetryable(err) {
			t.Fatalf("IsRetryable(%v) = true, want false", err)
		}
	}
}

func TestClassifyStatus(t *testing.T) {
	cases := map[int]ErrorCode{
		400: CodeBadRequest,
		401: CodeUnauthorized,
		403: CodeUnauthorized,
		404: CodeModelNotFound,
		408: CodeTimeout,
		413: CodeContextLength,
		422: CodeBadRequest,
		429: CodeRateLimited,
		500: CodeServer,
		503: CodeServer,
		504: CodeTimeout,
		529: CodeOverloaded,
		418: CodeUnknown,
	}
	for status, want := range cases {
		if got := ClassifyStatus(status); got != want {
			t.Fatalf("ClassifyStatus(%d) = %q, want %q", status, got, want)
		}
	}
}
