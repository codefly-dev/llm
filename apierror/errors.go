package apierror

import (
	"errors"
	"fmt"
	"time"
)

// ErrorCode classifies a provider error in a transport-agnostic way.
type ErrorCode string

const (
	CodeRateLimited     ErrorCode = "rate_limited"
	CodeQuotaExhausted  ErrorCode = "quota_exhausted"
	CodeOverloaded      ErrorCode = "overloaded"
	CodeServer          ErrorCode = "server_error"
	CodeUnauthorized    ErrorCode = "unauthorized"
	CodeBadRequest      ErrorCode = "bad_request"
	CodeContextLength   ErrorCode = "context_length_exceeded"
	CodeContentFiltered ErrorCode = "content_filtered"
	CodeModelNotFound   ErrorCode = "model_not_found"
	CodeStream          ErrorCode = "stream_error"
	CodeTimeout         ErrorCode = "timeout"
	CodeUnknown         ErrorCode = "unknown"
)

// ProviderError is the wrapped error type provider implementations return.
type ProviderError struct {
	Provider   string
	Code       ErrorCode
	Status     int
	Message    string
	RequestID  string
	RetryAfter time.Duration
	Wrapped    error
}

func (e *ProviderError) Error() string {
	if e == nil {
		return ""
	}
	if e.Status > 0 {
		return fmt.Sprintf("%s: %s [%d]: %s", e.Provider, e.Code, e.Status, e.Message)
	}
	return fmt.Sprintf("%s: %s: %s", e.Provider, e.Code, e.Message)
}

func (e *ProviderError) Unwrap() error { return e.Wrapped }

func (e *ProviderError) Is(target error) bool {
	var t *ProviderError
	if !errors.As(target, &t) {
		return false
	}
	return e.Code == t.Code
}

var (
	ErrRateLimited     = &ProviderError{Code: CodeRateLimited}
	ErrQuotaExhausted  = &ProviderError{Code: CodeQuotaExhausted}
	ErrOverloaded      = &ProviderError{Code: CodeOverloaded}
	ErrServer          = &ProviderError{Code: CodeServer}
	ErrUnauthorized    = &ProviderError{Code: CodeUnauthorized}
	ErrBadRequest      = &ProviderError{Code: CodeBadRequest}
	ErrContextLength   = &ProviderError{Code: CodeContextLength}
	ErrContentFiltered = &ProviderError{Code: CodeContentFiltered}
	ErrModelNotFound   = &ProviderError{Code: CodeModelNotFound}
	ErrStream          = &ProviderError{Code: CodeStream}
	ErrTimeout         = &ProviderError{Code: CodeTimeout}
)

func IsRetryable(err error) bool {
	return errors.Is(err, ErrRateLimited) ||
		errors.Is(err, ErrOverloaded) ||
		errors.Is(err, ErrServer) ||
		errors.Is(err, ErrStream) ||
		errors.Is(err, ErrTimeout)
}

// IsProviderTerminal reports failures that make another request to the same
// provider futile until operator-controlled credentials or billing state
// changes. Request-specific failures such as a bad prompt, context overflow,
// or a missing model are deliberately excluded: another model or request on
// the same provider may still succeed.
func IsProviderTerminal(err error) bool {
	return errors.Is(err, ErrQuotaExhausted) || errors.Is(err, ErrUnauthorized)
}

func ClassifyStatus(status int) ErrorCode {
	switch status {
	case 408, 504:
		return CodeTimeout
	case 429:
		return CodeRateLimited
	case 529:
		return CodeOverloaded
	case 413:
		return CodeContextLength
	case 404:
		return CodeModelNotFound
	case 400, 422:
		return CodeBadRequest
	case 401, 403:
		return CodeUnauthorized
	}
	if status >= 500 && status <= 599 {
		return CodeServer
	}
	return CodeUnknown
}
