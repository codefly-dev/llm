package llm

// ARCHITECTURE: provider error classification.
//
// Each provider's SDK surfaces errors differently (Anthropic returns
// *anthropic.Error and OpenAI returns *openai.Error). Retry and
// circuit-breaker middleware want to make
// transport-agnostic decisions — "is this retryable?" — without knowing
// which SDK raised the error. This file provides one classifier per
// provider and a shared timeout/cancellation fallback.
//
// Each provider's Call/Stream/CallWithTools funnels SDK errors through its
// classifier so the middleware layer sees ProviderError with a stable Code.

import (
	"context"
	"errors"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/codefly-dev/llm/apierror"
	openai "github.com/openai/openai-go"
)

// classifyAnthropic wraps an Anthropic SDK error into a ProviderError with
// the classified Code. Pass-through for nil and already-wrapped errors.
func classifyAnthropic(err error, now time.Time) error {
	if err == nil {
		return nil
	}
	if isWrapped(err) {
		return err
	}
	if code, status, ok := classifyTransport("anthropic", err); ok {
		return wrap("anthropic", code, status, err)
	}
	var apiErr *anthropic.Error
	if errors.As(err, &apiErr) {
		code := refineAnthropicCode(ClassifyStatus(apiErr.StatusCode), apiErr)
		return wrapWithMeta("anthropic", code, apiErr.StatusCode, apiErr.RequestID, retryAfterAt(apiErr.Response, now), err)
	}
	return wrap("anthropic", apierror.CodeUnknown, 0, err)
}

// classifyOpenAI wraps an OpenAI SDK error.
func classifyOpenAI(err error, now time.Time) error {
	if err == nil {
		return nil
	}
	if isWrapped(err) {
		return err
	}
	if code, status, ok := classifyTransport("openai", err); ok {
		return wrap("openai", code, status, err)
	}
	var apiErr *openai.Error
	if errors.As(err, &apiErr) {
		code := refineOpenAICode(ClassifyStatus(apiErr.StatusCode), apiErr)
		return wrapWithMeta("openai", code, apiErr.StatusCode, requestID(apiErr.Response, "x-request-id"), retryAfterAt(apiErr.Response, now), err)
	}
	return wrap("openai", apierror.CodeUnknown, 0, err)
}

// isWrapped reports whether err is already a *ProviderError — avoids double
// wrapping when middleware re-raises an error through a provider path.
func isWrapped(err error) bool {
	var pe *apierror.ProviderError
	return errors.As(err, &pe)
}

// wrap produces the ProviderError value providers return on the wire.
func wrap(provider string, code apierror.ErrorCode, status int, err error) error {
	return wrapWithMeta(provider, code, status, "", 0, err)
}

func wrapWithMeta(provider string, code apierror.ErrorCode, status int, requestID string, retryAfter time.Duration, err error) error {
	return &apierror.ProviderError{
		Provider:   provider,
		Code:       code,
		Status:     status,
		Message:    err.Error(),
		RequestID:  requestID,
		RetryAfter: retryAfter,
		Wrapped:    err,
	}
}

func refineAnthropicCode(code apierror.ErrorCode, err *anthropic.Error) apierror.ErrorCode {
	t := strings.ToLower(string(err.Type()))
	switch {
	case strings.Contains(t, "rate_limit"):
		return apierror.CodeRateLimited
	case strings.Contains(t, "overloaded"):
		return apierror.CodeOverloaded
	case strings.Contains(t, "authentication") || strings.Contains(t, "permission"):
		return apierror.CodeUnauthorized
	case strings.Contains(t, "not_found"):
		return apierror.CodeModelNotFound
	case strings.Contains(t, "too_large") || strings.Contains(t, "context"):
		return apierror.CodeContextLength
	case strings.Contains(t, "refusal") || strings.Contains(t, "filter"):
		return apierror.CodeContentFiltered
	default:
		return code
	}
}

func refineOpenAICode(code apierror.ErrorCode, err *openai.Error) apierror.ErrorCode {
	s := strings.ToLower(err.Code + " " + err.Type + " " + err.Message)
	switch {
	case strings.Contains(s, "insufficient_quota") ||
		strings.Contains(s, "billing_hard_limit_reached") ||
		strings.Contains(s, "billing hard limit") ||
		strings.Contains(s, "exceeded your current quota") ||
		strings.Contains(s, "plan and billing"):
		// OpenAI reports account-level billing exhaustion with HTTP 429, but
		// retrying cannot recover it. Keep it distinct from transient request
		// throttling so operators get the right remediation and retry policy.
		//
		// The machine codes (insufficient_quota, billing_hard_limit_reached)
		// live in Code/Type, which OpenAI-compatible gateways often strip while
		// forwarding the upstream message text. Match the billing-exhaustion
		// phrases too so a proxied quota error is still terminal. These phrases
		// are specific to account/billing exhaustion; transient per-minute
		// throttling ("rate limit reached, try again in Ns") never contains
		// them, so real rate limits still fall through to CodeRateLimited.
		return apierror.CodeQuotaExhausted
	case strings.Contains(s, "context_length") || strings.Contains(s, "maximum context"):
		return apierror.CodeContextLength
	case strings.Contains(s, "content_filter") || strings.Contains(s, "moderation"):
		return apierror.CodeContentFiltered
	case strings.Contains(s, "model_not_found") || strings.Contains(s, "does not exist"):
		return apierror.CodeModelNotFound
	case strings.Contains(s, "rate_limit"):
		return apierror.CodeRateLimited
	default:
		return code
	}
}

func requestID(resp *http.Response, header string) string {
	if resp == nil {
		return ""
	}
	return resp.Header.Get(header)
}

func retryAfterAt(resp *http.Response, now time.Time) time.Duration {
	if resp == nil {
		return 0
	}
	if ms := resp.Header.Get("Retry-After-Ms"); ms != "" {
		if n, err := strconv.Atoi(ms); err == nil && n > 0 {
			return time.Duration(n) * time.Millisecond
		}
	}
	if sec := resp.Header.Get("Retry-After"); sec != "" {
		if n, err := strconv.ParseFloat(sec, 64); err == nil && n > 0 {
			return time.Duration(n * float64(time.Second))
		}
		if t, err := http.ParseTime(sec); err == nil {
			if d := t.Sub(now); d > 0 {
				return d
			}
		}
	}
	return 0
}

// classifyTransport covers errors that happen below the HTTP layer — ctx
// cancel, DNS failure, TCP reset, read/write timeout — before the SDK ever
// parsed an HTTP status. Returns (code, 0, true) when matched.
func classifyTransport(_ string, err error) (apierror.ErrorCode, int, bool) {
	if errors.Is(err, context.DeadlineExceeded) {
		return apierror.CodeTimeout, 0, true
	}
	var netErr net.Error
	if errors.As(err, &netErr) && netErr.Timeout() {
		return apierror.CodeTimeout, 0, true
	}
	var urlErr *url.Error
	if errors.As(err, &urlErr) && urlErr.Timeout() {
		return apierror.CodeTimeout, 0, true
	}
	return "", 0, false
}
