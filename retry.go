package llm

import (
	"context"
	"fmt"
	"time"

	"github.com/codefly-dev/llm/apierror"
	"github.com/codefly-dev/llm/retry"
)

// RetryMiddleware returns a middleware that retries on transient errors.
// It uses exponential backoff with jitter. Only retries on 5xx, 429, and
// timeout/connection errors. Does NOT retry on 4xx (except 429).
//
// CALLER-CTX DISCIPLINE: a dead caller context is NEVER retried. Per-attempt
// timeouts classify as CodeTimeout (retryable — the next attempt gets a fresh
// budget), and an expired CALLER deadline classifies the same way; without an
// explicit guard the loop would burn attempts against a context that can no
// longer succeed (a real trace showed 4 attempts fired after the caller's
// deadline had already passed). Every loop below therefore checks ctx.Err()
// BEFORE each attempt and before/after each backoff sleep, returning the ctx
// error (wrapped, with the last provider error preserved) immediately.
func RetryMiddleware(maxRetries int, baseBackoff time.Duration) Middleware {
	if maxRetries <= 0 {
		maxRetries = 3
	}
	if baseBackoff <= 0 {
		baseBackoff = time.Second
	}
	return func(next Client) Client {
		return &retryMW{next: next, maxRetries: maxRetries, baseBackoff: baseBackoff}
	}
}

type retryMW struct {
	next        Client
	maxRetries  int
	baseBackoff time.Duration
}

func (m *retryMW) Call(ctx context.Context, prompt string) (string, error) {
	return m.CallWithOptions(ctx, prompt, RequestOptions{})
}

func (m *retryMW) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error) {
	var lastErr error
	for attempt := 0; attempt <= m.maxRetries; attempt++ {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return "", retryAbortedByContext(ctxErr, lastErr)
		}
		if attempt > 0 {
			if err := m.sleep(ctx, attempt, lastErr); err != nil {
				return "", retryAbortedByContext(err, lastErr)
			}
		}
		out, err := CallWithOptions(ctx, m.next, prompt, opts)
		if err == nil {
			return out, nil
		}
		if !isRetryable(err) {
			return out, err
		}
		lastErr = err
	}
	return "", lastErr
}

func (m *retryMW) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (string, error) {
	return m.StreamWithOptions(ctx, prompt, RequestOptions{}, onChunk)
}

func (m *retryMW) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(text string) error) (string, error) {
	var lastErr error
	for attempt := 0; attempt <= m.maxRetries; attempt++ {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return "", retryAbortedByContext(ctxErr, lastErr)
		}
		if attempt > 0 {
			if err := m.sleep(ctx, attempt, lastErr); err != nil {
				return "", retryAbortedByContext(err, lastErr)
			}
		}
		madeProgress := false
		wrapped := onChunk
		if onChunk != nil {
			wrapped = func(text string) error {
				if text != "" {
					madeProgress = true
				}
				return onChunk(text)
			}
		}
		out, err := StreamWithOptions(ctx, m.next, prompt, opts, wrapped)
		if out != "" {
			madeProgress = true
		}
		if err == nil {
			return out, nil
		}
		if madeProgress || !isRetryable(err) {
			return out, err
		}
		lastErr = err
	}
	return "", lastErr
}

func (m *retryMW) CallCached(ctx context.Context, system, user string) (string, error) {
	return m.CallCachedWithOptions(ctx, system, user, RequestOptions{})
}

func (m *retryMW) CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (string, error) {
	var lastErr error
	for attempt := 0; attempt <= m.maxRetries; attempt++ {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return "", retryAbortedByContext(ctxErr, lastErr)
		}
		if attempt > 0 {
			if err := m.sleep(ctx, attempt, lastErr); err != nil {
				return "", retryAbortedByContext(err, lastErr)
			}
		}
		out, err := CallCachedWithOptions(ctx, m.next, system, user, opts)
		if err == nil {
			return out, nil
		}
		if !isRetryable(err) {
			return out, err
		}
		lastErr = err
	}
	return "", lastErr
}

func (m *retryMW) CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (Message, error) {
	return m.CallWithToolsOptions(ctx, system, messages, tools, RequestOptions{})
}

func (m *retryMW) CallWithToolsOptions(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error) {
	var lastErr error
	for attempt := 0; attempt <= m.maxRetries; attempt++ {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return Message{}, retryAbortedByContext(ctxErr, lastErr)
		}
		if attempt > 0 {
			if err := m.sleep(ctx, attempt, lastErr); err != nil {
				return Message{}, retryAbortedByContext(err, lastErr)
			}
		}
		out, err := CallWithToolsOptions(ctx, m.next, system, messages, tools, opts)
		if err == nil {
			return out, nil
		}
		if !isRetryable(err) {
			return out, err
		}
		lastErr = err
	}
	return Message{}, lastErr
}

func (m *retryMW) Unwrap() Client { return m.next }

func (m *retryMW) sleep(ctx context.Context, attempt int, lastErr error) error {
	return retry.Sleep(ctx, attempt, m.baseBackoff, lastErr)
}

// retryAbortedByContext is the typed return for "the caller's context died
// before/during retry". It wraps the ctx error (errors.Is against
// context.Canceled / context.DeadlineExceeded works) and preserves the last
// provider error as text so the failure trace shows WHAT was being retried.
func retryAbortedByContext(ctxErr, lastErr error) error {
	if lastErr != nil {
		return fmt.Errorf("llm retry aborted: caller context done: %w (last provider error: %v)", ctxErr, lastErr)
	}
	return fmt.Errorf("llm retry aborted: caller context done before first attempt: %w", ctxErr)
}

// isRetryable reports whether err warrants a retry with backoff.
//
// Preferred path: typed ProviderError sentinels (see pkg/apierror).
// Built-in providers wrap SDK errors into ProviderError on the way out, so
// [errors.Is] against llm.ErrRateLimited / llm.ErrOverloaded / llm.ErrServer
// / llm.ErrTimeout matches without touching error strings.
//
// Custom providers must return a typed ProviderError (or a typed transport
// timeout). Arbitrary error text is never parsed into control flow.
func isRetryable(err error) bool {
	if err == nil {
		return false
	}
	if IsRetryable(err) {
		return true
	}
	if code, _, ok := classifyTransport("", err); ok {
		return code == apierror.CodeTimeout || code == apierror.CodeServer
	}
	return false
}
