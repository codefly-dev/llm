package llm

// ARCHITECTURE: client capability detection + invocation helpers.
//
// A llm.Client is the minimal interface; richer capabilities (caching, native
// tool calling, real usage reporting, event streaming, provider self-ID) are
// optional and detected at runtime via type assertions that also unwrap
// middleware. The As* helpers do that detection. Tool invocation is deliberately
// fail-closed: supported providers must implement native, typed tool calling.

import (
	"context"
	"fmt"
)

// ── Capability detection ────────────────────────────────────────

// AsCacheable reports whether c supports prompt caching.
func AsCacheable(c Client) (CacheableClient, bool) {
	cc, ok := c.(CacheableClient)
	return cc, ok
}

// AsToolClient reports whether c supports native tool calling, unwrapping
// middleware to find the capability.
func AsToolClient(c Client) (ToolClient, bool) {
	if tc, ok := c.(ToolClient); ok {
		return tc, true
	}
	if u, ok := c.(interface{ Unwrap() Client }); ok {
		return AsToolClient(u.Unwrap())
	}
	return nil, false
}

// AsUsageProvider reports whether c (or anything it unwraps to) can report
// real API token counts.
func AsUsageProvider(c Client) (UsageProvider, bool) {
	if up, ok := c.(UsageProvider); ok {
		return up, true
	}
	if u, ok := c.(interface{ Unwrap() Client }); ok {
		return AsUsageProvider(u.Unwrap())
	}
	return nil, false
}

// AsEventStreamClient reports whether c supports normalized event streaming.
func AsEventStreamClient(c Client) (EventStreamClient, bool) {
	if esc, ok := c.(EventStreamClient); ok {
		return esc, true
	}
	if u, ok := c.(interface{ Unwrap() Client }); ok {
		return AsEventStreamClient(u.Unwrap())
	}
	return nil, false
}

// AsProviderMeta reports whether c (or anything it unwraps to) exposes
// provider/model self-identification.
func AsProviderMeta(c Client) (ProviderMeta, bool) {
	if m, ok := c.(ProviderMeta); ok {
		return m, true
	}
	if u, ok := c.(interface{ Unwrap() Client }); ok {
		return AsProviderMeta(u.Unwrap())
	}
	return nil, false
}

// ── Invocation helpers ──────────────────────────────────────────

// CallWithCaching uses CallCached when supported, otherwise concatenates
// system + user into a single prompt.
func CallWithCaching(ctx context.Context, c Client, system, user string) (string, error) {
	if c == nil {
		return "", fmt.Errorf("llm: no client configured")
	}
	if cc, ok := AsCacheable(c); ok {
		return cc.CallCached(ctx, system, user)
	}
	return c.Call(ctx, system+"\n\n"+user)
}

func CallWithOptions(ctx context.Context, c Client, prompt string, opts RequestOptions) (string, error) {
	if c == nil {
		return "", fmt.Errorf("llm: no client configured")
	}
	if oc, ok := c.(OptionedClient); ok {
		return oc.CallWithOptions(ctx, prompt, opts)
	}
	return c.Call(ctx, prompt)
}

func StreamWithOptions(ctx context.Context, c Client, prompt string, opts RequestOptions, onChunk func(string) error) (string, error) {
	if c == nil {
		return "", fmt.Errorf("llm: no client configured")
	}
	if oc, ok := c.(OptionedClient); ok {
		return oc.StreamWithOptions(ctx, prompt, opts, onChunk)
	}
	return c.Stream(ctx, prompt, onChunk)
}

func CallCachedWithOptions(ctx context.Context, c Client, system, user string, opts RequestOptions) (string, error) {
	if c == nil {
		return "", fmt.Errorf("llm: no client configured")
	}
	if cc, ok := c.(OptionedCacheableClient); ok {
		return cc.CallCachedWithOptions(ctx, system, user, opts)
	}
	return CallWithCaching(ctx, c, system, user)
}

func CallWithToolsOptions(ctx context.Context, c Client, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error) {
	if c == nil {
		return Message{}, fmt.Errorf("llm: no client configured")
	}
	if tc, ok := c.(OptionedToolClient); ok {
		return tc.CallWithToolsOptions(ctx, system, messages, tools, opts)
	}
	if tc, ok := c.(ToolClient); ok {
		return tc.CallWithTools(ctx, system, messages, tools)
	}
	return Message{}, fmt.Errorf("llm: client %T does not implement native tool calling", c)
}

// ── Event-stream adapter ────────────────────────────────────────

func StreamEventsFromClient(ctx context.Context, c Client, req StreamRequest) (<-chan StreamEvent, error) {
	if c == nil {
		return nil, fmt.Errorf("llm: no client configured")
	}
	if err := ValidateToolDefs(req.Tools); err != nil {
		return nil, fmt.Errorf("llm: invalid tool schema: %w", err)
	}

	out := make(chan StreamEvent, 16)
	go func() {
		defer close(out)

		if len(req.Messages) > 0 || len(req.Tools) > 0 {
			tc, ok := AsToolClient(c)
			if !ok {
				_ = sendEvent(ctx, out, StreamEvent{Type: StreamEventError, Error: "client does not support tool calling", Synthetic: true})
				return
			}
			msg, err := tc.CallWithTools(ctx, req.System, req.Messages, req.Tools)
			if err != nil {
				_ = sendEvent(ctx, out, StreamEvent{Type: StreamEventError, Error: err.Error(), Synthetic: true})
				return
			}
			emitMessageEvents(ctx, out, msg)
			return
		}

		full, err := c.Stream(ctx, req.Prompt, func(text string) error {
			return sendEvent(ctx, out, StreamEvent{Type: StreamEventTextDelta, Text: text})
		})
		if err != nil {
			_ = sendEvent(ctx, out, StreamEvent{Type: StreamEventError, Text: full, Error: err.Error()})
			return
		}
		msg := Message{Role: "assistant", Content: full}
		_ = sendEvent(ctx, out, StreamEvent{Type: StreamEventMessageEnd, Message: &msg, Synthetic: true})
	}()
	return out, nil
}

func emitMessageEvents(ctx context.Context, out chan<- StreamEvent, msg Message) {
	for i := range msg.Reasoning {
		r := msg.Reasoning[i]
		_ = sendEvent(ctx, out, StreamEvent{Type: StreamEventReasoningDelta, Text: r.Text, Reasoning: &r, Synthetic: true})
	}
	if msg.Content != "" {
		_ = sendEvent(ctx, out, StreamEvent{Type: StreamEventTextDelta, Text: msg.Content, Synthetic: true})
	}
	for i := range msg.ToolCalls {
		tc := msg.ToolCalls[i]
		_ = sendEvent(ctx, out, StreamEvent{Type: StreamEventToolCallStart, ToolCall: &tc, Synthetic: true})
		_ = sendEvent(ctx, out, StreamEvent{Type: StreamEventToolCallEnd, ToolCall: &tc, Synthetic: true})
	}
	_ = sendEvent(ctx, out, StreamEvent{Type: StreamEventMessageEnd, Message: &msg, Synthetic: true})
}

func sendEvent(ctx context.Context, out chan<- StreamEvent, event StreamEvent) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case out <- event:
		return nil
	}
}
