package llm

import (
	"context"
	"errors"
	"testing"
)

func TestTerminalProviderGateStopsSiblingCallsToFailedProvider(t *testing.T) {
	gate := NewTerminalProviderGate()
	quota := &terminalFailureInjectionClient{
		provider: "openai",
		model:    "gpt-primary",
		err: &ProviderError{
			Provider: "openai",
			Code:     CodeQuotaExhausted,
			Status:   429,
			Message:  "injected exhausted quota",
		},
	}
	sibling := &terminalFailureInjectionClient{provider: "openai", model: "gpt-cheap", response: "must not run"}
	fallback := &terminalFailureInjectionClient{provider: "anthropic", model: "claude", response: "fallback-ok"}

	if _, err := gate.Wrap(quota).Call(context.Background(), "first"); !errors.Is(err, ErrQuotaExhausted) {
		t.Fatalf("first call error = %v, want quota exhausted", err)
	}
	if _, err := gate.Wrap(sibling).Call(context.Background(), "second"); !errors.Is(err, ErrQuotaExhausted) {
		t.Fatalf("sibling call error = %v, want latched quota exhausted", err)
	}
	if sibling.calls != 0 {
		t.Fatalf("same-provider sibling reached transport %d times, want 0", sibling.calls)
	}
	got, err := gate.Wrap(fallback).Call(context.Background(), "fallback")
	if err != nil || got != "fallback-ok" {
		t.Fatalf("cross-provider fallback = %q, %v", got, err)
	}
	if fallback.calls != 1 {
		t.Fatalf("cross-provider fallback calls = %d, want 1", fallback.calls)
	}
}

func TestTerminalProviderGateDoesNotLatchTransientFailure(t *testing.T) {
	gate := NewTerminalProviderGate()
	transient := &terminalFailureInjectionClient{
		provider: "openai",
		model:    "gpt-primary",
		err:      &ProviderError{Provider: "openai", Code: CodeRateLimited, Status: 429, Message: "injected transient limit"},
	}
	sibling := &terminalFailureInjectionClient{provider: "openai", model: "gpt-cheap", response: "recovered"}

	if _, err := gate.Wrap(transient).Call(context.Background(), "first"); !errors.Is(err, ErrRateLimited) {
		t.Fatalf("first call error = %v, want rate limited", err)
	}
	got, err := gate.Wrap(sibling).Call(context.Background(), "second")
	if err != nil || got != "recovered" {
		t.Fatalf("sibling after transient failure = %q, %v", got, err)
	}
	if sibling.calls != 1 {
		t.Fatalf("sibling calls = %d, want 1", sibling.calls)
	}
}

func TestTerminalProviderGateCoversNativeToolCalls(t *testing.T) {
	gate := NewTerminalProviderGate()
	failing := &terminalFailureInjectionClient{
		provider: "openai",
		model:    "gpt-primary",
		err:      &ProviderError{Provider: "openai", Code: CodeUnauthorized, Status: 401, Message: "injected invalid credential"},
	}
	sibling := &terminalFailureInjectionClient{provider: "openai", model: "gpt-tools"}

	first, ok := AsToolClient(gate.Wrap(failing))
	if !ok {
		t.Fatal("wrapped client lost native tool capability")
	}
	if _, err := first.CallWithTools(context.Background(), "system", nil, nil); !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("first tool call error = %v, want unauthorized", err)
	}
	second, ok := AsToolClient(gate.Wrap(sibling))
	if !ok {
		t.Fatal("wrapped sibling lost native tool capability")
	}
	if _, err := second.CallWithTools(context.Background(), "system", nil, nil); !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("sibling tool call error = %v, want latched unauthorized", err)
	}
	if sibling.calls != 0 {
		t.Fatalf("same-provider sibling tool transport calls = %d, want 0", sibling.calls)
	}
}

// terminalFailureInjectionClient is the sanctioned failure-injection boundary
// for provider states that cannot be produced deterministically in tests.
type terminalFailureInjectionClient struct {
	provider string
	model    string
	response string
	err      error
	calls    int
}

func (c *terminalFailureInjectionClient) Provider() string { return c.provider }
func (c *terminalFailureInjectionClient) Model() string    { return c.model }

func (c *terminalFailureInjectionClient) Call(context.Context, string) (string, error) {
	c.calls++
	return c.response, c.err
}

func (c *terminalFailureInjectionClient) Stream(context.Context, string, func(string) error) (string, error) {
	c.calls++
	return c.response, c.err
}

func (c *terminalFailureInjectionClient) CallWithTools(context.Context, string, []Message, []ToolDef) (Message, error) {
	c.calls++
	return Message{Role: "assistant", Content: c.response}, c.err
}
