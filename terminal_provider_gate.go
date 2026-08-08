package llm

// ARCHITECTURE: provider retries answer whether one request should be tried
// again. TerminalProviderGate answers the wider run-scoped question: after a
// provider has rejected billing or credentials, should any sibling client in
// this run call that provider again? The gate is intentionally supplied by the
// orchestration owner so its lifetime matches one run, while this package owns
// provider error semantics and capability-preserving client wrapping.

import (
	"context"
	"errors"
	"strings"
	"sync"
)

// TerminalProviderGate remembers the first terminal failure per provider and
// prevents sibling clients from repeating a request that cannot succeed. A
// gate may wrap several models and providers; only the failed provider is
// blocked, so an explicitly permitted cross-provider fallback remains usable.
type TerminalProviderGate struct {
	mu       sync.RWMutex
	failures map[string]error
}

// NewTerminalProviderGate creates an empty run-scoped provider gate.
func NewTerminalProviderGate() *TerminalProviderGate {
	return &TerminalProviderGate{failures: map[string]error{}}
}

// Wrap applies the gate to every supported client call shape while preserving
// the wrapped client's richer capabilities through the normal llm invocation
// helpers. A nil client remains nil so existing optional-client wiring keeps
// its meaning.
func (g *TerminalProviderGate) Wrap(next Client) Client {
	if g == nil || next == nil {
		return next
	}
	provider := ""
	if meta, ok := AsProviderMeta(next); ok {
		provider = normalizeProvider(meta.Provider())
	}
	return &terminalProviderGateClient{gate: g, next: next, provider: provider}
}

// Failure returns the first terminal failure observed for provider.
func (g *TerminalProviderGate) Failure(provider string) (error, bool) {
	if g == nil {
		return nil, false
	}
	provider = normalizeProvider(provider)
	if provider == "" {
		return nil, false
	}
	g.mu.RLock()
	err, ok := g.failures[provider]
	g.mu.RUnlock()
	return err, ok
}

func (g *TerminalProviderGate) remember(provider string, err error) {
	provider = normalizeProvider(provider)
	if g == nil || provider == "" || !IsProviderTerminal(err) {
		return
	}
	g.mu.Lock()
	if _, exists := g.failures[provider]; !exists {
		g.failures[provider] = err
	}
	g.mu.Unlock()
}

func normalizeProvider(provider string) string {
	return strings.ToLower(strings.TrimSpace(provider))
}

type terminalProviderGateClient struct {
	gate     *TerminalProviderGate
	next     Client
	provider string
}

func (c *terminalProviderGateClient) before() error {
	if err, blocked := c.gate.Failure(c.provider); blocked {
		return err
	}
	return nil
}

func (c *terminalProviderGateClient) observe(err error) error {
	if err == nil || !IsProviderTerminal(err) {
		return err
	}
	provider := c.provider
	var providerErr *ProviderError
	if errors.As(err, &providerErr) && normalizeProvider(providerErr.Provider) != "" {
		provider = providerErr.Provider
	}
	c.gate.remember(provider, err)
	return err
}

func (c *terminalProviderGateClient) Call(ctx context.Context, prompt string) (string, error) {
	if err := c.before(); err != nil {
		return "", err
	}
	result, err := c.next.Call(ctx, prompt)
	return result, c.observe(err)
}

func (c *terminalProviderGateClient) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error) {
	if err := c.before(); err != nil {
		return "", err
	}
	result, err := CallWithOptions(ctx, c.next, prompt, opts)
	return result, c.observe(err)
}

func (c *terminalProviderGateClient) Stream(ctx context.Context, prompt string, onChunk func(string) error) (string, error) {
	if err := c.before(); err != nil {
		return "", err
	}
	result, err := c.next.Stream(ctx, prompt, onChunk)
	return result, c.observe(err)
}

func (c *terminalProviderGateClient) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(string) error) (string, error) {
	if err := c.before(); err != nil {
		return "", err
	}
	result, err := StreamWithOptions(ctx, c.next, prompt, opts, onChunk)
	return result, c.observe(err)
}

func (c *terminalProviderGateClient) CallCached(ctx context.Context, system, user string) (string, error) {
	if err := c.before(); err != nil {
		return "", err
	}
	result, err := CallWithCaching(ctx, c.next, system, user)
	return result, c.observe(err)
}

func (c *terminalProviderGateClient) CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (string, error) {
	if err := c.before(); err != nil {
		return "", err
	}
	result, err := CallCachedWithOptions(ctx, c.next, system, user, opts)
	return result, c.observe(err)
}

func (c *terminalProviderGateClient) CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (Message, error) {
	return c.CallWithToolsOptions(ctx, system, messages, tools, RequestOptions{})
}

func (c *terminalProviderGateClient) CallWithToolsOptions(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error) {
	if err := c.before(); err != nil {
		return Message{}, err
	}
	result, err := CallWithToolsOptions(ctx, c.next, system, messages, tools, opts)
	return result, c.observe(err)
}

func (c *terminalProviderGateClient) Unwrap() Client { return c.next }
