package llm

// ARCHITECTURE: Anthropic provider backed by the official anthropic-sdk-go.
// Implements Client, CacheableClient (prompt caching), ToolClient (native
// tool use), UsageProvider (real token counts incl. cache read/create),
// and ProviderMeta.
//
// The SDK handles SSE parsing, request marshaling, retries on the HTTP
// layer, and tool-block serialization — things the prior hand-rolled
// implementation did by hand in ~600 LOC. We keep only the adapter logic
// between llm spec types and the SDK's types.

import (
	"context"
	"encoding/json"
	"fmt"
	"sync/atomic"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/benbjohnson/clock"
)

// Anthropic is the Mind-facing client for Claude. Wraps the SDK client,
// adds Mind's token/cost accounting, and exposes the spec interfaces.
type Anthropic struct {
	sdk   anthropic.Client
	model string
	clock clock.Clock

	// timeouts holds the per-attempt discipline knobs (see timeouts.go).
	// The SDK client itself carries NO request timeout and streams have no
	// native inter-event bound, so without these a wedged provider call can
	// outlive any task deadline the CALLER forgot to plumb. Zero value =
	// package defaults; configure via NewAnthropicWithTimeouts / SetTimeouts
	// or the WithTimeouts ClientOption.
	timeouts TimeoutConfig

	// lastUsage is stored atomically — multiple sessions share one Anthropic instance.
	lastUsage atomic.Pointer[Usage]
}

// NewAnthropic returns an Anthropic client for the given API key and model id,
// with the default per-attempt timeout + stream stall watchdog (timeouts.go).
// Empty key is rejected — use the cassette layer for offline tests.
func NewAnthropic(apiKey, model string) *Anthropic {
	return NewAnthropicWithTimeouts(apiKey, model, TimeoutConfig{})
}

// NewAnthropicWithTimeouts is NewAnthropic with explicit per-attempt
// timeout / stall watchdog knobs. Zero fields select defaults; negative
// fields disable that brake (see TimeoutConfig).
func NewAnthropicWithTimeouts(apiKey, model string, timeouts TimeoutConfig) *Anthropic {
	sdk := anthropic.NewClient(option.WithAPIKey(apiKey))
	return &Anthropic{sdk: sdk, model: model, clock: clock.New(), timeouts: timeouts}
}

// SetTimeouts implements TimeoutConfigurable so NewClient's WithTimeouts
// option can tune the knobs after the provider factory built the client.
// Call before the client is shared across goroutines (construction time).
func (c *Anthropic) SetTimeouts(cfg TimeoutConfig) { c.timeouts = cfg }

func (c *Anthropic) SetRuntimeClock(runtimeClock clock.Clock) {
	if runtimeClock != nil {
		c.clock = runtimeClock
	}
}

func (c *Anthropic) classify(err error) error {
	return classifyAnthropic(err, c.clock.Now())
}

func (c *Anthropic) Provider() string { return "anthropic" }
func (c *Anthropic) Model() string    { return c.model }

// LastCallUsage returns the token counts from the most recent API call.
// Returns nil before any call. Values include cache-read tokens where the
// API reported them, so the cost middleware can bill fresh input at list
// rate and cached input at the cache-read discount.
func (c *Anthropic) LastCallUsage() *Usage {
	u := c.lastUsage.Load()
	if u == nil {
		return nil
	}
	cp := *u
	return &cp
}

// ── Core entry points ───────────────────────────────────────────

// Call sends a single user prompt and returns the full assistant response.
func (c *Anthropic) Call(ctx context.Context, prompt string) (string, error) {
	return c.CallWithOptions(ctx, prompt, RequestOptions{})
}

func (c *Anthropic) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error) {
	return c.StreamWithOptions(ctx, prompt, opts, nil)
}

// Stream sends a prompt and invokes onChunk for each text delta as it
// arrives. Returns the concatenated text. onChunk may be nil.
func (c *Anthropic) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (string, error) {
	return c.StreamWithOptions(ctx, prompt, RequestOptions{}, onChunk)
}

func (c *Anthropic) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(text string) error) (string, error) {
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(c.model),
		MaxTokens: defaultMaxTokens(opts, 8192),
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
		},
	}
	applyAnthropicOptions(&params, opts)
	return c.runStream(ctx, params, onChunk)
}

func (c *Anthropic) StreamEvents(ctx context.Context, req StreamRequest) (<-chan StreamEvent, error) {
	return StreamEventsFromClient(ctx, c, req)
}

// CallCached sends a system prompt marked for caching and a fresh user
// prompt. Calls with the same system prefix hit Anthropic's prompt cache.
func (c *Anthropic) CallCached(ctx context.Context, system, user string) (string, error) {
	return c.CallCachedWithOptions(ctx, system, user, RequestOptions{})
}

func (c *Anthropic) CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (string, error) {
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(c.model),
		MaxTokens: defaultMaxTokens(opts, 8192),
		System: []anthropic.TextBlockParam{{
			Text:         system,
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		}},
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(user)),
		},
	}
	applyAnthropicOptions(&params, opts)
	return c.runStream(ctx, params, nil)
}

// CallWithTools runs one turn of a multi-turn tool conversation. Non-streaming
// because tool-use parsing is cleaner on a fully-formed message; callers that
// need streaming tool deltas should extend this later.
func (c *Anthropic) CallWithTools(
	ctx context.Context,
	system string,
	messages []Message,
	tools []ToolDef,
) (Message, error) {
	return c.CallWithToolsOptions(ctx, system, messages, tools, RequestOptions{})
}

func (c *Anthropic) CallWithToolsOptions(
	ctx context.Context,
	system string,
	messages []Message,
	tools []ToolDef,
	opts RequestOptions,
) (Message, error) {
	if err := ValidateToolDefs(tools); err != nil {
		return Message{}, fmt.Errorf("anthropic: invalid tool schema: %w", err)
	}
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(c.model),
		MaxTokens: defaultMaxTokens(opts, 8192),
		Messages:  toAnthropicMessages(messages),
	}
	if system != "" {
		params.System = []anthropic.TextBlockParam{{
			Text:         system,
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		}}
	}
	if len(tools) > 0 {
		params.Tools = toAnthropicTools(tools)
	}
	// applyAnthropicOptions runs AFTER the custom function tools are set so
	// the hosted web-search server tool (WebSearchMaxUses) is APPENDED, not
	// clobbered. Order: [custom function tools…, hosted web_search].
	applyAnthropicOptions(&params, opts)

	// Per-attempt discipline (timeouts.go): bound this attempt with a child
	// context so a wedged request can never outlive the attempt budget. The
	// caller's own deadline still wins when it is sooner. Deadline
	// observability: warn when the attempt starts with almost no headroom.
	timeouts := c.timeouts.withDefaults()
	warnNearDeadline(ctx, "anthropic", c.model, timeouts, c.clock)
	attemptCtx, cancelAttempt := timeouts.attemptContext(ctx)
	defer cancelAttempt()

	resp, err := c.sdk.Messages.New(attemptCtx, params)
	if err != nil {
		// Attribute the failure precisely: if OUR per-attempt budget fired
		// (caller ctx still alive) say so — classifyTransport maps the
		// wrapped DeadlineExceeded to CodeTimeout, which is retryable, so the
		// retry middleware re-attempts with a fresh budget.
		if ctx.Err() == nil && attemptCtx.Err() != nil {
			return Message{}, c.classify(fmt.Errorf("anthropic: per-attempt timeout %s exceeded: %w", timeouts.PerAttempt, err))
		}
		return Message{}, c.classify(err)
	}

	c.storeUsage(resp.Usage)
	return fromAnthropicMessage(resp), nil
}

func applyAnthropicOptions(params *anthropic.MessageNewParams, opts RequestOptions) {
	if opts.Temperature != nil {
		params.Temperature = param.NewOpt(*opts.Temperature)
	}
	if opts.TopP != nil {
		params.TopP = param.NewOpt(*opts.TopP)
	}
	switch PriceLevel(opts.PriceLevel) {
	case PriceLevelStandard:
		params.ServiceTier = anthropic.MessageNewParamsServiceTierStandardOnly
	}
	// HOSTED web search: attach Anthropic's web_search_20250305 server tool
	// so the model can search the web server-side at Anthropic. The response
	// carries web_search_tool_result blocks (sources) + a synthesized answer;
	// usage.ServerToolUse.WebSearchRequests is billed on the LLM key and
	// already flows through storeUsage → the cost middleware. This is appended
	// alongside any custom function tools (toAnthropicTools) the caller sets.
	if opts.WebSearchMaxUses > 0 {
		params.Tools = append(params.Tools, anthropicWebSearchTool(opts.WebSearchMaxUses))
	}
	// REASONING EFFORT → budgeted extended thinking. Extended-thinking
	// models (currently Haiku 4.5) accept a thinking token budget on the
	// stable API. Adaptive models (Sonnet 5 and Opus 4.8) do
	// not — they run at their server-default adaptive effort,
	// so we leave Thinking unset for them (the difficulty→effort signal is
	// still recorded on the selection for observability). The `effort`
	// param for those models is beta-only; migrating the client to the
	// beta endpoint is tracked separately.
	if budget := anthropicThinkingBudget(string(params.Model), opts.ReasoningEffort); budget > 0 {
		// Extended thinking requires max_tokens > budget_tokens and the
		// default temperature (custom temperature/top_p are rejected).
		if params.MaxTokens <= int64(budget) {
			params.MaxTokens = int64(budget) + 8192
		}
		params.Temperature = param.Opt[float64]{}
		params.TopP = param.Opt[float64]{}
		params.Thinking = anthropic.ThinkingConfigParamOfEnabled(int64(budget))
	}
}

// anthropicThinkingBudget maps a reasoning effort onto an extended-thinking
// token budget for the given (bare) Anthropic model id. Returns 0 — meaning
// "do not set budgeted thinking" — for ReasoningNone and for effort/adaptive
// models that do not accept thinking.budget_tokens on the stable API.
func anthropicThinkingBudget(model string, effort ReasoningEffort) int {
	if effort == ReasoningNone || !anthropicSupportsBudgetedThinking(model) {
		return 0
	}
	switch effort {
	case ReasoningLow:
		return 2048
	case ReasoningMedium:
		return 6144
	case ReasoningHigh:
		return 12288
	default:
		return 0
	}
}

// anthropicSupportsBudgetedThinking reports whether a current supported
// model accepts the stable API's thinking.budget_tokens. This is deliberately
// an allow-list: sending a capability to an unknown or newly released model
// is an API correctness bug, while omitting an optional capability is safe.
func anthropicSupportsBudgetedThinking(model string) bool {
	return model == AnthropicModelHaiku
}

// anthropicWebSearchTool builds the SDK's hosted web-search server tool
// param (web_search_20250305), capped at maxUses searches per request.
func anthropicWebSearchTool(maxUses int) anthropic.ToolUnionParam {
	return anthropic.ToolUnionParam{OfWebSearchTool20250305: &anthropic.WebSearchTool20250305Param{
		MaxUses: param.NewOpt(int64(maxUses)),
	}}
}

// ── Stream runner ──────────────────────────────────────────────

// runStream drives Messages.NewStreaming, accumulates the full response,
// fires onChunk for each text delta, and records usage.
//
// CONTEXT DISCIPLINE (timeouts.go): the stream runs under a per-attempt
// child context of the caller's ctx PLUS an inter-event stall watchdog.
// The SDK attaches the context to the underlying HTTP request
// (http.NewRequestWithContext), so cancelling it aborts the SSE body read
// promptly — there is no detached/background context anywhere on this path;
// the ctx passed in here is the one that bounds the wire. A trace review
// showed a stream running 39 minutes with no error when the CALLER's ctx
// carried no deadline — the attempt budget + watchdog make that impossible
// regardless of what the caller plumbed.
func (c *Anthropic) runStream(
	ctx context.Context,
	params anthropic.MessageNewParams,
	onChunk func(text string) error,
) (string, error) {
	// Per-attempt budget + inter-event stall watchdog (timeouts.go). The
	// watchdog observes MODEL events only — the SDK swallows keepalive `ping`
	// events inside stream.Next() — so its window must stay well above the
	// provider's normal inter-delta gap.
	guard := newStreamGuard(ctx, c.timeouts, "anthropic", c.model, c.clock)
	defer guard.close()

	stream := c.sdk.Messages.NewStreaming(guard.ctx(), params)
	defer func() { _ = stream.Close() }()

	var acc anthropic.Message
	var full []byte

	for stream.Next() {
		guard.reset()
		event := stream.Current()
		if err := acc.Accumulate(event); err != nil {
			return string(full), fmt.Errorf("anthropic: accumulate: %w", err)
		}
		// Extract text deltas from content_block_delta events and fan them
		// out to onChunk. Other event types (usage, tool_use deltas, stop)
		// are tracked inside Accumulate.
		if event.Type == "content_block_delta" && event.Delta.Type == "text_delta" && event.Delta.Text != "" {
			full = append(full, event.Delta.Text...)
			if onChunk != nil {
				if err := onChunk(event.Delta.Text); err != nil {
					return string(full), err
				}
			}
		}
	}
	if err := stream.Err(); err != nil {
		return string(full), guard.attribute(err, c.classify)
	}
	// Defensive ctx discipline: a stream must never report clean success when
	// the caller's ctx is already dead (e.g. the server closed the stream in
	// the same instant the deadline fired) — callers rely on the error to
	// stop looping.
	if ctxErr := guard.callerExpired(c.classify); ctxErr != nil {
		return string(full), ctxErr
	}

	c.storeUsage(acc.Usage)

	// If there were no text deltas (rare — pure tool-use response), fall
	// back to the accumulated text blocks so Call/Stream still return
	// meaningful text.
	if len(full) == 0 {
		for _, block := range acc.Content {
			if block.Type == "text" && block.Text != "" {
				full = append(full, block.Text...)
			}
		}
	}
	return string(full), nil
}

// storeUsage records the real token counts from the API so the cost
// middleware can bill precisely.
func (c *Anthropic) storeUsage(u anthropic.Usage) {
	c.lastUsage.Store(&Usage{
		Provider:              "anthropic",
		Model:                 c.model,
		PriceLevel:            string(u.ServiceTier),
		InputTokens:           int(u.InputTokens),
		OutputTokens:          int(u.OutputTokens),
		CachedTokens:          int(u.CacheReadInputTokens),
		CachedInputTokens:     int(u.CacheReadInputTokens),
		CacheReadInputTokens:  int(u.CacheReadInputTokens),
		CacheWriteInputTokens: int(u.CacheCreationInputTokens),
		HostedToolCalls:       int(u.ServerToolUse.WebSearchRequests + u.ServerToolUse.WebFetchRequests),
		WebSearchRequests:     int(u.ServerToolUse.WebSearchRequests),
		WebFetchRequests:      int(u.ServerToolUse.WebFetchRequests),
	})
}

// ── Conversion helpers ──────────────────────────────────────────

// toAnthropicMessages converts Mind's wire messages into SDK
// MessageParam values. Tool calls on assistant turns become tool_use
// blocks; tool_result messages become user-role tool_result blocks
// (Anthropic's convention).
//
// CACHING: when there are 2+ messages, the FINAL message's last
// content block gets a cache_control: ephemeral marker. Anthropic
// caches everything up to and including the marker, so a subsequent
// supervisor turn that extends the conversation with a new tool
// result + assistant reply reads the entire prior history at 10%
// rate. Combined with the system block (CallWithToolsOptions) and
// the last tool def (toAnthropicTools), that's 3 of Anthropic's 4
// allowed cache breakpoints — every static layer is cached.
//
// Like the tool-cache marker, this is an SDK-layer change with no
// impact on the recorder hash (computed from Mind's pre-translation
// Message slice).
func toAnthropicMessages(msgs []Message) []anthropic.MessageParam {
	out := make([]anthropic.MessageParam, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case "user":
			out = append(out, anthropic.NewUserMessage(anthropic.NewTextBlock(m.Content)))

		case "assistant":
			var blocks []anthropic.ContentBlockParamUnion
			for _, r := range m.Reasoning {
				switch r.Type {
				case "thinking":
					if r.Text != "" && r.Signature != "" {
						blocks = append(blocks, anthropic.NewThinkingBlock(r.Signature, r.Text))
					}
				case "redacted_thinking":
					if r.Data != "" {
						blocks = append(blocks, anthropic.NewRedactedThinkingBlock(r.Data))
					}
				}
			}
			if m.Content != "" {
				blocks = append(blocks, anthropic.NewTextBlock(m.Content))
			}
			for _, tc := range m.ToolCalls {
				blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, tc.Args, tc.Name))
			}
			if len(blocks) > 0 {
				out = append(out, anthropic.NewAssistantMessage(blocks...))
			}

		case "tool_result":
			isErr := m.ToolResult != nil && m.ToolResult.IsError
			out = append(out, anthropic.NewUserMessage(
				anthropic.NewToolResultBlock(m.ToolCallID, m.Content, isErr),
			))
		}
	}
	if len(out) >= 2 {
		applyMessageCacheBreakpoint(&out[len(out)-1])
	}
	return out
}

// applyMessageCacheBreakpoint marks the last cache-eligible content
// block of the final message with cache_control: ephemeral.
func applyMessageCacheBreakpoint(m *anthropic.MessageParam) {
	if m == nil || len(m.Content) == 0 {
		return
	}
	blocks := m.Content
	last := &blocks[len(blocks)-1]
	// Anthropic supports cache_control on text / tool_use /
	// tool_result blocks but not on thinking blocks. If the last
	// block is a thinking variant, walk backward to find the most
	// recent cache-eligible block.
	for i := len(blocks) - 1; i >= 0; i-- {
		b := &blocks[i]
		switch {
		case b.OfText != nil:
			b.OfText.CacheControl = anthropic.NewCacheControlEphemeralParam()
			return
		case b.OfToolUse != nil:
			b.OfToolUse.CacheControl = anthropic.NewCacheControlEphemeralParam()
			return
		case b.OfToolResult != nil:
			b.OfToolResult.CacheControl = anthropic.NewCacheControlEphemeralParam()
			return
		}
	}
	_ = last
}

// toAnthropicTools converts Mind's ToolDef list into SDK
// ToolUnionParam values. The SDK's union is wide (code-exec,
// memory, etc.); we only use the plain ToolParam variant.
//
// CACHING: the LAST tool gets a `cache_control: ephemeral` marker.
// Anthropic caches everything UP TO AND INCLUDING the marker, so
// marking the last tool extends the cached prefix from "just the
// system block" (set in CallWithToolsOptions) to "system + entire
// tool catalog". For a typical run (~10–25 tools × ~200 tokens of
// schema each) this is several thousand tokens of stable prefix
// hit on every turn after the first within the 5-minute TTL.
//
// Cache markers are an SDK-layer concern, NOT a Mind-internal
// ToolDef field — adding them here doesn't change the recorder
// hash (which serializes Mind's ToolDef before SDK translation),
// so cassettes stay valid.
func toAnthropicTools(tools []ToolDef) []anthropic.ToolUnionParam {
	out := make([]anthropic.ToolUnionParam, 0, len(tools))
	for i, t := range tools {
		schema := toolInputSchema(t)
		tp := anthropic.ToolParam{
			Name:        t.Name,
			Description: param.NewOpt(t.Description),
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties:  schemaProperties(schema),
				Required:    schemaRequired(schema),
				ExtraFields: schemaExtraFields(schema, "properties", "required"),
			},
		}
		if t.StrictInput {
			// BOUNDARY: strictness is part of Mind's typed tool contract. Let
			// Anthropic enforce the JSON Schema while generating arguments so
			// malformed completion artifacts never enter orchestration state.
			tp.Strict = param.NewOpt(true)
		}
		// Mark the LAST tool with ephemeral cache_control so the
		// entire tool catalog becomes part of the cached prefix
		// alongside the system block.
		if i == len(tools)-1 {
			tp.CacheControl = anthropic.NewCacheControlEphemeralParam()
		}
		out = append(out, anthropic.ToolUnionParam{OfTool: &tp})
	}
	return out
}

// fromAnthropicMessage converts an SDK Message (the provider's response)
// back into Mind's wire Message. Text blocks concatenate into Content; each
// tool_use block becomes a ToolCall.
func fromAnthropicMessage(resp *anthropic.Message) Message {
	msg := Message{Role: "assistant"}
	for _, block := range resp.Content {
		switch block.Type {
		case "thinking":
			msg.Reasoning = append(msg.Reasoning, ReasoningBlock{
				Provider:  "anthropic",
				Type:      "thinking",
				Text:      block.Thinking,
				Signature: block.Signature,
			})
		case "redacted_thinking":
			msg.Reasoning = append(msg.Reasoning, ReasoningBlock{
				Provider: "anthropic",
				Type:     "redacted_thinking",
				Data:     block.Data,
			})
		case "text":
			msg.Content += block.Text
		case "tool_use":
			args, _ := anthropicToolInputToMap(block.Input)
			msg.ToolCalls = append(msg.ToolCalls, ToolCall{
				ID:   block.ID,
				Name: block.Name,
				Args: args,
			})
		}
	}
	return msg
}

// anthropicToolInputToMap converts the SDK's opaque tool input (a
// json.RawMessage) into the map[string]any shape Mind uses on the wire.
func anthropicToolInputToMap(input json.RawMessage) (map[string]any, error) {
	if len(input) == 0 {
		return map[string]any{}, nil
	}
	var out map[string]any
	if err := json.Unmarshal(input, &out); err != nil {
		return nil, err
	}
	return out, nil
}

// Compile-time contract assertions.
var (
	_ Client                  = (*Anthropic)(nil)
	_ OptionedClient          = (*Anthropic)(nil)
	_ CacheableClient         = (*Anthropic)(nil)
	_ OptionedCacheableClient = (*Anthropic)(nil)
	_ ToolClient              = (*Anthropic)(nil)
	_ OptionedToolClient      = (*Anthropic)(nil)
	_ EventStreamClient       = (*Anthropic)(nil)
	_ UsageProvider           = (*Anthropic)(nil)
	_ ProviderMeta            = (*Anthropic)(nil)
	_ HostedWebSearcher       = (*Anthropic)(nil)
	_ TimeoutConfigurable     = (*Anthropic)(nil)
)
