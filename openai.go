package llm

// ARCHITECTURE: OpenAI provider backed by the official openai-go SDK.
// Implements Client (completion + streaming), ToolClient (native function
// calling through the current Responses API), UsageProvider (real token counts incl. cached
// prompt tokens), and ProviderMeta.
//
// Unlike Anthropic, OpenAI has no explicit `CacheControl` marker in the
// request — prompt caching is automatic on the server side based on token
// count. There is no CacheableClient interface on OpenAI. The middleware
// path falls through to plain Call via CallWithCaching.

import (
	"context"
	"errors"
	"fmt"
	"sync/atomic"
	"time"

	"github.com/benbjohnson/clock"
	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/shared"
)

// OpenAI is the Mind-facing client for GPT models.
type OpenAI struct {
	sdk   openai.Client
	model string
	clock clock.Clock

	// timeouts applies the same provider-attempt discipline used by
	// Anthropic. The OpenAI SDK has no end-to-end request deadline of its
	// own, so every live completion and Responses API turn must carry a
	// bounded child context. Zero values select the package defaults.
	timeouts TimeoutConfig

	lastUsage atomic.Pointer[Usage]
}

// NewOpenAI returns an OpenAI client for the given API key and model id.
func NewOpenAI(apiKey, model string) *OpenAI {
	sdk := openai.NewClient(option.WithAPIKey(apiKey))
	return &OpenAI{sdk: sdk, model: model, clock: clock.New()}
}

// SetTimeouts implements TimeoutConfigurable. Call it during client
// construction, before the provider is shared across sessions.
func (c *OpenAI) SetTimeouts(cfg TimeoutConfig) { c.timeouts = cfg }

func (c *OpenAI) SetRuntimeClock(runtimeClock clock.Clock) {
	if runtimeClock != nil {
		c.clock = runtimeClock
	}
}

func (c *OpenAI) classify(err error) error {
	return classifyOpenAI(err, c.clock.Now())
}

func (c *OpenAI) Provider() string { return "openai" }

func (c *OpenAI) Model() string { return c.model }

// LastCallUsage returns token counts from the most recent call, including
// the cached-prompt subset OpenAI reports via prompt_tokens_details.
// Returns nil before any call.
func (c *OpenAI) LastCallUsage() *Usage {
	u := c.lastUsage.Load()
	if u == nil {
		return nil
	}
	cp := *u
	return &cp
}

// Call sends a single user prompt and returns the full assistant response.
func (c *OpenAI) Call(ctx context.Context, prompt string) (string, error) {
	return c.CallWithOptions(ctx, prompt, RequestOptions{})
}

func (c *OpenAI) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error) {
	return c.StreamWithOptions(ctx, prompt, opts, nil)
}

// Stream sends a prompt and invokes onChunk for each content delta. Returns
// the concatenated text. onChunk may be nil.
func (c *OpenAI) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (string, error) {
	return c.StreamWithOptions(ctx, prompt, RequestOptions{}, onChunk)
}

func (c *OpenAI) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(text string) error) (string, error) {
	params := openai.ChatCompletionNewParams{
		Model: shared.ChatModel(c.model),
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.UserMessage(prompt),
		},
		StreamOptions: openai.ChatCompletionStreamOptionsParam{
			IncludeUsage: param.NewOpt(true),
		},
	}
	applyOpenAIOptions(&params, opts)
	return c.runStream(ctx, params, onChunk)
}

func (c *OpenAI) StreamEvents(ctx context.Context, req StreamRequest) (<-chan StreamEvent, error) {
	return StreamEventsFromClient(ctx, c, req)
}

// CallWithTools runs one turn of a multi-turn tool conversation using
// OpenAI's native function-calling protocol.
func (c *OpenAI) CallWithTools(
	ctx context.Context,
	system string,
	messages []Message,
	tools []ToolDef,
) (Message, error) {
	return c.CallWithToolsOptions(ctx, system, messages, tools, RequestOptions{})
}

func (c *OpenAI) CallWithToolsOptions(
	ctx context.Context,
	system string,
	messages []Message,
	tools []ToolDef,
	opts RequestOptions,
) (Message, error) {
	if err := ValidateToolDefs(tools); err != nil {
		return Message{}, fmt.Errorf("openai: invalid tool schema: %w", err)
	}
	return c.callResponsesWithTools(ctx, system, messages, tools, opts)
}

func applyOpenAIOptions(params *openai.ChatCompletionNewParams, opts RequestOptions) {
	if opts.MaxTokens > 0 {
		params.MaxCompletionTokens = param.NewOpt(int64(opts.MaxTokens))
	}
	// REASONING EFFORT → OpenAI's native reasoning_effort, but ONLY on
	// current reasoning models (the GPT-5.6 family). Those models reject
	// temperature/top_p, so when effort applies we skip them; otherwise the
	// generation knobs flow as usual.
	if effort := openAIReasoningEffort(string(params.Model), opts.ReasoningEffort); effort != "" {
		params.ReasoningEffort = effort
		return
	}
	if opts.Temperature != nil {
		params.Temperature = param.NewOpt(*opts.Temperature)
	}
	if opts.TopP != nil {
		params.TopP = param.NewOpt(*opts.TopP)
	}
}

// openAIReasoningEffort maps a reasoning effort onto OpenAI's reasoning_effort
// value for the given model. Returns "" (leave unset) for ReasoningNone and
// for unknown models. Mind's executable OpenAI roster is GPT-5.6 only.
func openAIReasoningEffort(model string, effort ReasoningEffort) shared.ReasoningEffort {
	if effort == ReasoningNone || !openAIIsReasoningModel(model) {
		return ""
	}
	switch effort {
	case ReasoningLow:
		return shared.ReasoningEffortLow
	case ReasoningMedium:
		return shared.ReasoningEffortMedium
	case ReasoningHigh:
		return shared.ReasoningEffortHigh
	default:
		return ""
	}
}

// openAIIsReasoningModel reports whether a model accepts reasoning_effort.
func openAIIsReasoningModel(model string) bool {
	return model == OpenAIModelGPT56Sol ||
		model == OpenAIModelGPT56Terra ||
		model == OpenAIModelGPT56Luna
}

// runStream drives Chat.Completions.NewStreaming, accumulates content
// deltas, fans them out to onChunk, and records final usage.
func (c *OpenAI) runStream(
	ctx context.Context,
	params openai.ChatCompletionNewParams,
	onChunk func(text string) error,
) (string, error) {
	timeouts := c.timeouts.withDefaults()
	warnNearDeadline(ctx, "openai", c.model, timeouts, c.clock)

	attemptCtx, cancelAttempt := timeouts.attemptContext(ctx)
	defer cancelAttempt()

	streamCtx, cancelStream := context.WithCancelCause(attemptCtx)
	defer cancelStream(nil)
	var watchdog *time.Timer
	if timeouts.StreamStall > 0 {
		stall := timeouts.StreamStall
		watchdog = time.AfterFunc(stall, func() { cancelStream(stallError("openai", stall)) })
		defer watchdog.Stop()
	}

	stream := c.sdk.Chat.Completions.NewStreaming(streamCtx, params)
	defer func() { _ = stream.Close() }()

	var full []byte
	var lastUsage openai.CompletionUsage

	for stream.Next() {
		if watchdog != nil {
			watchdog.Reset(timeouts.StreamStall)
		}
		chunk := stream.Current()
		if chunk.Usage.TotalTokens > 0 {
			lastUsage = chunk.Usage
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta.Content
		if delta == "" {
			continue
		}
		full = append(full, delta...)
		if onChunk != nil {
			if err := onChunk(delta); err != nil {
				return string(full), err
			}
		}
	}
	if err := stream.Err(); err != nil {
		if cause := context.Cause(streamCtx); cause != nil && errors.Is(cause, ErrStreamStalled) {
			return string(full), cause
		}
		if ctxErr := ctx.Err(); ctxErr != nil {
			return string(full), c.classify(fmt.Errorf("%w (stream aborted: %v)", ctxErr, err))
		}
		if attemptCtx.Err() != nil {
			return string(full), c.classify(fmt.Errorf("openai: per-attempt timeout %s exceeded: %w (stream error: %v)", timeouts.PerAttempt, context.DeadlineExceeded, err))
		}
		return string(full), c.classify(err)
	}
	if ctxErr := ctx.Err(); ctxErr != nil {
		return string(full), c.classify(ctxErr)
	}

	c.storeUsage(lastUsage)
	return string(full), nil
}

func (c *OpenAI) storeUsage(u openai.CompletionUsage) {
	c.lastUsage.Store(&Usage{
		Provider:             c.Provider(),
		Model:                c.model,
		InputTokens:          int(u.PromptTokens),
		OutputTokens:         int(u.CompletionTokens),
		ReasoningTokens:      int(u.CompletionTokensDetails.ReasoningTokens),
		CachedTokens:         int(u.PromptTokensDetails.CachedTokens),
		CachedInputTokens:    int(u.PromptTokensDetails.CachedTokens),
		CacheReadInputTokens: int(u.PromptTokensDetails.CachedTokens),
	})
}

// ── Conversion helpers ──────────────────────────────────────────

// Compile-time contract assertions.
var (
	_ Client              = (*OpenAI)(nil)
	_ OptionedClient      = (*OpenAI)(nil)
	_ ToolClient          = (*OpenAI)(nil)
	_ OptionedToolClient  = (*OpenAI)(nil)
	_ EventStreamClient   = (*OpenAI)(nil)
	_ UsageProvider       = (*OpenAI)(nil)
	_ ProviderMeta        = (*OpenAI)(nil)
	_ TimeoutConfigurable = (*OpenAI)(nil)
)
