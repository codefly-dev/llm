// Package contract is the provider-facing LLM contract shared between Mind and
// codefly. It is a leaf: plain request/response/tool/model/pricing/streaming
// types and the client interfaces, with NO provider SDK imports and NO Mind
// harness types (Objective, Capability, ContextEnvelope, …). Both the Mind
// harness contract (github.com/codefly-dev/mind/pkg/spec) and the provider
// client implementation (github.com/codefly-dev/mind/pkg/llm, and eventually
// this module's own client package) import it, so it must not depend on either.
//
// ARCHITECTURE: the wire types physically live here — below both the harness
// contract and the SDK-backed clients — precisely so neither layer has to
// import the other. spec's LLMRequest/LLMResponse embed these types
// (spec -> contract); the provider clients produce and consume them
// (client -> contract); there is no cycle because contract imports only stdlib.
package contract

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// Model identifies an LLM provider and optional model id.
type Model string

// ModelFamily is a run-level provider constraint. Difficulty still chooses a
// role within the family; the family only bounds which provider may execute
// the call. Empty means unrestricted/custom selector behavior.
type ModelFamily string

const (
	ModelFamilyAnthropic ModelFamily = "anthropic"
	ModelFamilyOpenAI    ModelFamily = "openai"
)

// ParseModelFamily validates a user/configuration value without silently
// treating a typo as the default provider.
func ParseModelFamily(raw string) (ModelFamily, error) {
	family := ModelFamily(strings.ToLower(strings.TrimSpace(raw)))
	switch family {
	case "", ModelFamilyAnthropic, ModelFamilyOpenAI:
		return family, nil
	default:
		return "", fmt.Errorf("unsupported model family %q", raw)
	}
}

func (f ModelFamily) Validate() error {
	_, err := ParseModelFamily(string(f))
	return err
}

func (f ModelFamily) Provider() string { return string(f) }

const (
	ModelAnthropic Model = "anthropic"
	ModelOpenAI    Model = "openai"
	// Anthropic's executable roster is intentionally closed: every call uses
	// one of these three current models. A cassette pins the response, not an
	// obsolete live model; adding a model is an explicit catalog/roster change.
	ModelAnthropicHaiku  Model = "anthropic/claude-haiku-4-5-20251001"
	ModelAnthropicSonnet Model = "anthropic/claude-sonnet-5"
	ModelAnthropicOpus   Model = "anthropic/claude-opus-4-8"
	// OpenAI current GPT-5.6 family (July 2026). Use canonical tier IDs so
	// recorder keys and comparisons name the actual family member; gpt-5.6 is
	// an alias for Sol.
	ModelOpenAIGPT56      Model = "openai/gpt-5.6-sol"
	ModelOpenAIGPT56Sol   Model = "openai/gpt-5.6-sol"
	ModelOpenAIGPT56Terra Model = "openai/gpt-5.6-terra"
	ModelOpenAIGPT56Luna  Model = "openai/gpt-5.6-luna"
)

// Ref parses a "provider/model-id" (optionally "@version") Model string
// into a structured ModelRef. A bare "provider" yields {Provider} only.
// Inverse of ModelRef.String().
func (m Model) Ref() ModelRef {
	s := string(m)
	i := strings.IndexByte(s, '/')
	if i < 0 {
		return ModelRef{Provider: s}
	}
	ref := ModelRef{Provider: s[:i]}
	s = s[i+1:]
	if j := strings.IndexByte(s, '@'); j >= 0 {
		ref.Version = s[j+1:]
		s = s[:j]
	}
	ref.Model = s
	return ref
}

// ModelRef is a structured provider/model/version reference.
type ModelRef struct {
	Provider string
	Model    string
	Version  string
}

func (m ModelRef) String() string {
	if m.Provider == "" && m.Model == "" {
		return ""
	}
	if m.Provider == "" {
		return m.Model
	}
	if m.Model == "" {
		return m.Provider
	}
	if m.Version != "" {
		return m.Provider + "/" + m.Model + "@" + m.Version
	}
	return m.Provider + "/" + m.Model
}

// Options configures an LLM client at construction time.
type Options struct {
	Model           string
	AnthropicAPIKey string
	OpenAIAPIKey    string
	RateLimit       float64
}

// Client is the minimum LLM abstraction: text in, text out, optionally streamed.
type Client interface {
	Call(ctx context.Context, prompt string) (string, error)
	Stream(ctx context.Context, prompt string, onChunk func(text string) error) (full string, err error)
}

// CacheableClient extends Client with provider prompt caching.
type CacheableClient interface {
	Client
	CallCached(ctx context.Context, system, user string) (string, error)
}

// ReasoningEffort is a provider-agnostic reasoning level. The Anthropic
// client maps it to a thinking-token budget for extended-thinking models;
// adaptive models (Sonnet 5 and Opus 4.8) run at their server default.
// Empty (ReasoningNone) means "do not request extra reasoning".
type ReasoningEffort string

const (
	ReasoningNone   ReasoningEffort = ""
	ReasoningLow    ReasoningEffort = "low"
	ReasoningMedium ReasoningEffort = "medium"
	ReasoningHigh   ReasoningEffort = "high"
)

// RequestOptions carries per-call generation controls.
type RequestOptions struct {
	MaxTokens       int             `json:"max_tokens,omitempty"`
	Temperature     *float64        `json:"temperature,omitempty"`
	TopP            *float64        `json:"top_p,omitempty"`
	PriceLevel      string          `json:"price_level,omitempty"`
	ReasoningEffort ReasoningEffort `json:"reasoning_effort,omitempty"`

	// WebSearchMaxUses, when > 0, asks the provider to attach its HOSTED
	// web-search server tool to this call, capped at this many searches.
	// On Anthropic this emits the web_search_20250305 server tool: the
	// model runs the search server-side at Anthropic (billed on the LLM
	// key Mind already holds) and the response carries web_search_tool_result
	// blocks (sources) plus the synthesized answer. Zero = no hosted search.
	// This is how Mind's `web_search` tool works without a Tavily/SerpAPI key.
	WebSearchMaxUses int `json:"web_search_max_uses,omitempty"`
}

type OptionedClient interface {
	Client
	CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error)
	StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(text string) error) (full string, err error)
}

type OptionedCacheableClient interface {
	CacheableClient
	CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (string, error)
}

type OptionedToolClient interface {
	ToolClient
	CallWithToolsOptions(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error)
}

// Message is one turn in a multi-turn LLM conversation.
type Message struct {
	Role       string           `json:"role"`
	Content    string           `json:"content,omitempty"`
	Reasoning  []ReasoningBlock `json:"reasoning,omitempty"`
	ToolCalls  []ToolCall       `json:"tool_calls,omitempty"`
	ToolResult *ToolResult      `json:"tool_result,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
}

// ToolCall is a provider-emitted request to invoke a named tool.
type ToolCall struct {
	ID       string         `json:"id"`
	Name     string         `json:"name"`
	Args     map[string]any `json:"args"`
	RawArgs  string         `json:"raw_args,omitempty"`
	ArgError string         `json:"arg_error,omitempty"`
}

// ReasoningBlock preserves provider-native reasoning/thinking state.
type ReasoningBlock struct {
	Provider         string `json:"provider,omitempty"`
	Type             string `json:"type"`
	Text             string `json:"text,omitempty"`
	Signature        string `json:"signature,omitempty"`
	Data             string `json:"data,omitempty"`
	EncryptedContent string `json:"encrypted_content,omitempty"`
}

// ToolResult is the outcome of executing a tool call.
type ToolResult struct {
	CallID  string `json:"call_id"`
	Content string `json:"content"`
	IsError bool   `json:"is_error,omitempty"`
}

// ToolDef declares a tool the model is permitted to call.
type ToolDef struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  []ToolParam    `json:"parameters,omitempty"`
	InputSchema map[string]any `json:"input_schema,omitempty"`
	// StrictInput asks providers with native schema-constrained tool use to
	// guarantee that generated arguments conform to InputSchema. Providers
	// that cannot honor the guarantee must fail the request rather than weaken
	// Mind's typed protocol at the decoding boundary.
	StrictInput bool `json:"strict_input,omitempty"`
}

// ToolParam describes one flat tool parameter.
type ToolParam struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
	Required    bool   `json:"required"`
}

type ToolClient interface {
	Client
	CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (Message, error)
}

// ProviderMeta is the self-identification interface every provider implements.
type ProviderMeta interface {
	Provider() string
	Model() string
}

// Usage holds token, cost, and timing data for one LLM call.
type Usage struct {
	Provider              string  `json:"provider"`
	Model                 string  `json:"model"`
	PriceLevel            string  `json:"price_level,omitempty"`
	Requests              int     `json:"requests,omitempty"`
	InputTokens           int     `json:"input_tokens"`
	OutputTokens          int     `json:"output_tokens"`
	ReasoningTokens       int     `json:"reasoning_tokens,omitempty"`
	ReasoningOutputExtra  bool    `json:"reasoning_output_extra,omitempty"`
	CachedTokens          int     `json:"cached_tokens"`
	CachedInputTokens     int     `json:"cached_input_tokens,omitempty"`
	CachedOutputTokens    int     `json:"cached_output_tokens,omitempty"`
	CacheReadInputTokens  int     `json:"cache_read_input_tokens,omitempty"`
	CacheWriteInputTokens int     `json:"cache_write_input_tokens,omitempty"`
	ToolCalls             int     `json:"tool_calls,omitempty"`
	HostedToolCalls       int     `json:"hosted_tool_calls,omitempty"`
	WebSearchRequests     int     `json:"web_search_requests,omitempty"`
	WebFetchRequests      int     `json:"web_fetch_requests,omitempty"`
	InputCostUSD          float64 `json:"input_cost_usd,omitempty"`
	OutputCostUSD         float64 `json:"output_cost_usd,omitempty"`
	ReasoningCostUSD      float64 `json:"reasoning_cost_usd,omitempty"`
	CacheReadCostUSD      float64 `json:"cache_read_cost_usd,omitempty"`
	CacheWriteCostUSD     float64 `json:"cache_write_cost_usd,omitempty"`
	RequestCostUSD        float64 `json:"request_cost_usd,omitempty"`
	ToolCostUSD           float64 `json:"tool_cost_usd,omitempty"`
	WebSearchCostUSD      float64 `json:"web_search_cost_usd,omitempty"`
	WebFetchCostUSD       float64 `json:"web_fetch_cost_usd,omitempty"`
	CostUSD               float64 `json:"cost_usd"`
	DurationMS            int64   `json:"duration_ms"`
	Estimated             bool    `json:"estimated,omitempty"`
}

type UsageProvider interface {
	LastCallUsage() *Usage
}

// LLMUsage is the canonical per-call usage record — an alias for Usage so
// adapter/debug/trace-export code shares one struct definition.
type LLMUsage = Usage

type LLMTransport interface {
	CompleteRaw(ctx context.Context, req ProviderLLMRequest) (ProviderLLMResponse, error)
}

type ProviderLLMRequest struct {
	Provider string
	Model    string
	System   string
	Messages []ProviderMessage
	Tools    []ProviderToolDef
	MaxOut   int
	Metadata map[string]any
}

type ProviderMessage struct {
	Role    string
	Content string
	Extra   map[string]any
}

type ProviderToolDef struct {
	Name        string
	Description string
	Schema      map[string]any
	Version     int
}

type ProviderLLMResponse struct {
	Content    string
	ToolCalls  []ProviderToolCall
	StopReason string
	Usage      ProviderUsage
	Raw        map[string]any
}

type ProviderToolCall struct {
	ID   string
	Name string
	Args map[string]any
}

type ProviderUsage struct {
	InputTokens  int
	OutputTokens int
	CostUSD      float64
}

// ModelInfo describes a model's context limits and pricing contract.
type ModelInfo struct {
	Provider         string         `json:"provider"`
	Model            string         `json:"model"`
	ContextWindow    int            `json:"context_window"`
	MaxOutput        int            `json:"max_output"`
	InputPricePer1M  float64        `json:"input_price_per_1m"`
	OutputPricePer1M float64        `json:"output_price_per_1m"`
	Pricing          PricingProfile `json:"pricing,omitempty"`
}

type PriceLevel string

const (
	PriceLevelStandard PriceLevel = "standard"
	PriceLevelBatch    PriceLevel = "batch"
	PriceLevelPriority PriceLevel = "priority"
)

type PriceComponent string

const (
	PriceComponentInputTokens           PriceComponent = "input_tokens"
	PriceComponentOutputTokens          PriceComponent = "output_tokens"
	PriceComponentReasoningTokens       PriceComponent = "reasoning_tokens"
	PriceComponentCacheReadInputTokens  PriceComponent = "cache_read_input_tokens"
	PriceComponentCacheWriteInputTokens PriceComponent = "cache_write_input_tokens"
	PriceComponentRequest               PriceComponent = "request"
	PriceComponentToolCall              PriceComponent = "tool_call"
	PriceComponentWebSearchRequest      PriceComponent = "web_search_request"
	PriceComponentWebFetchRequest       PriceComponent = "web_fetch_request"
)

type PriceRule struct {
	Component PriceComponent `json:"component"`
	Level     PriceLevel     `json:"level,omitempty"`
	USD       float64        `json:"usd"`
	UnitSize  int            `json:"unit_size,omitempty"`
}

type PricingProfile struct {
	Currency     string      `json:"currency,omitempty"`
	DefaultLevel PriceLevel  `json:"default_level,omitempty"`
	Rules        []PriceRule `json:"rules,omitempty"`
}

type UsageCost struct {
	InputUSD      float64
	OutputUSD     float64
	ReasoningUSD  float64
	CacheReadUSD  float64
	CacheWriteUSD float64
	RequestUSD    float64
	ToolUSD       float64
	WebSearchUSD  float64
	WebFetchUSD   float64
	TotalUSD      float64
}

// CallRecord captures one LLM call for debug inspection.
type CallRecord struct {
	Timestamp       time.Time `json:"timestamp"`
	Model           string    `json:"model"`
	Agent           string    `json:"agent"`
	PromptHash      string    `json:"prompt_hash"`
	PromptPreview   string    `json:"prompt_preview"`
	ResponsePreview string    `json:"response_preview"`
	FullPrompt      string    `json:"full_prompt"`
	FullResponse    string    `json:"full_response"`
	InputTokens     int       `json:"input_tokens"`
	OutputTokens    int       `json:"output_tokens"`
	CachedTokens    int       `json:"cached_tokens"`
	CostUSD         float64   `json:"cost_usd"`
	DurationMS      int64     `json:"duration_ms"`
	CacheHit        bool      `json:"cache_hit"`
	ToolCallCount   int       `json:"tool_call_count"`
}

type RecordMode int

const (
	RecordReplayOnly RecordMode = iota
	RecordAlways
	// RecordOnMiss replays certified calls when present and invokes the real
	// provider only for absent recordings. It exists for explicit cassette
	// healing; CI remains fail-closed on RecordReplayOnly.
	RecordOnMiss
)

type StreamEventType string

const (
	StreamEventTextDelta      StreamEventType = "text_delta"
	StreamEventReasoningDelta StreamEventType = "reasoning_delta"
	StreamEventToolCallStart  StreamEventType = "tool_call_start"
	StreamEventToolCallDelta  StreamEventType = "tool_call_delta"
	StreamEventToolCallEnd    StreamEventType = "tool_call_end"
	StreamEventMessageEnd     StreamEventType = "message_end"
	StreamEventError          StreamEventType = "error"
)

type StreamRequest struct {
	Prompt   string    `json:"prompt,omitempty"`
	System   string    `json:"system,omitempty"`
	Messages []Message `json:"messages,omitempty"`
	Tools    []ToolDef `json:"tools,omitempty"`
}

type StreamEvent struct {
	Type      StreamEventType `json:"type"`
	Text      string          `json:"text,omitempty"`
	ToolCall  *ToolCall       `json:"tool_call,omitempty"`
	Reasoning *ReasoningBlock `json:"reasoning,omitempty"`
	Message   *Message        `json:"message,omitempty"`
	Usage     *Usage          `json:"usage,omitempty"`
	Error     string          `json:"error,omitempty"`
	Synthetic bool            `json:"synthetic,omitempty"`
}

type EventStreamClient interface {
	Client
	StreamEvents(ctx context.Context, req StreamRequest) (<-chan StreamEvent, error)
}

// Block is one parsed element from a structured LLM stream.
type Block struct {
	Kind  string `json:"kind"`
	Delta string `json:"delta,omitempty"`
	Text  string `json:"text,omitempty"`
	Done  bool   `json:"done,omitempty"`
	Err   error  `json:"-"`
}

type BlockSchema interface {
	Feed(delta string) []Block
	Close() []Block
}

type Middleware func(Client) Client

// Limiter caps API calls. Implementations decide where rate-limit state
// lives: in-process, Redis, or another shared backend.
type Limiter interface {
	Allow(ctx context.Context) error
}

// Reasoned is the JSON envelope for reasoning-first typed calls.
type Reasoned[T any] struct {
	Reasoning string `json:"reasoning" llm:"brief auditable decision rationale grounded in the supplied evidence; do not expose hidden chain-of-thought"`
	Answer    T      `json:"answer" llm:"the complete typed answer"`
}
