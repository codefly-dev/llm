package llm

// ARCHITECTURE: pkg/llm is the public provider-client package — clients,
// messages, tool-call schemas, request options, usage, costs, model IDs,
// middleware, and recorder/cassette helpers belong here, and provider-facing
// code should import pkg/llm.
//
// This file is the ONE deliberate re-export boundary, NOT deletion debt: the
// concrete wire structs and provider error types physically live in pkg/spec
// and pkg/apierror so the lower harness/runtime contracts (supervisor, etc.)
// can share them without importing pkg/llm and creating a dependency cycle.
// pkg/llm re-exports those shapes under their public names so callers get one
// import. New contracts still belong in pkg/spec; this file only surfaces them.
//
// Everything with real behavior lives in intent-named files (invoke.go,
// tool_schema.go, classify.go, …) — this file holds aliases and trivial
// pass-throughs only.

import (
	"github.com/codefly-dev/llm/apierror"
	"github.com/codefly-dev/llm/contract"
)

// ── Core interfaces ─────────────────────────────────────────────

type (
	Client                  = contract.Client
	CacheableClient         = contract.CacheableClient
	ToolClient              = contract.ToolClient
	UsageProvider           = contract.UsageProvider
	ProviderMeta            = contract.ProviderMeta
	EventStreamClient       = contract.EventStreamClient
	OptionedClient          = contract.OptionedClient
	OptionedCacheableClient = contract.OptionedCacheableClient
	OptionedToolClient      = contract.OptionedToolClient
)

// ── Wire types ──────────────────────────────────────────────────

type (
	Message         = contract.Message
	ToolCall        = contract.ToolCall
	ToolResult      = contract.ToolResult
	ReasoningBlock  = contract.ReasoningBlock
	ToolDef         = contract.ToolDef
	ToolParam       = contract.ToolParam
	Usage           = contract.Usage
	RequestOptions  = contract.RequestOptions
	ReasoningEffort = contract.ReasoningEffort
	ModelInfo       = contract.ModelInfo
	PricingProfile  = contract.PricingProfile
	PriceRule       = contract.PriceRule
	PriceLevel      = contract.PriceLevel
	PriceComponent  = contract.PriceComponent
	UsageCost       = contract.UsageCost
	Model           = contract.Model
	Options         = contract.Options
	CallRecord      = contract.CallRecord
	Block           = contract.Block
	StreamRequest   = contract.StreamRequest
	StreamEvent     = contract.StreamEvent
	StreamEventType = contract.StreamEventType
)

// ── Structured streaming ────────────────────────────────────────

// BlockSchema parses a raw LLM stream into structured Blocks.
//
// Kinds and markers are domain-specific — callers supply whatever maps to
// their prompt format. The spec does NOT enumerate canonical kinds; the
// string in Block.Kind means whatever the schema that emitted it wants it
// to mean.
type BlockSchema = contract.BlockSchema

// ── Composition primitive ───────────────────────────────────────

// Middleware wraps a Client and returns a new Client with added behavior.
type Middleware = contract.Middleware

// Reasoned is the JSON envelope for reasoning-first typed calls.
type Reasoned[T any] = contract.Reasoned[T]

// ── Cassette modes ──────────────────────────────────────────────

type RecordMode = contract.RecordMode

const (
	RecordReplayOnly = contract.RecordReplayOnly
	RecordAlways     = contract.RecordAlways
	RecordOnMiss     = contract.RecordOnMiss

	PriceLevelStandard = contract.PriceLevelStandard
	PriceLevelBatch    = contract.PriceLevelBatch
	PriceLevelPriority = contract.PriceLevelPriority

	PriceComponentInputTokens           = contract.PriceComponentInputTokens
	PriceComponentOutputTokens          = contract.PriceComponentOutputTokens
	PriceComponentReasoningTokens       = contract.PriceComponentReasoningTokens
	PriceComponentCacheReadInputTokens  = contract.PriceComponentCacheReadInputTokens
	PriceComponentCacheWriteInputTokens = contract.PriceComponentCacheWriteInputTokens
	PriceComponentRequest               = contract.PriceComponentRequest
	PriceComponentToolCall              = contract.PriceComponentToolCall
	PriceComponentWebSearchRequest      = contract.PriceComponentWebSearchRequest
	PriceComponentWebFetchRequest       = contract.PriceComponentWebFetchRequest

	StreamEventTextDelta      = contract.StreamEventTextDelta
	StreamEventReasoningDelta = contract.StreamEventReasoningDelta
	StreamEventToolCallStart  = contract.StreamEventToolCallStart
	StreamEventToolCallDelta  = contract.StreamEventToolCallDelta
	StreamEventToolCallEnd    = contract.StreamEventToolCallEnd
	StreamEventMessageEnd     = contract.StreamEventMessageEnd
	StreamEventError          = contract.StreamEventError
)

// ── Error classification ────────────────────────────────────────

type (
	ProviderError = apierror.ProviderError
	ErrorCode     = apierror.ErrorCode
)

const (
	CodeRateLimited     = apierror.CodeRateLimited
	CodeOverloaded      = apierror.CodeOverloaded
	CodeServer          = apierror.CodeServer
	CodeUnauthorized    = apierror.CodeUnauthorized
	CodeBadRequest      = apierror.CodeBadRequest
	CodeContextLength   = apierror.CodeContextLength
	CodeContentFiltered = apierror.CodeContentFiltered
	CodeModelNotFound   = apierror.CodeModelNotFound
	CodeStream          = apierror.CodeStream
	CodeTimeout         = apierror.CodeTimeout
	CodeUnknown         = apierror.CodeUnknown
)

// Sentinel errors for errors.Is matching. Use with llm.IsRetryable or
// errors.Is(err, llm.ErrRateLimited) etc.
var (
	ErrRateLimited     = apierror.ErrRateLimited
	ErrOverloaded      = apierror.ErrOverloaded
	ErrServer          = apierror.ErrServer
	ErrUnauthorized    = apierror.ErrUnauthorized
	ErrBadRequest      = apierror.ErrBadRequest
	ErrContextLength   = apierror.ErrContextLength
	ErrContentFiltered = apierror.ErrContentFiltered
	ErrModelNotFound   = apierror.ErrModelNotFound
	ErrStream          = apierror.ErrStream
	ErrTimeout         = apierror.ErrTimeout
)

// IsRetryable is true for rate-limit, overloaded, server, and timeout errors.
func IsRetryable(err error) bool {
	return apierror.IsRetryable(err)
}

// ClassifyStatus maps an HTTP status to an ErrorCode.
func ClassifyStatus(status int) ErrorCode {
	return apierror.ClassifyStatus(status)
}

// ── Model constants ─────────────────────────────────────────────

const (
	ModelAnthropic  = contract.ModelAnthropic
	ModelOpenAI     = contract.ModelOpenAI
	ReasoningNone   = contract.ReasoningNone
	ReasoningLow    = contract.ReasoningLow
	ReasoningMedium = contract.ReasoningMedium
	ReasoningHigh   = contract.ReasoningHigh

	ModelAnthropicHaiku   = contract.ModelAnthropicHaiku
	ModelAnthropicSonnet  = contract.ModelAnthropicSonnet
	ModelAnthropicOpus    = contract.ModelAnthropicOpus
	ModelOpenAIGPT56      = contract.ModelOpenAIGPT56
	ModelOpenAIGPT56Sol   = contract.ModelOpenAIGPT56Sol
	ModelOpenAIGPT56Terra = contract.ModelOpenAIGPT56Terra
	ModelOpenAIGPT56Luna  = contract.ModelOpenAIGPT56Luna
)
