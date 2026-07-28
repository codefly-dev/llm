package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	oteltrace "go.opentelemetry.io/otel/trace"
)

// OTELMiddleware creates a middleware that wraps every LLM call in an OTEL span.
// This captures Call, Stream, CallCached, and CallWithTools at the transport level.
func OTELMiddleware(model string) Middleware {
	return func(next Client) Client {
		return &otelMW{next: next, model: model, transportIdentity: ClientTransportIdentity(next)}
	}
}

func otelMiddleware(model string, transportIdentity TransportIdentity) Middleware {
	return func(next Client) Client {
		return &otelMW{next: next, model: model, transportIdentity: transportIdentity}
	}
}

type otelMW struct {
	next              Client
	model             string
	transportIdentity TransportIdentity
}

func (m *otelMW) Call(ctx context.Context, prompt string) (string, error) {
	return m.CallWithOptions(ctx, prompt, RequestOptions{})
}

func (m *otelMW) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error) {
	attrs := m.baseLLMAttrs("chat")
	attrs = append(attrs, attribute.Int("llm.prompt_len", len(prompt)), attribute.String("input.mime_type", "text/plain"))
	if captureOTELContent() {
		attrs = append(attrs, attribute.String("input.value", prompt))
	}
	_, modelName := providerAndModel(m.model)
	ctx, span := otel.Tracer("mind").Start(ctx, llmSpanName("chat", modelName), oteltrace.WithAttributes(attrs...))
	defer span.End()

	resp, err := CallWithOptions(ctx, m.next, prompt, opts)
	if err != nil {
		recordSpanError(span, err)
		return resp, err
	}
	attrs = []attribute.KeyValue{
		attribute.Int("llm.response_len", len(resp)),
		attribute.String("output.mime_type", "text/plain"),
	}
	if captureOTELContent() {
		attrs = append(attrs, attribute.String("output.value", resp))
	}
	span.SetAttributes(attrs...)
	m.recordUsage(span)
	return resp, nil
}

func (m *otelMW) Stream(ctx context.Context, prompt string, onChunk func(string) error) (string, error) {
	return m.StreamWithOptions(ctx, prompt, RequestOptions{}, onChunk)
}

func (m *otelMW) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(string) error) (string, error) {
	attrs := m.baseLLMAttrs("chat")
	attrs = append(attrs,
		attribute.Int("llm.prompt_len", len(prompt)),
		attribute.Bool("llm.request.stream", true),
		attribute.String("input.mime_type", "text/plain"),
	)
	if captureOTELContent() {
		attrs = append(attrs, attribute.String("input.value", prompt))
	}
	_, modelName := providerAndModel(m.model)
	ctx, span := otel.Tracer("mind").Start(ctx, llmSpanName("chat", modelName), oteltrace.WithAttributes(attrs...))
	defer span.End()

	resp, err := StreamWithOptions(ctx, m.next, prompt, opts, onChunk)
	if err != nil {
		recordSpanError(span, err)
		return resp, err
	}
	attrs = []attribute.KeyValue{
		attribute.Int("llm.response_len", len(resp)),
		attribute.String("output.mime_type", "text/plain"),
	}
	if captureOTELContent() {
		attrs = append(attrs, attribute.String("output.value", resp))
	}
	span.SetAttributes(attrs...)
	m.recordUsage(span)
	return resp, nil
}

func (m *otelMW) CallCached(ctx context.Context, system, user string) (string, error) {
	return m.CallCachedWithOptions(ctx, system, user, RequestOptions{})
}

func (m *otelMW) CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (string, error) {
	attrs := m.baseLLMAttrs("chat")
	attrs = append(attrs,
		attribute.Int("llm.system_len", len(system)),
		attribute.Int("llm.user_len", len(user)),
		attribute.String("input.mime_type", "application/json"),
	)
	if captureOTELContent() {
		attrs = append(attrs,
			attribute.String("llm.input_messages.0.message.role", "system"),
			attribute.String("llm.input_messages.0.message.content", system),
			attribute.String("llm.input_messages.1.message.role", "user"),
			attribute.String("llm.input_messages.1.message.content", user),
		)
	}
	_, modelName := providerAndModel(m.model)
	ctx, span := otel.Tracer("mind").Start(ctx, llmSpanName("chat", modelName), oteltrace.WithAttributes(attrs...))
	defer span.End()

	resp, err := CallCachedWithOptions(ctx, m.next, system, user, opts)
	if err != nil {
		recordSpanError(span, err)
		return resp, err
	}
	attrs = []attribute.KeyValue{
		attribute.Int("llm.response_len", len(resp)),
		attribute.String("output.mime_type", "text/plain"),
	}
	if captureOTELContent() {
		attrs = append(attrs, attribute.String("output.value", resp))
	}
	span.SetAttributes(attrs...)
	m.recordUsage(span)
	return resp, nil
}

func (m *otelMW) CallWithTools(ctx context.Context, system string, messages []Message, toolDefs []ToolDef) (Message, error) {
	return m.CallWithToolsOptions(ctx, system, messages, toolDefs, RequestOptions{})
}

func (m *otelMW) CallWithToolsOptions(ctx context.Context, system string, messages []Message, toolDefs []ToolDef, opts RequestOptions) (Message, error) {
	attrs := m.baseLLMAttrs("chat")
	attrs = append(attrs,
		attribute.Int("llm.system_len", len(system)),
		attribute.Int("llm.messages", len(messages)),
		attribute.Int("llm.tools", len(toolDefs)),
		attribute.String("input.mime_type", "application/json"),
	)
	attrs = append(attrs, toolDefAttributes(toolDefs, captureOTELContent())...)
	if captureOTELContent() {
		attrs = append(attrs, messageInputAttributes(system, messages)...)
	}
	_, modelName := providerAndModel(m.model)
	ctx, span := otel.Tracer("mind").Start(ctx, llmSpanName("chat", modelName), oteltrace.WithAttributes(attrs...))
	defer span.End()

	resp, err := CallWithToolsOptions(ctx, m.next, system, messages, toolDefs, opts)
	if err != nil {
		recordSpanError(span, err)
		return resp, err
	}
	attrs = []attribute.KeyValue{
		attribute.Int("llm.response_len", len(resp.Content)),
		attribute.Int("llm.tool_calls", len(resp.ToolCalls)),
		attribute.String("output.mime_type", "application/json"),
	}
	attrs = append(attrs, toolCallOutputAttributes(resp.ToolCalls, captureOTELContent())...)
	if captureOTELContent() {
		attrs = append(attrs,
			attribute.String("output.value", messageJSON(resp)),
			attribute.String("llm.output_messages.0.message.role", resp.Role),
			attribute.String("llm.output_messages.0.message.content", resp.Content),
		)
	}
	span.SetAttributes(attrs...)
	m.recordUsage(span)
	return resp, nil
}

func (m *otelMW) Unwrap() Client { return m.next }
func (m *otelMW) TransportIdentity() TransportIdentity {
	return m.transportIdentity
}

func (m *otelMW) baseLLMAttrs(operation string) []attribute.KeyValue {
	provider, modelName := providerAndModel(m.model)
	return []attribute.KeyValue{
		attribute.String("llm.model", m.model),
		attribute.String("openinference.span.kind", "LLM"),
		attribute.String("llm.system", provider),
		attribute.String("llm.provider", provider),
		attribute.String("llm.model_name", modelName),
		attribute.String("llm.operation", operation),
		attribute.String("llm.transport.identity", string(m.transportIdentity)),
	}
}

// recordUsage extracts real token counts from the underlying provider
// (if it implements UsageProvider) and attaches them to the span.
func (m *otelMW) recordUsage(span oteltrace.Span) {
	provider := findUsageProvider(m.next)
	if provider == nil {
		return
	}
	u := provider.LastCallUsage()
	if u == nil {
		return
	}
	span.SetAttributes(
		attribute.Int("llm.tokens.input", u.InputTokens),
		attribute.Int("llm.tokens.output", u.OutputTokens),
		attribute.Int("llm.tokens.reasoning", u.ReasoningTokens),
		attribute.Int("llm.tokens.cached", u.CachedTokens),
		attribute.Int("llm.requests", u.Requests),
		attribute.Int("llm.tool_calls", u.ToolCalls),
		attribute.Int("llm.hosted_tool_calls", u.HostedToolCalls),
		attribute.Int("llm.tool_calls.web_search", u.WebSearchRequests),
		attribute.Int("llm.tool_calls.web_fetch", u.WebFetchRequests),
		attribute.Int("llm.token_count.prompt", u.InputTokens),
		attribute.Int("llm.token_count.completion", u.OutputTokens),
		attribute.Int("llm.token_count.completion_details.reasoning", u.ReasoningTokens),
		attribute.Int("llm.token_count.total", u.InputTokens+u.OutputTokens),
		attribute.Int("llm.token_count.prompt_details.cache_read", u.CacheReadInputTokens),
		attribute.Int("llm.token_count.prompt_details.cache_write", u.CacheWriteInputTokens),
		attribute.Float64("llm.cost.prompt", u.InputCostUSD+u.CacheReadCostUSD+u.CacheWriteCostUSD),
		attribute.Float64("llm.cost.completion", u.OutputCostUSD+u.ReasoningCostUSD),
		attribute.Float64("llm.cost.completion_details.reasoning", u.ReasoningCostUSD),
		attribute.Float64("llm.cost.total", u.CostUSD),
		attribute.Float64("llm.cost.prompt_details.input", u.InputCostUSD),
		attribute.Float64("llm.cost.prompt_details.cache_read", u.CacheReadCostUSD),
		attribute.Float64("llm.cost.prompt_details.cache_write", u.CacheWriteCostUSD),
		attribute.Float64("llm.cost.completion_details.output", u.OutputCostUSD),
		attribute.Float64("llm.cost.requests", u.RequestCostUSD),
		attribute.Float64("llm.cost.tools", u.ToolCostUSD+u.WebSearchCostUSD+u.WebFetchCostUSD),
		attribute.Float64("llm.cost.tools.web_search", u.WebSearchCostUSD),
		attribute.Float64("llm.cost.tools.web_fetch", u.WebFetchCostUSD),
		attribute.String("llm.response.model", u.Model),
	)
}

func providerAndModel(full string) (string, string) {
	provider, modelName, ok := strings.Cut(full, "/")
	provider = strings.ToLower(strings.TrimSpace(provider))
	modelName = strings.TrimSpace(modelName)
	if !ok {
		modelName = provider
		if provider == "" {
			provider = "unknown"
		}
	}
	if modelName == "" {
		modelName = provider
	}
	return provider, modelName
}

func llmSpanName(operation, modelName string) string {
	if modelName == "" {
		return operation
	}
	return operation + " " + modelName
}

// captureOTELContent gates whether OTEL spans capture full
// request/response content (prompts, tool args, results). Delegates
// to pkg/config so the env contract has ONE parser. See
// pkg/config/recorder.go for the truthy-string set.
func captureOTELContent() bool {
	return otelCaptureContentEnabled()
}

func recordSpanError(span oteltrace.Span, err error) {
	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
	span.SetAttributes(attribute.String("error.type", fmt.Sprintf("%T", err)))
}

func toolDefAttributes(tools []ToolDef, captureSchema bool) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, len(tools)*2)
	for i, tool := range tools {
		prefix := fmt.Sprintf("llm.tools.%d.tool", i)
		attrs = append(attrs, attribute.String(prefix+".name", tool.Name))
		if captureSchema {
			attrs = append(attrs, attribute.String(prefix+".json_schema", toolDefJSON(tool)))
		}
	}
	return attrs
}

func messageInputAttributes(system string, messages []Message) []attribute.KeyValue {
	attrs := []attribute.KeyValue{}
	idx := 0
	if system != "" {
		attrs = append(attrs,
			attribute.String(fmt.Sprintf("llm.input_messages.%d.message.role", idx), "system"),
			attribute.String(fmt.Sprintf("llm.input_messages.%d.message.content", idx), system),
		)
		idx++
	}
	for _, msg := range messages {
		prefix := fmt.Sprintf("llm.input_messages.%d.message", idx)
		attrs = append(attrs,
			attribute.String(prefix+".role", msg.Role),
			attribute.String(prefix+".content", msg.Content),
		)
		if msg.ToolCallID != "" {
			attrs = append(attrs, attribute.String(prefix+".tool_call_id", msg.ToolCallID))
		}
		idx++
	}
	return attrs
}

func toolCallOutputAttributes(calls []ToolCall, captureArgs bool) []attribute.KeyValue {
	attrs := make([]attribute.KeyValue, 0, len(calls)*3)
	for i, call := range calls {
		prefix := fmt.Sprintf("llm.output_messages.0.message.tool_calls.%d.tool_call", i)
		attrs = append(attrs,
			attribute.String(prefix+".id", call.ID),
			attribute.String(prefix+".function.name", call.Name),
		)
		if captureArgs {
			attrs = append(attrs, attribute.String(prefix+".function.arguments", marshalJSONString(call.Args)))
		}
	}
	return attrs
}

func toolDefJSON(tool ToolDef) string {
	return marshalJSONString(map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":        tool.Name,
			"description": tool.Description,
			"parameters":  tool.Parameters,
		},
	})
}

func messageJSON(msg Message) string {
	return marshalJSONString(msg)
}

func marshalJSONString(v any) string {
	data, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(data)
}
