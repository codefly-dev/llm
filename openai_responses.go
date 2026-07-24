package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/openai/openai-go/packages/param"
	"github.com/openai/openai-go/responses"
	"github.com/openai/openai-go/shared"
)

// callResponsesWithTools uses OpenAI's Responses API for every native-tool
// turn. The current GPT family does not support reasoning plus function tools
// on Chat Completions; one modern transport avoids silent option downgrades and
// preserves reasoning state for resumable multi-turn runs.
func (c *OpenAI) callResponsesWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error) {
	params, err := openAIResponseParams(c.model, system, messages, tools, opts)
	if err != nil {
		return Message{}, fmt.Errorf("openai responses request: %w", err)
	}
	timeouts := c.timeouts.withDefaults()
	warnNearDeadline(ctx, "openai", c.model, timeouts, c.clock)
	attemptCtx, cancelAttempt := timeouts.attemptContext(ctx)
	defer cancelAttempt()

	resp, err := c.sdk.Responses.New(attemptCtx, params)
	if err != nil {
		if ctx.Err() == nil && attemptCtx.Err() != nil {
			return Message{}, c.classify(fmt.Errorf("openai: per-attempt timeout %s exceeded: %w (request error: %v)", timeouts.PerAttempt, context.DeadlineExceeded, err))
		}
		return Message{}, c.classify(err)
	}
	c.storeResponseUsage(resp.Usage)
	if resp.Error.Message != "" {
		return Message{}, fmt.Errorf("openai responses: %s: %s", resp.Error.Code, resp.Error.Message)
	}
	if resp.Status != "" && resp.Status != responses.ResponseStatusCompleted {
		reason := resp.IncompleteDetails.Reason
		if reason == "" {
			reason = string(resp.Status)
		}
		return Message{}, fmt.Errorf("openai responses: response %s: %s", resp.ID, reason)
	}
	msg, err := fromOpenAIResponse(resp)
	if err != nil {
		return Message{}, fmt.Errorf("openai responses: %w", err)
	}
	return msg, nil
}

func openAIResponseParams(model, system string, messages []Message, tools []ToolDef, opts RequestOptions) (responses.ResponseNewParams, error) {
	input, err := toOpenAIResponseInput(messages)
	if err != nil {
		return responses.ResponseNewParams{}, err
	}
	params := responses.ResponseNewParams{
		Model: shared.ResponsesModel(model),
		Input: responses.ResponseNewParamsInputUnion{OfInputItemList: input},
		Tools: toOpenAIResponseTools(tools),
		Include: []responses.ResponseIncludable{
			responses.ResponseIncludableReasoningEncryptedContent,
		},
		ParallelToolCalls: param.NewOpt(true),
		Store:             param.NewOpt(false),
	}
	if strings.TrimSpace(system) != "" {
		params.Instructions = param.NewOpt(system)
	}
	if opts.MaxTokens > 0 {
		params.MaxOutputTokens = param.NewOpt(int64(opts.MaxTokens))
	}
	if effort := openAIReasoningEffort(model, opts.ReasoningEffort); effort != "" {
		params.Reasoning = shared.ReasoningParam{Effort: effort}
	} else {
		if opts.Temperature != nil {
			params.Temperature = param.NewOpt(*opts.Temperature)
		}
		if opts.TopP != nil {
			params.TopP = param.NewOpt(*opts.TopP)
		}
	}
	return params, nil
}

func toOpenAIResponseTools(tools []ToolDef) []responses.ToolUnionParam {
	out := make([]responses.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		out = append(out, responses.ToolUnionParam{OfFunction: &responses.FunctionToolParam{
			Name:        tool.Name,
			Description: param.NewOpt(tool.Description),
			Parameters:  toolInputSchema(tool),
			// Mind validates the exact schema locally on both live and replayed
			// responses. Strict=false avoids provider-specific restrictions on
			// otherwise valid JSON Schema constructs such as optional fields.
			Strict: param.NewOpt(false),
		}})
	}
	return out
}

func toOpenAIResponseInput(messages []Message) (responses.ResponseInputParam, error) {
	input := make(responses.ResponseInputParam, 0, len(messages))
	for i, message := range messages {
		switch message.Role {
		case "user":
			input = append(input, responses.ResponseInputItemParamOfMessage(message.Content, responses.EasyInputMessageRoleUser))
		case "assistant":
			for j, block := range message.Reasoning {
				item, err := reasoningInputItem(block)
				if err != nil {
					return nil, fmt.Errorf("message %d reasoning block %d: %w", i, j, err)
				}
				input = append(input, item)
			}
			if message.Content != "" {
				input = append(input, responses.ResponseInputItemParamOfMessage(message.Content, responses.EasyInputMessageRoleAssistant))
			}
			for j, call := range message.ToolCalls {
				if strings.TrimSpace(call.ID) == "" || strings.TrimSpace(call.Name) == "" {
					return nil, fmt.Errorf("message %d tool call %d requires id and name", i, j)
				}
				arguments, err := json.Marshal(call.Args)
				if err != nil {
					return nil, fmt.Errorf("message %d tool call %d arguments: %w", i, j, err)
				}
				input = append(input, responses.ResponseInputItemParamOfFunctionCall(string(arguments), call.ID, call.Name))
			}
		case "tool_result":
			callID := strings.TrimSpace(message.ToolCallID)
			content := message.Content
			if message.ToolResult != nil {
				if callID == "" {
					callID = strings.TrimSpace(message.ToolResult.CallID)
				}
				content = message.ToolResult.Content
			}
			if callID == "" {
				return nil, fmt.Errorf("message %d tool result requires call id", i)
			}
			input = append(input, responses.ResponseInputItemParamOfFunctionCallOutput(callID, content))
		default:
			return nil, fmt.Errorf("message %d has unsupported role %q", i, message.Role)
		}
	}
	return input, nil
}

func reasoningInputItem(block ReasoningBlock) (responses.ResponseInputItemUnionParam, error) {
	if block.Provider != "" && block.Provider != "openai" {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("provider %q cannot be sent to OpenAI", block.Provider)
	}
	if strings.TrimSpace(block.Data) == "" {
		return responses.ResponseInputItemUnionParam{}, fmt.Errorf("reasoning item id is required")
	}
	item := responses.ResponseReasoningItemParam{
		ID:      block.Data,
		Summary: []responses.ResponseReasoningItemSummaryParam{},
	}
	if block.Text != "" {
		item.Summary = append(item.Summary, responses.ResponseReasoningItemSummaryParam{Text: block.Text})
	}
	if block.EncryptedContent != "" {
		item.EncryptedContent = param.NewOpt(block.EncryptedContent)
	}
	return responses.ResponseInputItemUnionParam{OfReasoning: &item}, nil
}

func fromOpenAIResponse(resp *responses.Response) (Message, error) {
	if resp == nil {
		return Message{}, fmt.Errorf("nil response")
	}
	message := Message{Role: "assistant", Content: resp.OutputText()}
	for i, item := range resp.Output {
		switch item.Type {
		case "message":
			for _, content := range item.Content {
				if content.Type == "refusal" && content.Refusal != "" && message.Content == "" {
					message.Content = content.Refusal
				}
			}
		case "function_call":
			call := item.AsFunctionCall()
			if strings.TrimSpace(call.CallID) == "" || strings.TrimSpace(call.Name) == "" {
				return Message{}, fmt.Errorf("output item %d function call requires call_id and name", i)
			}
			var args map[string]any
			argErr := ""
			if err := json.Unmarshal([]byte(call.Arguments), &args); err != nil {
				argErr = err.Error()
			}
			if args == nil {
				args = map[string]any{}
			}
			message.ToolCalls = append(message.ToolCalls, ToolCall{
				ID: call.CallID, Name: call.Name, Args: args,
				RawArgs: call.Arguments, ArgError: argErr,
			})
		case "reasoning":
			reasoning := item.AsReasoning()
			var summaries []string
			for _, summary := range reasoning.Summary {
				if summary.Text != "" {
					summaries = append(summaries, summary.Text)
				}
			}
			message.Reasoning = append(message.Reasoning, ReasoningBlock{
				Provider: "openai", Type: "reasoning", Text: strings.Join(summaries, "\n"),
				Data: reasoning.ID, EncryptedContent: reasoning.EncryptedContent,
			})
		default:
			return Message{}, fmt.Errorf("unsupported output item %d type %q", i, item.Type)
		}
	}
	return message, nil
}

func (c *OpenAI) storeResponseUsage(usage responses.ResponseUsage) {
	c.lastUsage.Store(&Usage{
		Provider:             c.Provider(),
		Model:                c.model,
		InputTokens:          int(usage.InputTokens),
		OutputTokens:         int(usage.OutputTokens),
		ReasoningTokens:      int(usage.OutputTokensDetails.ReasoningTokens),
		CachedTokens:         int(usage.InputTokensDetails.CachedTokens),
		CachedInputTokens:    int(usage.InputTokensDetails.CachedTokens),
		CacheReadInputTokens: int(usage.InputTokensDetails.CachedTokens),
	})
}
