package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Local is a model served behind an OpenAI-compatible chat endpoint that is
// not OpenAI's (Ollama, LM Studio, mlx_lm.server on this machine): free, for
// developing prompts and scoring them.
//
// It is its own provider rather than the OpenAI one pointed elsewhere,
// because what such servers need is not what OpenAI's models take: any
// model name, reasoning switched off for models that would otherwise think
// for a minute before each answer, and a schema-constrained answer where a
// model is asked to fill one strict tool. It is a Provider, so it sits behind the same
// recorder as any other model: tests replay its recorded answers.
//
// A tool call is answered through structured output: the one tool's input
// schema becomes the response's JSON schema, and the JSON that comes back is
// returned as that tool's call. Local models are far more reliable at
// filling a schema than at choosing to call a tool.
type Local struct {
	// BaseURL ends before /chat/completions (http://localhost:11434/v1).
	BaseURL string
	ModelID string
	// Thinking, when false, asks the model not to reason before answering
	// (reasoning_effort: none): on qwen3 that is 2 s instead of 70 s.
	Thinking bool
	HTTP     *http.Client

	mu   sync.Mutex
	last *Usage
}

// NewLocal is a local model with thinking off.
func NewLocal(baseURL, model string) *Local {
	return &Local{BaseURL: strings.TrimRight(baseURL, "/"), ModelID: model, HTTP: &http.Client{Timeout: 10 * time.Minute}}
}

func (l *Local) Provider() string { return "local" }
func (l *Local) Model() string    { return l.ModelID }

// LastCallUsage reports the last call's tokens and time (a local model
// costs nothing).
func (l *Local) LastCallUsage() *Usage {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.last
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// localDefaultMaxTokens bounds an answer when the caller sets no limit. A local
// model that starts repeating itself inside a JSON list never stops on its
// own: without a bound one call ran for minutes.
const localDefaultMaxTokens = 3000

// answer is the model's reply: text, or the tools it wants called.
type answer struct {
	Content   string `json:"content"`
	ToolCalls []struct {
		ID       string `json:"id"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	} `json:"tool_calls"`
}

func (l *Local) chat(ctx context.Context, messages []chatMessage, format map[string]any, maxTokens int) (string, error) {
	extra := map[string]any{}
	if format != nil {
		extra["response_format"] = format
	}
	reply, err := l.post(ctx, messages, extra, maxTokens)
	return reply.Content, err
}

// post sends one chat request. messages is anything that marshals to the
// OpenAI chat format; extra adds request fields (a response format, tools).
func (l *Local) post(ctx context.Context, messages any, extra map[string]any, maxTokens int) (answer, error) {
	if maxTokens <= 0 {
		maxTokens = localDefaultMaxTokens
	}
	body := map[string]any{"model": l.ModelID, "messages": messages, "temperature": 0, "stream": false, "max_tokens": maxTokens}
	if !l.Thinking {
		body["reasoning_effort"] = "none"
	}
	for key, value := range extra {
		body[key] = value
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return answer{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.BaseURL+"/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return answer{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	started := time.Now()
	resp, err := l.HTTP.Do(req)
	if err != nil {
		return answer{}, fmt.Errorf("local model %s at %s: %w", l.ModelID, l.BaseURL, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 16<<20))
	if err != nil {
		return answer{}, err
	}
	if resp.StatusCode != http.StatusOK {
		return answer{}, fmt.Errorf("local model %s answered %d: %s", l.ModelID, resp.StatusCode, strings.TrimSpace(string(raw[:min(len(raw), 300)])))
	}
	var parsed struct {
		Choices []struct {
			Message      answer `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(raw, &parsed); err != nil || len(parsed.Choices) == 0 {
		return answer{}, fmt.Errorf("local model %s: unreadable answer: %v", l.ModelID, err)
	}
	l.mu.Lock()
	l.last = &Usage{Provider: "local", Model: l.ModelID, Requests: 1, InputTokens: parsed.Usage.PromptTokens,
		OutputTokens: parsed.Usage.CompletionTokens, DurationMS: time.Since(started).Milliseconds()}
	l.mu.Unlock()
	if parsed.Choices[0].FinishReason == "length" {
		return answer{}, fmt.Errorf("local model %s: the answer was cut off at %d tokens (the model did not finish; it may be repeating itself)", l.ModelID, maxTokens)
	}
	return parsed.Choices[0].Message, nil
}

// callTools is a turn of a conversation in which the model chooses among
// tools: the OpenAI function-calling format, which Ollama and LM Studio
// both speak. The conversation so far carries the calls the model made and
// what each returned.
func (l *Local) callTools(ctx context.Context, system string, messages []Message, tools []ToolDef, maxTokens int) (Message, error) {
	chat := []map[string]any{{"role": "system", "content": system}}
	for _, m := range messages {
		switch {
		case m.ToolResult != nil:
			chat = append(chat, map[string]any{"role": "tool", "tool_call_id": m.ToolResult.CallID, "content": m.ToolResult.Content})
		case len(m.ToolCalls) > 0:
			var calls []map[string]any
			for _, c := range m.ToolCalls {
				arguments := c.RawArgs
				if arguments == "" {
					encoded, _ := json.Marshal(c.Args)
					arguments = string(encoded)
				}
				calls = append(calls, map[string]any{"id": c.ID, "type": "function",
					"function": map[string]any{"name": c.Name, "arguments": arguments}})
			}
			chat = append(chat, map[string]any{"role": "assistant", "content": m.Content, "tool_calls": calls})
		default:
			chat = append(chat, map[string]any{"role": m.Role, "content": m.Content})
		}
	}
	var offered []map[string]any
	for _, t := range tools {
		offered = append(offered, map[string]any{"type": "function",
			"function": map[string]any{"name": t.Name, "description": t.Description, "parameters": t.InputSchema}})
	}
	reply, err := l.post(ctx, chat, map[string]any{"tools": offered}, maxTokens)
	if err != nil {
		return Message{}, err
	}
	out := Message{Role: "assistant", Content: reply.Content}
	for i, c := range reply.ToolCalls {
		call := ToolCall{ID: c.ID, Name: c.Function.Name, RawArgs: c.Function.Arguments}
		if call.ID == "" {
			call.ID = fmt.Sprintf("call_%d", i+1)
		}
		if err := json.Unmarshal([]byte(c.Function.Arguments), &call.Args); err != nil {
			call.ArgError = err.Error()
		}
		out.ToolCalls = append(out.ToolCalls, call)
	}
	return out, nil
}

func (l *Local) Call(ctx context.Context, prompt string) (string, error) {
	return l.chat(ctx, []chatMessage{{Role: "user", Content: prompt}}, nil, 0)
}

func (l *Local) Stream(ctx context.Context, prompt string, onChunk func(string) error) (string, error) {
	out, err := l.Call(ctx, prompt)
	if err == nil && onChunk != nil {
		err = onChunk(out)
	}
	return out, err
}

// CallWithTools answers with a call to the one tool given, its arguments
// constrained to the tool's input schema.
func (l *Local) CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (Message, error) {
	return l.CallWithToolsOptions(ctx, system, messages, tools, RequestOptions{})
}

// CallWithToolsOptions is CallWithTools with the caller's token limit.
func (l *Local) CallWithToolsOptions(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error) {
	if len(tools) == 0 {
		return Message{}, fmt.Errorf("local model: no tool to call")
	}
	// One strict tool is a form to fill: the answer is constrained to its
	// schema. Anything else is a choice among tools.
	if len(tools) != 1 || !tools[0].StrictInput || tools[0].InputSchema == nil {
		return l.callTools(ctx, system, messages, tools, opts.MaxTokens)
	}
	chat := []chatMessage{{Role: "system", Content: system}}
	for _, m := range messages {
		chat = append(chat, chatMessage{Role: m.Role, Content: m.Content})
	}
	content, err := l.chat(ctx, chat, map[string]any{
		"type":        "json_schema",
		"json_schema": map[string]any{"name": tools[0].Name, "strict": true, "schema": tools[0].InputSchema},
	}, opts.MaxTokens)
	if err != nil {
		return Message{}, err
	}
	call := ToolCall{ID: "call_1", Name: tools[0].Name, RawArgs: content}
	if err := json.Unmarshal([]byte(content), &call.Args); err != nil {
		call.ArgError = err.Error()
	}
	return Message{Role: "assistant", ToolCalls: []ToolCall{call}}, nil
}
