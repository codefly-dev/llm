package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A fake OpenAI-compatible endpoint: checks the request, answers JSON.
func TestLocalAnswersATooLCallThroughStructuredOutput(t *testing.T) {
	var got map[string]any
	finish := "stop"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"finish_reason": finish, "message": map[string]any{"content": `{"operations":[{"record":"client"}]}`}}},
			"usage":   map[string]any{"prompt_tokens": 100, "completion_tokens": 12},
		})
	}))
	defer srv.Close()

	local := NewLocal(srv.URL+"/v1/", "qwen-test")
	tools := []ToolDef{{Name: "fill_records", StrictInput: true, InputSchema: map[string]any{"type": "object"}}}
	msg, err := local.CallWithToolsOptions(context.Background(), "be exact", []Message{{Role: "user", Content: "the call"}}, tools, RequestOptions{MaxTokens: 700})
	if err != nil {
		t.Fatal(err)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Name != "fill_records" || !strings.Contains(msg.ToolCalls[0].RawArgs, `"client"`) {
		t.Fatalf("the JSON answer comes back as the tool's call: %+v", msg)
	}
	format, _ := got["response_format"].(map[string]any)
	if got["model"] != "qwen-test" || got["max_tokens"] != float64(700) || got["reasoning_effort"] != "none" || got["temperature"] != float64(0) || format["type"] != "json_schema" {
		t.Fatalf("request: %v", got)
	}
	if u := local.LastCallUsage(); u == nil || u.InputTokens != 100 || u.OutputTokens != 12 || u.CostUSD != 0 {
		t.Fatalf("usage: %+v", u)
	}
	// No limit given: a default one is still sent, so a model that repeats
	// itself cannot run forever.
	if _, err := local.CallWithTools(context.Background(), "s", []Message{{Role: "user", Content: "u"}}, tools); err != nil || got["max_tokens"] != float64(localDefaultMaxTokens) {
		t.Fatalf("default limit: %v %v", err, got["max_tokens"])
	}
	// An answer cut off at the limit is an error, not half a JSON document.
	finish = "length"
	if _, err := local.CallWithTools(context.Background(), "s", []Message{{Role: "user", Content: "u"}}, tools); err == nil || !strings.Contains(err.Error(), "cut off") {
		t.Fatalf("a truncated answer must be an error: %v", err)
	}
	if _, err := local.CallWithTools(context.Background(), "s", nil, nil); err == nil {
		t.Fatal("no tool, no structured answer")
	}
}

// With several tools, or one that is not strict, the model chooses: the
// OpenAI function-calling format, with the conversation's earlier calls and
// what each returned.
func TestLocalLetsTheModelChooseAmongTools(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &got)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"finish_reason": "tool_calls", "message": map[string]any{
				"content": "",
				"tool_calls": []map[string]any{{"id": "call_9", "type": "function",
					"function": map[string]any{"name": "ApplyOperations", "arguments": `{"operations":[]}`}}},
			}}},
			"usage": map[string]any{"prompt_tokens": 50, "completion_tokens": 9},
		})
	}))
	defer srv.Close()

	local := NewLocal(srv.URL+"/v1", "qwen-test")
	tools := []ToolDef{
		{Name: "FindClients", Description: "Finds clients.", InputSchema: map[string]any{"type": "object"}},
		{Name: "ApplyOperations", Description: "Writes.", InputSchema: map[string]any{"type": "object"}},
	}
	history := []Message{
		{Role: "user", Content: "the call"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_1", Name: "FindClients", Args: map[string]any{"phone": "5550142"}}}},
		{Role: "tool_result", ToolCallID: "call_1", Content: `{"clients":[]}`, ToolResult: &ToolResult{CallID: "call_1", Content: `{"clients":[]}`}},
	}
	msg, err := local.CallWithToolsOptions(context.Background(), "be exact", history, tools, RequestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].ID != "call_9" || msg.ToolCalls[0].Name != "ApplyOperations" || msg.ToolCalls[0].Args["operations"] == nil {
		t.Fatalf("the model's call: %+v", msg)
	}
	if _, constrained := got["response_format"]; constrained {
		t.Fatal("a choice among tools is not a constrained answer")
	}
	offered, _ := got["tools"].([]any)
	sent, _ := got["messages"].([]any)
	if len(offered) != 2 || len(sent) != 4 {
		t.Fatalf("request: %d tools, %d messages", len(offered), len(sent))
	}
	called, _ := sent[2].(map[string]any)
	returned, _ := sent[3].(map[string]any)
	if called["role"] != "assistant" || called["tool_calls"] == nil || returned["role"] != "tool" || returned["tool_call_id"] != "call_1" || returned["content"] != `{"clients":[]}` {
		t.Fatalf("the earlier call and its result, in the OpenAI format: %v %v", called, returned)
	}
}
