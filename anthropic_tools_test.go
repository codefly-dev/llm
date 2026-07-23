package llm

import (
	"encoding/json"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
)

// These tests cover the Mind→SDK→Mind conversion helpers in anthropic.go.
// The SDK itself tests its JSON marshaling upstream; we only verify that
// our adapters produce the right SDK shape and unwrap responses cleanly.

func TestToAnthropicTools_ShapeAndRequired(t *testing.T) {
	tools := []ToolDef{
		{
			Name:        "read_file",
			Description: "Read a file",
			Parameters: []ToolParam{
				{Name: "path", Type: "string", Description: "File path", Required: true},
			},
		},
		{
			Name:        "search",
			Description: "Search for pattern",
			Parameters: []ToolParam{
				{Name: "pattern", Type: "string", Description: "Regex", Required: true},
				{Name: "max_results", Type: "integer", Description: "Limit", Required: false},
			},
		},
	}

	result := toAnthropicTools(tools)

	if len(result) != 2 {
		t.Fatalf("expected 2 tools, got %d", len(result))
	}
	if result[0].OfTool == nil || result[0].OfTool.Name != "read_file" {
		t.Fatalf("first tool not read_file: %+v", result[0])
	}
	if req := result[0].OfTool.InputSchema.Required; len(req) != 1 || req[0] != "path" {
		t.Errorf("required = %v, want [path]", req)
	}

	// Second tool: two properties, one required.
	props, ok := result[1].OfTool.InputSchema.Properties.(map[string]any)
	if !ok || len(props) != 2 {
		t.Errorf("second tool props = %v", result[1].OfTool.InputSchema.Properties)
	}
	if req := result[1].OfTool.InputSchema.Required; len(req) != 1 || req[0] != "pattern" {
		t.Errorf("second tool required = %v, want [pattern]", req)
	}
}

func TestToAnthropicTools_PreservesFullInputSchema(t *testing.T) {
	tools := []ToolDef{{
		Name:        "knowledge_query",
		Description: "query knowledge",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"intent": map[string]any{
					"type": "string",
					"enum": []string{"locate_files", "repair_recipe"},
				},
			},
			"required": []string{"intent"},
		},
	}}

	result := toAnthropicTools(tools)
	props, ok := result[0].OfTool.InputSchema.Properties.(map[string]any)
	if !ok {
		t.Fatalf("properties = %T, want map", result[0].OfTool.InputSchema.Properties)
	}
	intent := props["intent"].(map[string]any)
	enum := intent["enum"].([]any)
	if len(enum) != 2 || enum[0] != "locate_files" || enum[1] != "repair_recipe" {
		t.Fatalf("enum = %v, want full input schema enum", enum)
	}
	if req := result[0].OfTool.InputSchema.Required; len(req) != 1 || req[0] != "intent" {
		t.Fatalf("required = %v, want [intent]", req)
	}
}

func TestToAnthropicTools_PreservesStrictInputContract(t *testing.T) {
	result := toAnthropicTools([]ToolDef{
		{Name: "read_file", Description: "read", InputSchema: map[string]any{"type": "object"}},
		{Name: "final_response", Description: "finish", InputSchema: map[string]any{"type": "object"}, StrictInput: true},
	})

	if result[0].OfTool.Strict.Valid() {
		t.Fatalf("ordinary tool strict = %v, want omitted", result[0].OfTool.Strict)
	}
	if !result[1].OfTool.Strict.Valid() || !result[1].OfTool.Strict.Value {
		t.Fatalf("final_response strict = %v, want true", result[1].OfTool.Strict)
	}
}

func TestToAnthropicMessages_UserAssistantToolResult(t *testing.T) {
	messages := []Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Let me check", ToolCalls: []ToolCall{
			{ID: "tc_1", Name: "read_file", Args: map[string]any{"path": "main.go"}},
		}},
		{Role: "tool_result", ToolCallID: "tc_1", Content: "package main\n"},
	}

	result := toAnthropicMessages(messages)

	if len(result) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(result))
	}
	if result[0].Role != "user" {
		t.Errorf("msg[0].Role = %s, want user", result[0].Role)
	}
	if result[1].Role != "assistant" {
		t.Errorf("msg[1].Role = %s, want assistant", result[1].Role)
	}
	// Tool result maps to user role in Anthropic's protocol.
	if result[2].Role != "user" {
		t.Errorf("msg[2].Role = %s, want user (tool_result)", result[2].Role)
	}

	// Serialize the assistant turn and sanity-check the tool_use block.
	raw, err := json.Marshal(result[1])
	if err != nil {
		t.Fatalf("marshal assistant turn: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal assistant turn: %v", err)
	}
	content, _ := decoded["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("assistant content blocks = %d, want 2 (text + tool_use)", len(content))
	}
	toolBlock, _ := content[1].(map[string]any)
	if toolBlock["type"] != "tool_use" || toolBlock["id"] != "tc_1" || toolBlock["name"] != "read_file" {
		t.Errorf("tool_use block unexpected shape: %v", toolBlock)
	}
}

func TestAnthropicToolInputToMap(t *testing.T) {
	args, err := anthropicToolInputToMap(json.RawMessage(`{"path":"go.mod","lines":10}`))
	if err != nil {
		t.Fatal(err)
	}
	if args["path"] != "go.mod" {
		t.Errorf("path = %v", args["path"])
	}
	if v, _ := args["lines"].(float64); v != 10 {
		t.Errorf("lines = %v", args["lines"])
	}

	// Empty input — a tool_use block with no args is valid.
	empty, err := anthropicToolInputToMap(nil)
	if err != nil || len(empty) != 0 {
		t.Errorf("empty input: err=%v args=%v", err, empty)
	}
}

func TestAnthropicMessagePreservesThinkingBlocks(t *testing.T) {
	resp := &anthropic.Message{
		Content: []anthropic.ContentBlockUnion{
			{Type: "thinking", Thinking: "check the file", Signature: "sig_1"},
			{Type: "redacted_thinking", Data: "redacted"},
			{Type: "text", Text: "done"},
		},
	}

	msg := fromAnthropicMessage(resp)
	if msg.Content != "done" {
		t.Fatalf("Content = %q", msg.Content)
	}
	if len(msg.Reasoning) != 2 {
		t.Fatalf("Reasoning blocks = %d, want 2", len(msg.Reasoning))
	}
	if msg.Reasoning[0].Type != "thinking" || msg.Reasoning[0].Text != "check the file" || msg.Reasoning[0].Signature != "sig_1" {
		t.Fatalf("thinking block not preserved: %+v", msg.Reasoning[0])
	}
	if msg.Reasoning[1].Type != "redacted_thinking" || msg.Reasoning[1].Data != "redacted" {
		t.Fatalf("redacted block not preserved: %+v", msg.Reasoning[1])
	}

	roundTrip := toAnthropicMessages([]Message{msg})
	raw, err := json.Marshal(roundTrip[0])
	if err != nil {
		t.Fatal(err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	content := decoded["content"].([]any)
	if content[0].(map[string]any)["type"] != "thinking" {
		t.Fatalf("first block = %v, want thinking", content[0])
	}
	if content[1].(map[string]any)["type"] != "redacted_thinking" {
		t.Fatalf("second block = %v, want redacted_thinking", content[1])
	}
}
