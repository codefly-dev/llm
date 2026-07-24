package llm

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/openai/openai-go/responses"
)

func TestToOpenAIResponseTools_EmptyRequiredIsArray(t *testing.T) {
	tools := []ToolDef{{
		Name:        "code_diff",
		Description: "Return current diff",
		Parameters: []ToolParam{
			{Name: "path", Type: "string", Description: "optional path"},
		},
	}}

	raw, err := json.Marshal(toOpenAIResponseTools(tools)[0])
	if err != nil {
		t.Fatalf("marshal tool: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("decode tool: %v", err)
	}
	params := decoded["parameters"].(map[string]any)
	required, ok := params["required"].([]any)
	if !ok {
		t.Fatalf("required = %T %v, want JSON array", params["required"], params["required"])
	}
	if len(required) != 0 {
		t.Fatalf("required = %v, want empty array", required)
	}
}

func TestToOpenAIResponseTools_RequiredNames(t *testing.T) {
	tools := []ToolDef{{
		Name:        "read",
		Description: "Read file",
		Parameters: []ToolParam{
			{Name: "path", Type: "string", Required: true},
			{Name: "start_line", Type: "integer"},
		},
	}}

	raw, err := json.Marshal(toOpenAIResponseTools(tools)[0])
	if err != nil {
		t.Fatalf("marshal tool: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("decode tool: %v", err)
	}
	params := decoded["parameters"].(map[string]any)
	required := params["required"].([]any)
	if len(required) != 1 || required[0] != "path" {
		t.Fatalf("required = %v, want [path]", required)
	}
}

func TestToOpenAIResponseTools_PreservesFullInputSchema(t *testing.T) {
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

	raw, err := json.Marshal(toOpenAIResponseTools(tools)[0])
	if err != nil {
		t.Fatalf("marshal tool: %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("decode tool: %v", err)
	}
	params := decoded["parameters"].(map[string]any)
	props := params["properties"].(map[string]any)
	intent := props["intent"].(map[string]any)
	enum := intent["enum"].([]any)
	if len(enum) != 2 || enum[0] != "locate_files" || enum[1] != "repair_recipe" {
		t.Fatalf("enum = %v, want full input schema enum", enum)
	}
	if got := params["additionalProperties"]; got != false {
		t.Fatalf("additionalProperties = %v, want false", got)
	}
}

func TestValidateToolDefsRejectsNonCanonicalTypes(t *testing.T) {
	for _, alias := range []string{"str", "int", "float", "double", "bool", "list", "json", "map"} {
		t.Run(alias, func(t *testing.T) {
			tools := []ToolDef{{Name: "typed_tool", Parameters: []ToolParam{{Name: "value", Type: alias}}}}
			if err := ValidateToolDefs(tools); err == nil {
				t.Fatalf("non-canonical type %q was accepted", alias)
			}
		})
	}
}

func TestFromOpenAIResponse_InvalidToolArgsArePreserved(t *testing.T) {
	var response responses.Response
	if err := json.Unmarshal([]byte(`{
		"id":"resp_1",
		"status":"completed",
		"output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"read_file","arguments":"{\"path\":"}]
	}`), &response); err != nil {
		t.Fatal(err)
	}
	msg, err := fromOpenAIResponse(&response)
	if err != nil {
		t.Fatal(err)
	}

	if len(msg.ToolCalls) != 1 {
		t.Fatalf("tool calls = %d, want 1", len(msg.ToolCalls))
	}
	call := msg.ToolCalls[0]
	if call.ArgError == "" {
		t.Fatal("ArgError should preserve the JSON parse failure")
	}
	if call.RawArgs != `{"path":` {
		t.Fatalf("RawArgs = %q", call.RawArgs)
	}
	if err := ValidateToolCall(call, []ToolDef{{
		Name: "read_file",
		Parameters: []ToolParam{{
			Name: "path", Type: "string", Required: true,
		}},
	}}); err == nil {
		t.Fatal("ValidateToolCall should reject invalid JSON args")
	}
}

func TestOpenAIResponseParamsUsesResponsesToolProtocol(t *testing.T) {
	params, err := openAIResponseParams(OpenAIModelGPT56Luna, "system", []Message{
		{Role: "user", Content: "read main.go"},
		{Role: "assistant", ToolCalls: []ToolCall{{ID: "call_1", Name: "read", Args: map[string]any{"path": "main.go"}}}},
		{Role: "tool_result", ToolCallID: "call_1", Content: "package main"},
	}, []ToolDef{{Name: "read", InputSchema: map[string]any{
		"type": "object", "properties": map[string]any{"path": map[string]any{"type": "string"}}, "required": []string{"path"},
	}}}, RequestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(params)
	if err != nil {
		t.Fatal(err)
	}
	text := string(raw)
	for _, want := range []string{`"model":"gpt-5.6-luna"`, `"instructions":"system"`, `"function_call"`, `"function_call_output"`, `"tools"`} {
		if !strings.Contains(text, want) {
			t.Fatalf("Responses request missing %s: %s", want, text)
		}
	}
	if strings.Contains(text, `"status"`) {
		t.Fatalf("Responses input leaked output-only reasoning status: %s", text)
	}
	if strings.Contains(text, `"reasoning_effort"`) {
		t.Fatalf("Chat Completions reasoning_effort leaked into Responses request: %s", text)
	}
}

func TestValidateToolCallRejectsUndeclaredTools(t *testing.T) {
	call := ToolCall{
		ID:   "call_1",
		Name: "edit",
		Args: map[string]any{"path": "main.go"},
	}
	if err := ValidateToolCall(call, nil); err == nil {
		t.Fatal("ValidateToolCall should reject tool calls when no tools were declared")
	}
}
