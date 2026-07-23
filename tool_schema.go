package llm

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
)

func toolInputSchema(t ToolDef) map[string]any {
	if len(t.InputSchema) > 0 {
		schema := cloneToolSchema(t.InputSchema)
		if _, ok := schema["type"]; !ok {
			schema["type"] = "object"
		}
		if _, ok := schema["additionalProperties"]; !ok {
			schema["additionalProperties"] = false
		}
		return schema
	}
	props := map[string]any{}
	required := []string{}
	for _, p := range t.Parameters {
		props[p.Name] = map[string]any{
			"type":        NormalizeToolParamType(p.Type),
			"description": p.Description,
		}
		if p.Required {
			required = append(required, p.Name)
		}
	}
	return map[string]any{
		"type":                 "object",
		"properties":           props,
		"required":             required,
		"additionalProperties": false,
	}
}

func cloneToolSchema(in map[string]any) map[string]any {
	raw, err := json.Marshal(in)
	if err != nil {
		panic("clone tool schema: " + err.Error())
	}
	var out map[string]any
	if err := json.Unmarshal(raw, &out); err != nil {
		panic("clone tool schema: " + err.Error())
	}
	return out
}

func schemaProperties(schema map[string]any) map[string]any {
	props, _ := schema["properties"].(map[string]any)
	if props == nil {
		return map[string]any{}
	}
	return props
}

func schemaRequired(schema map[string]any) []string {
	switch raw := schema["required"].(type) {
	case nil:
		return nil
	case []string:
		out := make([]string, len(raw))
		copy(out, raw)
		return out
	case []any:
		out := make([]string, 0, len(raw))
		for _, item := range raw {
			if s, ok := item.(string); ok {
				out = append(out, s)
			}
		}
		return out
	default:
		return nil
	}
}

func schemaExtraFields(schema map[string]any, skip ...string) map[string]any {
	skipped := make(map[string]bool, len(skip))
	for _, key := range skip {
		skipped[key] = true
	}
	out := map[string]any{}
	for key, value := range schema {
		if skipped[key] {
			continue
		}
		out[key] = value
	}
	return out
}

// ── Tool-call validation ────────────────────────────────────────

var toolNamePattern = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]{0,63}$`)

// ValidateToolDefs checks that every tool (and its parameters) has a legal,
// unique name and a supported parameter type before the defs reach a provider.
func ValidateToolDefs(tools []ToolDef) error {
	seenTools := map[string]struct{}{}
	for _, tool := range tools {
		if !toolNamePattern.MatchString(tool.Name) {
			return fmt.Errorf("tool %q: invalid name", tool.Name)
		}
		if _, ok := seenTools[tool.Name]; ok {
			return fmt.Errorf("tool %q: duplicate name", tool.Name)
		}
		seenTools[tool.Name] = struct{}{}

		seenParams := map[string]struct{}{}
		for _, p := range tool.Parameters {
			if !toolNamePattern.MatchString(p.Name) {
				return fmt.Errorf("tool %q param %q: invalid name", tool.Name, p.Name)
			}
			if _, ok := seenParams[p.Name]; ok {
				return fmt.Errorf("tool %q param %q: duplicate name", tool.Name, p.Name)
			}
			seenParams[p.Name] = struct{}{}
			if !validToolParamType(p.Type) {
				return fmt.Errorf("tool %q param %q: unsupported type %q", tool.Name, p.Name, p.Type)
			}
		}
	}
	return nil
}

// ValidateToolCall checks a model-produced tool call against the tool defs:
// known tool, valid JSON args, all required params present, no unknown params,
// and each argument's runtime type matches the declared parameter type.
func ValidateToolCall(call ToolCall, tools []ToolDef) error {
	if call.ArgError != "" {
		return fmt.Errorf("tool %q args are invalid JSON: %s", call.Name, call.ArgError)
	}
	if len(tools) == 0 {
		return fmt.Errorf("unknown tool %q", call.Name)
	}
	def, ok := findToolDef(call.Name, tools)
	if !ok {
		return fmt.Errorf("unknown tool %q", call.Name)
	}
	if call.Args == nil {
		call.Args = map[string]any{}
	}

	params := map[string]ToolParam{}
	for _, p := range def.Parameters {
		params[p.Name] = p
		if p.Required {
			if _, ok := call.Args[p.Name]; !ok {
				return fmt.Errorf("tool %q missing required argument %q", call.Name, p.Name)
			}
		}
	}

	for name, value := range call.Args {
		p, ok := params[name]
		if !ok {
			return fmt.Errorf("tool %q has unknown argument %q", call.Name, name)
		}
		if err := validateArgType(value, p.Type); err != nil {
			return fmt.Errorf("tool %q argument %q: %w", call.Name, name, err)
		}
	}
	return nil
}

func findToolDef(name string, tools []ToolDef) (ToolDef, bool) {
	for _, t := range tools {
		if t.Name == name {
			return t, true
		}
	}
	return ToolDef{}, false
}

func validToolParamType(t string) bool {
	return NormalizeToolParamType(t) != ""
}

func validateArgType(value any, want string) error {
	want = NormalizeToolParamType(want)
	if want == "" {
		return fmt.Errorf("unsupported type")
	}
	if value == nil {
		return fmt.Errorf("got null, want %s", want)
	}
	switch want {
	case "string":
		if _, ok := value.(string); ok {
			return nil
		}
	case "integer":
		switch v := value.(type) {
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64:
			return nil
		case float64:
			if math.Trunc(v) == v {
				return nil
			}
		case float32:
			if math.Trunc(float64(v)) == float64(v) {
				return nil
			}
		}
	case "number":
		switch value.(type) {
		case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
			return nil
		}
	case "boolean":
		if _, ok := value.(bool); ok {
			return nil
		}
	case "array":
		switch value.(type) {
		case []any, []string, []int, []float64:
			return nil
		}
	case "object":
		if _, ok := value.(map[string]any); ok {
			return nil
		}
	}
	return fmt.Errorf("got %s, want %s", typeName(value), want)
}

// NormalizeToolParamType accepts only canonical JSON Schema primitive names.
// Schema authors must fix invalid types at the source; provider adapters never
// reinterpret aliases differently.
func NormalizeToolParamType(t string) string {
	switch strings.ToLower(strings.TrimSpace(t)) {
	case "string":
		return "string"
	case "integer":
		return "integer"
	case "number":
		return "number"
	case "boolean":
		return "boolean"
	case "array":
		return "array"
	case "object":
		return "object"
	default:
		return ""
	}
}

func typeName(v any) string {
	return strings.TrimPrefix(fmt.Sprintf("%T", v), "[]")
}
