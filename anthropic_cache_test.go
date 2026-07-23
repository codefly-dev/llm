package llm

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// THE TRAP THIS LOCKS: anthropic.CacheControlEphemeralParam{} is the ZERO
// value of an `omitzero` field — it silently serializes to NOTHING, so every
// cache_control marker built that way never reached the API. Prompt caching
// was dead project-wide (cache_write=0 on every recorded run) until the
// constructor form landed; a live probe confirmed write 2624 → read 2624 the
// moment NewCacheControlEphemeralParam() was used. This test fails if anyone
// reintroduces the zero-value form in the message/tool builders.
func TestAnthropicCacheControlActuallySerializes(t *testing.T) {
	// The zero-value form is invisible on the wire — the trap.
	zero, err := json.Marshal(anthropic.TextBlockParam{Text: "x", CacheControl: anthropic.CacheControlEphemeralParam{}})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(zero), "cache_control") {
		t.Fatalf("SDK behavior changed: zero-value CacheControl now serializes (%s) — revisit the constructor guidance", zero)
	}

	// Our message builder must produce blocks whose markers survive marshal.
	msgs := toAnthropicMessages([]Message{
		{Role: "user", Content: "first"},
		{Role: "assistant", Content: "ack"},
		{Role: "user", Content: "second"},
	})
	raw, err := json.Marshal(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(raw), `"cache_control"`) {
		t.Fatalf("toAnthropicMessages produced no serialized cache_control marker:\n%s", raw)
	}

	// The tool builder marks the last tool; that marker must survive too.
	tools := toAnthropicTools([]ToolDef{{Name: "a", Description: "d"}, {Name: "b", Description: "d"}})
	rawTools, err := json.Marshal(tools)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(rawTools), `"cache_control"`) {
		t.Fatalf("toAnthropicTools produced no serialized cache_control marker:\n%s", rawTools)
	}
}
