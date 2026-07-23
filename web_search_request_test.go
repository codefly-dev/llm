package llm

import (
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
)

// applyAnthropicOptions must attach Anthropic's HOSTED web_search server tool
// (web_search_20250305) when WebSearchMaxUses > 0, and leave the tool list
// untouched otherwise. This is the request-builder half of native web search;
// the end-to-end live behavior is covered by the cassette test in pkg/tools.
func TestApplyAnthropicOptions_AttachesHostedWebSearch(t *testing.T) {
	var params anthropic.MessageNewParams
	applyAnthropicOptions(&params, RequestOptions{WebSearchMaxUses: 4})

	if len(params.Tools) != 1 {
		t.Fatalf("tools len = %d, want 1 hosted web_search tool", len(params.Tools))
	}
	ws := params.Tools[0].OfWebSearchTool20250305
	if ws == nil {
		t.Fatal("expected OfWebSearchTool20250305 to be set")
	}
	if got := ws.MaxUses.Value; got != 4 {
		t.Fatalf("max_uses = %d, want 4", got)
	}
}

// When WebSearchMaxUses is 0 the hosted tool must NOT be attached, and any
// pre-set custom function tools must survive unchanged.
func TestApplyAnthropicOptions_NoHostedSearchByDefault(t *testing.T) {
	params := anthropic.MessageNewParams{Tools: toAnthropicTools([]ToolDef{{Name: "list_files"}})}
	applyAnthropicOptions(&params, RequestOptions{})

	if len(params.Tools) != 1 {
		t.Fatalf("tools len = %d, want the single custom tool preserved", len(params.Tools))
	}
	if params.Tools[0].OfWebSearchTool20250305 != nil {
		t.Fatal("hosted web_search must not be attached when WebSearchMaxUses == 0")
	}
	if params.Tools[0].OfTool == nil || params.Tools[0].OfTool.Name != "list_files" {
		t.Fatal("custom function tool must be preserved")
	}
}
