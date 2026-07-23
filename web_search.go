package llm

// ARCHITECTURE: provider-NATIVE web search backed by Anthropic's HOSTED
// web_search server tool. This is the "use the LLM key Mind already holds"
// path for the single `web_search` tool — no Tavily/SerpAPI key required.
//
// The model runs the search server-side at Anthropic. The response carries:
//   - text blocks       → the model's synthesized answer (with citations)
//   - web_search_tool_result blocks → the sources (title + url) it consulted
//
// Usage.ServerToolUse.WebSearchRequests is billed on the LLM key and already
// flows through storeUsage → the cost middleware (PriceComponentWebSearchRequest).
//
// pkg/tools defines the WebSearchProvider seam and a tiny NativeWebSearcher
// interface; this type implements that interface WITHOUT pkg/tools importing
// pkg/llm (the provider depends only on the interface). The server wires an
// AnthropicWebSearcher (built from the strong llm.Client) ahead of the
// credential-driven Tavily/SerpAPI providers.

import (
	"context"
	"fmt"
	"strings"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
)

// HostedWebSearcher is the capability interface for provider-HOSTED web search.
// *Anthropic implements it. A middleware-wrapped Client (Chain) does not, so
// FindWebSearcher unwraps the chain to reach the underlying provider — the same
// pattern findUsageProvider uses for cost accounting.
type HostedWebSearcher interface {
	WebSearch(ctx context.Context, query string, maxResults int) (WebSearchResult, error)
}

// FindWebSearcher walks through middleware wrappers to find a client that
// supports provider-hosted web search. Returns nil if none does (e.g. an
// OpenAI provider without the hosted tool, or a replay-only stub).
func FindWebSearcher(c Client) HostedWebSearcher {
	if ws, ok := c.(HostedWebSearcher); ok {
		return ws
	}
	if u, ok := c.(interface{ Unwrap() Client }); ok {
		return FindWebSearcher(u.Unwrap())
	}
	return nil
}

// WebSearchSource is one source the hosted search consulted: a title + URL.
type WebSearchSource struct {
	Title string
	URL   string
}

// WebSearchResult is the parsed outcome of a hosted web-search Complete:
// the model's synthesized answer plus the sources it consulted.
type WebSearchResult struct {
	Answer  string
	Sources []WebSearchSource
}

// WebSearch runs a single hosted-web-search turn: it asks the model to search
// the web for the query and returns the synthesized answer + the sources the
// search server returned. maxUses caps the number of server-side searches.
//
// This is non-streaming because the web_search_tool_result blocks are only
// cleanly available on a fully-formed message.
func (c *Anthropic) WebSearch(ctx context.Context, query string, maxResults int) (WebSearchResult, error) {
	if strings.TrimSpace(query) == "" {
		return WebSearchResult{}, fmt.Errorf("anthropic web search: empty query")
	}
	maxUses := maxResults
	if maxUses <= 0 {
		maxUses = 5
	}
	prompt := fmt.Sprintf(
		"Search the web for: %s\nReturn the key findings as a concise summary, and cite the source URLs you used.",
		query,
	)
	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(c.model),
		MaxTokens: 2048,
		Messages: []anthropic.MessageParam{
			anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)),
		},
		Tools: []anthropic.ToolUnionParam{{
			OfWebSearchTool20250305: &anthropic.WebSearchTool20250305Param{
				MaxUses: param.NewOpt(int64(maxUses)),
			},
		}},
	}

	resp, err := c.sdk.Messages.New(ctx, params)
	if err != nil {
		return WebSearchResult{}, c.classify(err)
	}
	c.storeUsage(resp.Usage)

	return extractWebSearchResult(resp), nil
}

// extractWebSearchResult walks a hosted-search response, concatenating text
// blocks into the synthesized answer and collecting sources from every
// web_search_tool_result block. Sources are de-duplicated by URL.
func extractWebSearchResult(resp *anthropic.Message) WebSearchResult {
	var answer strings.Builder
	var sources []WebSearchSource
	seen := map[string]bool{}

	for _, block := range resp.Content {
		switch block.Type {
		case "text":
			answer.WriteString(block.Text)
		case "web_search_tool_result":
			for _, r := range block.Content.AsWebSearchResultBlockArray() {
				if r.URL == "" || seen[r.URL] {
					continue
				}
				seen[r.URL] = true
				sources = append(sources, WebSearchSource{Title: r.Title, URL: r.URL})
			}
		}
	}
	return WebSearchResult{Answer: strings.TrimSpace(answer.String()), Sources: sources}
}
