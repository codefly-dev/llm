package llm

import (
	"context"
	"sync"
)

// transportProbeClient is a minimal in-package Client stub for middleware and
// stream tests: it records call count and echoes a fixed response.
type transportProbeClient struct {
	mu       sync.Mutex
	response string
	calls    int
}

func newTransportProbe(response string) *transportProbeClient {
	return &transportProbeClient{response: response}
}

func (c *transportProbeClient) Call(context.Context, string) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	return c.response, nil
}

func (c *transportProbeClient) Stream(_ context.Context, _ string, onChunk func(string) error) (string, error) {
	c.mu.Lock()
	c.calls++
	response := c.response
	c.mu.Unlock()
	if onChunk != nil {
		if err := onChunk(response); err != nil {
			return response, err
		}
	}
	return response, nil
}

func (c *transportProbeClient) CallCached(context.Context, string, string) (string, error) {
	return c.Call(context.Background(), "")
}

func (c *transportProbeClient) CallWithTools(context.Context, string, []Message, []ToolDef) (Message, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls++
	return Message{Role: "assistant", Content: c.response}, nil
}

// assembleByKind collapses a parsed block stream into kind -> concatenated
// delta text, dropping the terminal Done markers.
func assembleByKind(blocks []Block) map[string]string {
	out := map[string]string{}
	for _, b := range blocks {
		if b.Done {
			continue
		}
		out[b.Kind] += b.Delta
	}
	return out
}
