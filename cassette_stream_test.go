package llm

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// writeCassette writes a pre-seeded recording file matching the hash that
// RecorderMiddleware will look up for (model, prompt).
func writeCassette(t *testing.T, dir, model, prompt, response string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	hash := hashKey(model, prompt)
	rec := recording{
		Model:         model,
		RunID:         RecorderDefaultRunID,
		PromptHash:    hash,
		PromptPreview: prompt,
		Response:      response,
		RecordedAt:    "2026-01-01T00:00:00Z",
	}
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	path := filepath.Join(dir, hash+".json")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
}

// newReplayClient builds a Client that replays from dir and NEVER calls out.
// The Stub inner ensures any cassette miss is a deterministic failure rather
// than a network call — exactly the contract for CI.
func newReplayClient(dir, model string) Client {
	inner := StubError("live call should not happen during replay")
	return Chain(inner, RecorderMiddleware(dir, model, RecordReplayOnly))
}

func TestCassette_Completion(t *testing.T) {
	dir := t.TempDir()
	const model = "test/structured-1"
	const prompt = "write me a structured reply"
	const response = "<alpha>thought</alpha><beta>final</beta>"
	writeCassette(t, dir, model, prompt, response)

	c := newReplayClient(dir, model)
	got, err := c.Call(context.Background(), prompt)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if got != response {
		t.Fatalf("Call got %q, want %q", got, response)
	}
}

func TestCassette_StreamReplaysAsSingleChunk(t *testing.T) {
	// RecorderMiddleware.Stream replays the stored full response via one
	// onChunk call — that's the documented contract. This test pins it.
	dir := t.TempDir()
	const model = "test/structured-2"
	const prompt = "stream me"
	const response = "abc def ghi"
	writeCassette(t, dir, model, prompt, response)

	c := newReplayClient(dir, model)

	var chunks []string
	full, err := c.Stream(context.Background(), prompt, func(d string) error {
		chunks = append(chunks, d)
		return nil
	})
	if err != nil {
		t.Fatalf("Stream: %v", err)
	}
	if full != response {
		t.Fatalf("full = %q, want %q", full, response)
	}
	if len(chunks) != 1 || chunks[0] != response {
		t.Fatalf("chunks = %v, expected single chunk with full response", chunks)
	}
}

func TestCassette_StreamBlocks(t *testing.T) {
	// End-to-end: cassette → Stream → schema → channel → structured blocks.
	dir := t.TempDir()
	const model = "test/structured-3"
	const prompt = "emit alpha and beta"
	const response = "<alpha>I considered X then Y</alpha><beta>pick Y</beta>"
	writeCassette(t, dir, model, prompt, response)

	c := newReplayClient(dir, model)
	schema := &XMLTagSchema{Tags: []string{"alpha", "beta"}}

	blocks, result := StreamBlocks(context.Background(), c, prompt, schema, 0)
	got := CollectBlocks(blocks)

	assembled := assembleByKind(got)
	if assembled["alpha"] != "I considered X then Y" {
		t.Fatalf("alpha = %q", assembled["alpha"])
	}
	if assembled["beta"] != "pick Y" {
		t.Fatalf("beta = %q", assembled["beta"])
	}

	// Every non-Done block must have a matching Done=true terminator.
	doneSeen := map[string]bool{}
	for _, b := range got {
		if b.Done {
			doneSeen[b.Kind] = true
		}
	}
	if !doneSeen["alpha"] || !doneSeen["beta"] {
		t.Fatalf("missing Done events: %v", doneSeen)
	}

	res := <-result
	if res.Err != nil {
		t.Fatalf("result.Err = %v", res.Err)
	}
	if res.Full != response {
		t.Fatalf("full = %q, want %q", res.Full, response)
	}
}

func TestCassette_MissIsError(t *testing.T) {
	// RecordReplayOnly mode must fail loudly when the cassette is missing —
	// silent fallthrough would mask drift between tests and recordings.
	dir := t.TempDir()
	c := newReplayClient(dir, "test/nocassette")

	if _, err := c.Call(context.Background(), "unknown prompt"); err == nil {
		t.Fatal("expected error on cassette miss in RecordReplayOnly mode")
	}
}
