package llm

import (
	"strings"
	"testing"

	"github.com/benbjohnson/clock"
)

func TestVerifyRecorderConsumedRejectsAndThenClearsStaleRecording(t *testing.T) {
	dir := t.TempDir()
	model := "openai/gpt-test"
	persistRecorderTestEntry(t, dir, model, RecorderDefaultRunID, "used prompt")
	persistRecorderTestEntry(t, dir, model, RecorderDefaultRunID, "stale prompt")

	client := RecorderMiddleware(dir, model, RecordReplayOnly)(nil)
	if _, err := client.Call(t.Context(), "used prompt"); err != nil {
		t.Fatal(err)
	}
	if err := VerifyRecorderConsumed(client); err == nil ||
		!strings.Contains(err.Error(), "consumed 0, recorded 1") {
		t.Fatalf("stale recording verification error = %v", err)
	}

	if _, err := client.Call(t.Context(), "stale prompt"); err != nil {
		t.Fatal(err)
	}
	if err := VerifyRecorderConsumed(client); err != nil {
		t.Fatalf("fully consumed cassette rejected: %v", err)
	}
}

func TestVerifyRecorderConsumedScopesNamedRuns(t *testing.T) {
	dir := t.TempDir()
	model := "openai/gpt-test"
	persistRecorderTestEntry(t, dir, model, RecorderDefaultRunID, "default prompt")
	persistRecorderTestEntry(t, dir, model, "variance-1", "named prompt")

	defaultClient := RecorderMiddleware(dir, model, RecordReplayOnly)(nil)
	if _, err := defaultClient.Call(t.Context(), "default prompt"); err != nil {
		t.Fatal(err)
	}
	if err := VerifyRecorderConsumed(defaultClient); err != nil {
		t.Fatalf("default run claimed a named sibling: %v", err)
	}

	namedClient := RecorderMiddlewareWithRunID(dir, model, RecordReplayOnly, "variance-1")(nil)
	if _, err := namedClient.Call(t.Context(), "named prompt"); err != nil {
		t.Fatal(err)
	}
	if err := VerifyRecorderConsumed(namedClient); err != nil {
		t.Fatalf("named run verification failed: %v", err)
	}
}

func TestVerifyRecorderConsumedAggregatesModelRouterClients(t *testing.T) {
	dir := t.TempDir()
	cheapModel := "openai/gpt-cheap"
	strongModel := "openai/gpt-strong"
	persistRecorderTestEntry(t, dir, cheapModel, RecorderDefaultRunID, "classify")
	persistRecorderTestEntry(t, dir, strongModel, RecorderDefaultRunID, "execute")

	cheap := RecorderMiddleware(dir, cheapModel, RecordReplayOnly)(nil)
	strong := RecorderMiddleware(dir, strongModel, RecordReplayOnly)(nil)
	if _, err := cheap.Call(t.Context(), "classify"); err != nil {
		t.Fatal(err)
	}
	if _, err := strong.Call(t.Context(), "execute"); err != nil {
		t.Fatal(err)
	}
	if err := VerifyRecorderConsumed(cheap, strong); err != nil {
		t.Fatalf("router-wide cassette verification failed: %v", err)
	}
}

// persistRecorderTestEntry exercises the production recorder's atomic JSON
// writer against a real filesystem. Model behavior is outside this test: the
// persisted response exists only to validate cassette ownership and lifecycle.
func persistRecorderTestEntry(t *testing.T, dir, model, runID, prompt string) {
	t.Helper()
	recorder := &recorderMW{
		dir: dir, model: model, mode: RecordAlways, runID: runID,
		clock: clock.NewMock(),
	}
	path := recorder.recordingPath(hashKey(model, prompt) + ".json")
	if err := recorder.record(path, hashKey(model, prompt), prompt, "recorded response", nil); err != nil {
		t.Fatal(err)
	}
}
