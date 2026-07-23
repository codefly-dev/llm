package llm

import (
	"context"
	"testing"
)

func TestToolRecorderOnMissReplaysHitsAndRecordsMisses(t *testing.T) {
	dir := t.TempDir()
	existingArgs := map[string]any{"path": "existing.go"}
	record := NewToolRecorder(&healToolExecutor{response: "recorded"}, dir, RecordAlways, nil)
	if _, err := record.Call(t.Context(), "read_file", existingArgs); err != nil {
		t.Fatal(err)
	}

	healInner := &healToolExecutor{response: "healed"}
	heal := NewToolRecorder(healInner, dir, RecordOnMiss, nil)
	existing, err := heal.Call(t.Context(), "read_file", existingArgs)
	if err != nil {
		t.Fatalf("replay existing: %v", err)
	}
	if existing != "recorded" || healInner.calls != 0 {
		t.Fatalf("existing=%q calls=%d, want recorded/0", existing, healInner.calls)
	}
	missing, err := heal.Call(t.Context(), "read_file", map[string]any{"path": "missing.go"})
	if err != nil {
		t.Fatalf("record miss: %v", err)
	}
	if missing != "healed" || healInner.calls != 1 {
		t.Fatalf("missing=%q calls=%d, want healed/1", missing, healInner.calls)
	}
}

type healToolExecutor struct {
	calls    int
	response string
}

func (e *healToolExecutor) Call(context.Context, string, map[string]any) (string, error) {
	e.calls++
	return e.response, nil
}
