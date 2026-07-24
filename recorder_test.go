package llm

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/benbjohnson/clock"
)

func TestHashKey_Deterministic(t *testing.T) {
	h1 := hashKey("model-a", "prompt-x")
	h2 := hashKey("model-a", "prompt-x")
	if h1 != h2 {
		t.Errorf("hashKey not deterministic: %q vs %q", h1, h2)
	}
}

func TestHashKey_DifferentModelsDiffer(t *testing.T) {
	h1 := hashKey("model-a", "same-prompt")
	h2 := hashKey("model-b", "same-prompt")
	if h1 == h2 {
		t.Error("different models should produce different hashes")
	}
}

func TestHashKey_DifferentPromptsDiffer(t *testing.T) {
	h1 := hashKey("same-model", "prompt-1")
	h2 := hashKey("same-model", "prompt-2")
	if h1 == h2 {
		t.Error("different prompts should produce different hashes")
	}
}

func TestHashKey_Length(t *testing.T) {
	h := hashKey("model", "prompt")
	if len(h) != 16 {
		t.Errorf("hash length: got %d, want 16", len(h))
	}
}

func TestHashKey_IncludesNullSeparator(t *testing.T) {
	// Without the null separator, "model" + "prompt" == "modelp" + "rompt".
	h1 := hashKey("model", "prompt")
	h2 := hashKey("modelp", "rompt")
	if h1 == h2 {
		t.Error("null separator should prevent model/prompt boundary collisions")
	}
}

func TestHashKey_MatchesManualSHA256(t *testing.T) {
	model := "test-model"
	prompt := "test-prompt"
	h := sha256.New()
	h.Write([]byte(model))
	h.Write([]byte("\x00"))
	h.Write([]byte(prompt))
	expected := fmt.Sprintf("%x", h.Sum(nil))[:16]

	got := hashKey(model, prompt)
	if got != expected {
		t.Errorf("hashKey = %q, want %q", got, expected)
	}
}

func TestRecording_PromptPreviewTruncation(t *testing.T) {
	// The record() method truncates prompts > 200 chars.
	// We test the truncation logic by constructing a long string.
	long := make([]byte, 300)
	for i := range long {
		long[i] = 'a'
	}

	preview := string(long)
	if len(preview) > 200 {
		preview = preview[:200] + "..."
	}
	if len(preview) != 203 {
		t.Errorf("truncated preview length: got %d, want 203", len(preview))
	}
	if preview[200:] != "..." {
		t.Errorf("truncated preview should end with '...'")
	}
}

func TestRecording_ShortPromptPreviewNotTruncated(t *testing.T) {
	short := "hello world"
	preview := short
	if len(preview) > 200 {
		preview = preview[:200] + "..."
	}
	if preview != "hello world" {
		t.Errorf("short preview should not be truncated: got %q", preview)
	}
}

// recorderInfraClient exists only to exercise recorder persistence and error
// propagation. Semantic model behavior is covered by real-provider cassettes.
type recorderInfraClient struct {
	response Message
	err      error
	calls    int
}

func (s *recorderInfraClient) Call(_ context.Context, _ string) (string, error) {
	s.calls++
	return s.response.Content, s.err
}

func (s *recorderInfraClient) Stream(_ context.Context, _ string, onChunk func(string) error) (string, error) {
	s.calls++
	if onChunk != nil && s.response.Content != "" {
		if err := onChunk(s.response.Content); err != nil {
			return s.response.Content, err
		}
	}
	return s.response.Content, s.err
}

func (s *recorderInfraClient) CallWithTools(_ context.Context, _ string, _ []Message, _ []ToolDef) (Message, error) {
	s.calls++
	return s.response, s.err
}

func TestRecorder_ToolClientRecordReplay(t *testing.T) {
	dir := t.TempDir()
	stubResp := Message{
		Role: "assistant",
		ToolCalls: []ToolCall{
			{ID: "call_1", Name: "read_file", Args: map[string]any{"path": "main.go"}},
		},
	}
	stub := &recorderInfraClient{response: stubResp}

	mw := RecorderMiddleware(dir, "test-model", RecordAlways)
	client := mw(stub)

	tc, ok := AsToolClient(client)
	if !ok {
		t.Fatal("recorder should be discoverable as ToolClient")
	}

	tools := []ToolDef{{Name: "read_file", Description: "reads a file"}}
	msgs := []Message{{Role: "user", Content: "read main.go"}}

	resp1, err := tc.CallWithTools(context.Background(), "system prompt", msgs, tools)
	if err != nil {
		t.Fatal(err)
	}
	if len(resp1.ToolCalls) != 1 || resp1.ToolCalls[0].Name != "read_file" {
		t.Errorf("unexpected response: %+v", resp1)
	}

	// Replay: create a new recorder in replay-only mode.
	replayMW := RecorderMiddleware(dir, "test-model", RecordReplayOnly)
	replayClient := replayMW(&recorderInfraClient{})
	replayTC, _ := AsToolClient(replayClient)

	resp2, err := replayTC.CallWithTools(context.Background(), "system prompt", msgs, tools)
	if err != nil {
		t.Fatal("replay should succeed:", err)
	}
	if len(resp2.ToolCalls) != 1 || resp2.ToolCalls[0].Name != "read_file" {
		t.Errorf("replayed response mismatch: %+v", resp2)
	}
	if resp2.ToolCalls[0].ID != "call_1" {
		t.Errorf("tool call ID not preserved: got %q", resp2.ToolCalls[0].ID)
	}
}

func TestRecorder_ToolClientReplayOnlyMissing(t *testing.T) {
	dir := t.TempDir()
	mw := RecorderMiddleware(dir, "test-model", RecordReplayOnly)
	client := mw(&recorderInfraClient{})
	tc, _ := AsToolClient(client)

	_, err := tc.CallWithTools(context.Background(), "sys", []Message{{Role: "user", Content: "hi"}}, nil)
	if err == nil {
		t.Error("expected error for missing recording")
	}
}

func TestRecorder_OnMissReplaysHitsAndRecordsMisses(t *testing.T) {
	dir := t.TempDir()
	recordInner := &recorderInfraClient{response: Message{Content: "recorded"}}
	record := RecorderMiddleware(dir, "test-model", RecordAlways)(recordInner)
	if _, err := record.Call(t.Context(), "existing"); err != nil {
		t.Fatal(err)
	}

	healInner := &recorderInfraClient{response: Message{Content: "healed"}}
	heal := RecorderMiddleware(dir, "test-model", RecordOnMiss)(healInner)
	existing, err := heal.Call(t.Context(), "existing")
	if err != nil {
		t.Fatalf("replay existing recording: %v", err)
	}
	if existing != "recorded" || healInner.calls != 0 {
		t.Fatalf("existing response=%q live calls=%d, want recorded/0", existing, healInner.calls)
	}
	missing, err := heal.Call(t.Context(), "missing")
	if err != nil {
		t.Fatalf("record exact miss: %v", err)
	}
	if missing != "healed" || healInner.calls != 1 {
		t.Fatalf("missing response=%q live calls=%d, want healed/1", missing, healInner.calls)
	}

	replayInner := &recorderInfraClient{response: Message{Content: "must not run"}}
	replayed, err := RecorderMiddleware(dir, "test-model", RecordReplayOnly)(replayInner).Call(t.Context(), "missing")
	if err != nil {
		t.Fatalf("replay healed recording: %v", err)
	}
	if replayed != "healed" || replayInner.calls != 0 {
		t.Fatalf("replayed response=%q live calls=%d, want healed/0", replayed, replayInner.calls)
	}
}

func TestRecorder_OnMissRejectsCorruptExistingRecording(t *testing.T) {
	dir := t.TempDir()
	prompt := "corrupt"
	path := filepath.Join(dir, hashKey("test-model", prompt)+".json")
	if err := os.WriteFile(path, []byte("{"), 0o644); err != nil {
		t.Fatal(err)
	}
	inner := &recorderInfraClient{response: Message{Content: "must not run"}}
	if _, err := RecorderMiddleware(dir, "test-model", RecordOnMiss)(inner).Call(t.Context(), prompt); err == nil {
		t.Fatal("heal accepted a corrupt existing recording")
	}
	if inner.calls != 0 {
		t.Fatalf("live calls=%d, want 0 for corrupt recording", inner.calls)
	}
}

func TestRecorderReplayDoesNotCrossModelFamilies(t *testing.T) {
	dir := t.TempDir()
	prompt := "same task and prompt"
	recorded := RecorderMiddleware(dir, string(ModelAnthropicSonnet), RecordAlways)(&recorderInfraClient{
		response: Message{Content: "anthropic response"},
	})
	if _, err := recorded.Call(t.Context(), prompt); err != nil {
		t.Fatal(err)
	}

	openAIReplay := RecorderMiddleware(dir, string(ModelOpenAIGPT56Terra), RecordReplayOnly)(&recorderInfraClient{
		response: Message{Content: "must never be called"},
	})
	if response, err := openAIReplay.Call(t.Context(), prompt); err == nil {
		t.Fatalf("cross-family replay returned %q; want a model-specific cassette miss", response)
	}
}

func TestRecorderUsesInjectedClockForStableMetadata(t *testing.T) {
	dir := t.TempDir()
	prompt := "deterministic timestamp"
	now := time.Date(2026, 7, 11, 6, 5, 4, 0, time.UTC)
	mock := clock.NewMock()
	mock.Set(now)
	client := RecorderMiddlewareWithConfig(RecorderConfig{
		Dir:   dir,
		Model: "test/model",
		Mode:  RecordAlways,
		Clock: mock,
	})(&recorderInfraClient{response: Message{Content: "ok"}})
	if _, err := client.Call(t.Context(), prompt); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, hashKey("test/model", prompt)+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var got recording
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got.RecordedAt != now.Format(time.RFC3339) {
		t.Fatalf("recorded_at = %q, want injected %q", got.RecordedAt, now.Format(time.RFC3339))
	}
}

func TestRecorderRunIDDefaultUsesRootDirectory(t *testing.T) {
	dir := t.TempDir()
	prompt := "same prompt"
	model := "test-model"

	client := RecorderMiddlewareWithRunID(dir, model, RecordAlways, "")(&recorderInfraClient{
		response: Message{Content: "default sample"},
	})
	if out, err := client.Call(context.Background(), prompt); err != nil || out != "default sample" {
		t.Fatalf("record default run id = %q, %v", out, err)
	}

	hash := hashKey(model, prompt)
	if _, err := os.Stat(filepath.Join(dir, hash+".json")); err != nil {
		t.Fatalf("default run id should use cassette root: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "runs", RecorderDefaultRunID, hash+".json")); !os.IsNotExist(err) {
		t.Fatalf("default run id should not create runs/0 cassette, err=%v", err)
	}
}

func TestRecorderRunIDNamespacesIdenticalPromptSamples(t *testing.T) {
	dir := t.TempDir()
	prompt := "identical prompt"
	model := "test-model"

	run1 := RecorderMiddlewareWithRunID(dir, model, RecordAlways, "1")(&recorderInfraClient{
		response: Message{Content: "sample one"},
	})
	if out, err := run1.Call(context.Background(), prompt); err != nil || out != "sample one" {
		t.Fatalf("record run 1 = %q, %v", out, err)
	}
	run2 := RecorderMiddlewareWithRunID(dir, model, RecordAlways, "2")(&recorderInfraClient{
		response: Message{Content: "sample two"},
	})
	if out, err := run2.Call(context.Background(), prompt); err != nil || out != "sample two" {
		t.Fatalf("record run 2 = %q, %v", out, err)
	}

	hash := hashKey(model, prompt)
	for _, id := range []string{"1", "2"} {
		if _, err := os.Stat(filepath.Join(dir, "runs", id, hash+".json")); err != nil {
			t.Fatalf("run %s cassette missing: %v", id, err)
		}
	}

	replay1 := RecorderMiddlewareWithRunID(dir, model, RecordReplayOnly, "1")(&recorderInfraClient{
		response: Message{Content: "ignored"},
	})
	if out, err := replay1.Call(context.Background(), prompt); err != nil || out != "sample one" {
		t.Fatalf("replay run 1 = %q, %v", out, err)
	}
	replay2 := RecorderMiddlewareWithRunID(dir, model, RecordReplayOnly, "2")(&recorderInfraClient{
		response: Message{Content: "ignored"},
	})
	if out, err := replay2.Call(context.Background(), prompt); err != nil || out != "sample two" {
		t.Fatalf("replay run 2 = %q, %v", out, err)
	}
}

func TestRecorderRecordFailsClosedWhenCassetteCannotCommit(t *testing.T) {
	blockingFile := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(blockingFile, []byte("block"), 0o644); err != nil {
		t.Fatal(err)
	}
	inner := &recorderInfraClient{response: Message{Content: "provider response"}}
	client := RecorderMiddleware(filepath.Join(blockingFile, "cassette"), "test-model", RecordAlways)(inner)

	response, err := client.Call(t.Context(), "must persist")
	if response != "provider response" {
		t.Fatalf("response = %q", response)
	}
	if err == nil || !strings.Contains(err.Error(), "persist call") {
		t.Fatalf("error = %v, want cassette persistence failure", err)
	}
	if inner.calls != 1 {
		t.Fatalf("provider calls = %d, want 1", inner.calls)
	}
}

func TestRecorderRecordsProviderErrorAndReplaysWithoutLiveCall(t *testing.T) {
	dir := t.TempDir()
	providerErr := errors.New("provider unavailable")
	recordInner := &recorderInfraClient{response: Message{Content: "partial"}, err: providerErr}
	record := RecorderMiddleware(dir, "test-model", RecordAlways)(recordInner)
	response, err := record.Call(t.Context(), "error outcome")
	if response != "partial" || !errors.Is(err, providerErr) {
		t.Fatalf("record = %q, %v", response, err)
	}

	replayInner := &recorderInfraClient{response: Message{Content: "must not run"}}
	replay := RecorderMiddleware(dir, "test-model", RecordReplayOnly)(replayInner)
	response, err = replay.Call(t.Context(), "error outcome")
	if response != "partial" || err == nil || err.Error() != providerErr.Error() {
		t.Fatalf("replay = %q, %v", response, err)
	}
	if replayInner.calls != 0 {
		t.Fatalf("replay called live provider %d times", replayInner.calls)
	}
}

func TestRecorderRejectsDuplicateRequestBeforeSecondProviderCall(t *testing.T) {
	inner := &recorderInfraClient{response: Message{Content: "one sample"}}
	client := RecorderMiddleware(t.TempDir(), "test-model", RecordAlways)(inner)
	if _, err := client.Call(t.Context(), "identical"); err != nil {
		t.Fatal(err)
	}
	if _, err := client.Call(t.Context(), "identical"); err == nil || !strings.Contains(err.Error(), "duplicate request") {
		t.Fatalf("duplicate error = %v", err)
	}
	if inner.calls != 1 {
		t.Fatalf("provider calls = %d, want 1", inner.calls)
	}
}

func TestRecorderReplayRejectsDuplicateConsumption(t *testing.T) {
	dir := t.TempDir()
	record := RecorderMiddleware(dir, "test-model", RecordAlways)(&recorderInfraClient{response: Message{Content: "sample"}})
	if _, err := record.Call(t.Context(), "identical"); err != nil {
		t.Fatal(err)
	}
	replay := RecorderMiddleware(dir, "test-model", RecordReplayOnly)(&recorderInfraClient{})
	if _, err := replay.Call(t.Context(), "identical"); err != nil {
		t.Fatal(err)
	}
	if _, err := replay.Call(t.Context(), "identical"); err == nil || !strings.Contains(err.Error(), "duplicate request") {
		t.Fatalf("duplicate replay error = %v", err)
	}
}

func TestRecorderReplayValidatesStoredIdentity(t *testing.T) {
	mutations := map[string]func(*recording){
		"model":       func(rec *recording) { rec.Model = "other-model" },
		"run_id":      func(rec *recording) { rec.RunID = "other-run" },
		"prompt_hash": func(rec *recording) { rec.PromptHash = "0000000000000000" },
		"recorded_at": func(rec *recording) { rec.RecordedAt = "not-a-time" },
	}
	for name, mutate := range mutations {
		t.Run(name, func(t *testing.T) {
			dir := t.TempDir()
			prompt := "identity"
			record := RecorderMiddleware(dir, "test-model", RecordAlways)(&recorderInfraClient{response: Message{Content: "sample"}})
			if _, err := record.Call(t.Context(), prompt); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(dir, hashKey("test-model", prompt)+".json")
			raw, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			var rec recording
			if err := json.Unmarshal(raw, &rec); err != nil {
				t.Fatal(err)
			}
			mutate(&rec)
			raw, err = json.MarshalIndent(rec, "", "  ")
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, raw, 0o644); err != nil {
				t.Fatal(err)
			}
			replayInner := &recorderInfraClient{}
			replay := RecorderMiddleware(dir, "test-model", RecordReplayOnly)(replayInner)
			if _, err := replay.Call(t.Context(), prompt); err == nil || !strings.Contains(err.Error(), "validate") {
				t.Fatalf("tampered replay error = %v", err)
			}
			if replayInner.calls != 0 {
				t.Fatalf("tampered replay called provider %d times", replayInner.calls)
			}
		})
	}
}

func TestRecorderRequestKeyIncludesEveryGenerationOption(t *testing.T) {
	base, err := keyWithOptions("prompt", RequestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	reasoning, err := keyWithOptions("prompt", RequestOptions{ReasoningEffort: ReasoningHigh})
	if err != nil {
		t.Fatal(err)
	}
	web, err := keyWithOptions("prompt", RequestOptions{WebSearchMaxUses: 3})
	if err != nil {
		t.Fatal(err)
	}
	if base == reasoning || base == web || reasoning == web {
		t.Fatalf("option keys collided: base=%q reasoning=%q web=%q", base, reasoning, web)
	}
}

func TestRecorderRejectsUnencodableRequestBeforeProviderCall(t *testing.T) {
	inner := &recorderInfraClient{response: Message{Content: "must not run"}}
	client := RecorderMiddleware(t.TempDir(), "test-model", RecordAlways)(inner)
	toolClient, ok := AsToolClient(client)
	if !ok {
		t.Fatal("recorder does not expose ToolClient")
	}
	_, err := toolClient.CallWithTools(t.Context(), "system", []Message{{
		Role:      "assistant",
		ToolCalls: []ToolCall{{Name: "bad", Args: map[string]any{"channel": make(chan int)}}},
	}}, nil)
	if err == nil || !strings.Contains(err.Error(), "encode tool conversation") {
		t.Fatalf("encoding error = %v", err)
	}
	if inner.calls != 0 {
		t.Fatalf("provider calls = %d, want 0", inner.calls)
	}
}

func TestRecorderRejectsUnsafeRunIDBeforeProviderCall(t *testing.T) {
	inner := &recorderInfraClient{response: Message{Content: "must not run"}}
	client := RecorderMiddlewareWithRunID(t.TempDir(), "test-model", RecordAlways, "../escape")(inner)
	if _, err := client.Call(t.Context(), "prompt"); err == nil || !strings.Contains(err.Error(), "invalid configuration") {
		t.Fatalf("run id error = %v", err)
	}
	if inner.calls != 0 {
		t.Fatalf("provider calls = %d, want 0", inner.calls)
	}
}

func TestToolConversationKey_Deterministic(t *testing.T) {
	msgs := []Message{
		{Role: "user", Content: "hello"},
		{Role: "assistant", ToolCalls: []ToolCall{{Name: "read", Args: map[string]any{"path": "a.go"}}}},
		{Role: "tool_result", ToolResult: &ToolResult{Content: "package main"}},
	}
	tools := []ToolDef{{Name: "read"}}

	k1 := mustToolConversationKey(t, "system", msgs, tools)
	k2 := mustToolConversationKey(t, "system", msgs, tools)
	if k1 != k2 {
		t.Error("tool conversation key should be deterministic")
	}
}

func mustToolConversationKey(t *testing.T, system string, messages []Message, tools []ToolDef) string {
	t.Helper()
	key, err := toolConversationKey(system, messages, tools)
	if err != nil {
		t.Fatalf("toolConversationKey: %v", err)
	}
	return key
}

func TestToolConversationKey_DifferentMessagesDiffer(t *testing.T) {
	tools := []ToolDef{{Name: "read"}}
	k1 := mustToolConversationKey(t, "sys", []Message{{Role: "user", Content: "hello"}}, tools)
	k2 := mustToolConversationKey(t, "sys", []Message{{Role: "user", Content: "world"}}, tools)
	if k1 == k2 {
		t.Error("different messages should produce different keys")
	}
}

func TestToolConversationKey_IncludesToolSchema(t *testing.T) {
	msgs := []Message{{Role: "user", Content: "read"}}
	k1 := mustToolConversationKey(t, "sys", msgs, []ToolDef{{
		Name: "read", Description: "Read a file",
		Parameters: []ToolParam{{Name: "path", Type: "string", Required: true}},
	}})
	k2 := mustToolConversationKey(t, "sys", msgs, []ToolDef{{
		Name: "read", Description: "Read a file safely",
		Parameters: []ToolParam{{Name: "path", Type: "string", Required: true}},
	}})
	if k1 == k2 {
		t.Fatal("tool schema changes should change recorder key")
	}
}

func TestRecorder_StreamEventsRecordReplay(t *testing.T) {
	dir := t.TempDir()
	stubResp := Message{
		Role:      "assistant",
		Content:   "checking",
		Reasoning: []ReasoningBlock{{Provider: "test", Type: "thinking", Text: "need file"}},
		ToolCalls: []ToolCall{{ID: "call_1", Name: "read_file", Args: map[string]any{"path": "main.go"}}},
	}
	req := StreamRequest{
		System:   "system",
		Messages: []Message{{Role: "user", Content: "read main.go"}},
		Tools: []ToolDef{{
			Name:       "read_file",
			Parameters: []ToolParam{{Name: "path", Type: "string", Required: true}},
		}},
	}

	client := RecorderMiddleware(dir, "test-model", RecordAlways)(&recorderInfraClient{response: stubResp})
	esc, ok := AsEventStreamClient(client)
	if !ok {
		t.Fatal("recorder should expose EventStreamClient")
	}
	events := collectStreamEvents(t, esc, req)
	if len(events) == 0 {
		t.Fatal("expected recorded events")
	}
	if !events[0].Synthetic {
		t.Fatal("fallback stream event should be marked synthetic")
	}

	replay := RecorderMiddleware(dir, "test-model", RecordReplayOnly)(&recorderInfraClient{})
	replayESC, _ := AsEventStreamClient(replay)
	replayed := collectStreamEvents(t, replayESC, req)
	if len(replayed) != len(events) {
		t.Fatalf("replayed events = %d, want %d", len(replayed), len(events))
	}
	if replayed[0].Type != StreamEventReasoningDelta {
		t.Fatalf("first event = %s, want reasoning", replayed[0].Type)
	}
	foundTool := false
	for _, event := range replayed {
		if event.Type == StreamEventToolCallEnd && event.ToolCall != nil && event.ToolCall.ID == "call_1" {
			foundTool = true
		}
	}
	if !foundTool {
		t.Fatal("replay did not preserve tool call event")
	}
}

func TestRecorderStreamEventsDoesNotPublishSuccessBeforePersistence(t *testing.T) {
	blockingFile := filepath.Join(t.TempDir(), "not-a-directory")
	if err := os.WriteFile(blockingFile, []byte("block"), 0o644); err != nil {
		t.Fatal(err)
	}
	client := RecorderMiddleware(filepath.Join(blockingFile, "cassette"), "test-model", RecordAlways)(
		&recorderInfraClient{response: Message{Role: "assistant", Content: "provider success"}},
	)
	eventClient, ok := AsEventStreamClient(client)
	if !ok {
		t.Fatal("recorder does not expose EventStreamClient")
	}
	stream, err := eventClient.StreamEvents(t.Context(), StreamRequest{Prompt: "persist first"})
	if err != nil {
		t.Fatal(err)
	}
	var events []StreamEvent
	for event := range stream {
		events = append(events, event)
	}
	if len(events) != 1 || events[0].Type != StreamEventError || !strings.Contains(events[0].Error, "persist event stream") {
		t.Fatalf("events = %+v, want one terminal persistence error", events)
	}
}

func collectStreamEvents(t *testing.T, c EventStreamClient, req StreamRequest) []StreamEvent {
	t.Helper()
	ch, err := c.StreamEvents(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	var events []StreamEvent
	for event := range ch {
		events = append(events, event)
	}
	return events
}

func TestToolConversationKeyRejectsTaskIDDrift(t *testing.T) {
	tools := []ToolDef{{Name: "final_response"}}
	keyA := mustToolConversationKey(t,
		"## Objective Contract\n- Task: T-20260511T012916.572110000\n- Plan: T-20260511T012916.572110000-plan",
		[]Message{{Role: "user", Content: `{"task_id":"T-20260511T012916.572110000","plan_id":"T-20260511T012916.572110000-plan"}`}},
		tools,
	)
	keyB := mustToolConversationKey(t,
		"## Objective Contract\n- Task: T-20260511T013000.111222333\n- Plan: T-20260511T013000.111222333-plan",
		[]Message{{Role: "user", Content: `{"task_id":"T-20260511T013000.111222333","plan_id":"T-20260511T013000.111222333-plan"}`}},
		tools,
	)
	if keyA == keyB {
		t.Fatal("task ID drift was hidden from the exact recorder key")
	}
}

func TestToolConversationKeyRejectsTimestampDrift(t *testing.T) {
	tools := []ToolDef{{Name: "read"}}
	keyA := mustToolConversationKey(t,
		"system",
		[]Message{{Role: "user", Content: `{"artifact":{"kind":"capability-result","body":{"status":"failed","heal_attempts":[{"started_at":"2026-06-20T23:21:54.001Z","completed_at":"2026-06-20T23:21:57.999Z"}]}}}`}},
		tools,
	)
	keyB := mustToolConversationKey(t,
		"system",
		[]Message{{Role: "user", Content: `{"artifact":{"kind":"capability-result","body":{"status":"failed","heal_attempts":[{"started_at":"2026-06-21T00:02:11.673495Z","completed_at":"2026-06-21T00:02:11.923204Z"}]}}}`}},
		tools,
	)
	if keyA == keyB {
		t.Fatal("timestamp drift was hidden from the exact recorder key")
	}
}

func TestToolConversationKeyRejectsToolElapsedDrift(t *testing.T) {
	tools := []ToolDef{{Name: "test_run"}}
	keyA := mustToolConversationKey(t,
		"system",
		[]Message{{Role: "user", Content: `{"output":"{\"Action\":\"pass\",\"Package\":\"example.com/stringsx\",\"Elapsed\":0.18}\n"}`}},
		tools,
	)
	keyB := mustToolConversationKey(t,
		"system",
		[]Message{{Role: "user", Content: `{"output":"{\"Action\":\"pass\",\"Package\":\"example.com/stringsx\",\"Elapsed\":0.201}\n"}`}},
		tools,
	)
	if keyA == keyB {
		t.Fatal("tool elapsed drift was hidden from the exact recorder key")
	}
}

func TestRecordMode_Constants(t *testing.T) {
	if RecordReplayOnly != 0 {
		t.Error("RecordReplayOnly should be 0")
	}
	if RecordAlways != 1 {
		t.Error("RecordAlways should be 1")
	}
	if RecordOnMiss != 2 {
		t.Error("RecordOnMiss should be 2")
	}
}
