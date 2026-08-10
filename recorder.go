package llm

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/benbjohnson/clock"
	"github.com/codefly-dev/core/wool"
	"github.com/codefly-dev/llm/apierror"
	"go.opentelemetry.io/otel/attribute"
	oteltrace "go.opentelemetry.io/otel/trace"
)

// RecordMode and its constants live in pkg/spec; this file uses the
// aliases re-exported from pkg/llm.

// recording is the JSON structure stored on disk.
type recording struct {
	Model             string            `json:"model"`
	TransportIdentity TransportIdentity `json:"transport_identity,omitempty"`
	RunID             string            `json:"run_id,omitempty"`
	PromptHash        string            `json:"prompt_hash"`
	PromptPreview     string            `json:"prompt_preview"`
	Request           string            `json:"request,omitempty"`
	Response          string            `json:"response"`
	TypedResponse     json.RawMessage   `json:"typed_response,omitempty"`
	ToolResponse      json.RawMessage   `json:"tool_response,omitempty"`
	Events            []StreamEvent     `json:"events,omitempty"`
	Usage             *Usage            `json:"usage,omitempty"`
	Error             *recordedFailure  `json:"error,omitempty"`
	RecordedAt        string            `json:"recorded_at"`
}

// recordedFailure preserves the machine-readable provider classification that
// controls retry and terminal behavior. Error text alone is not replay-safe:
// rebuilding it with errors.New makes quota/auth failures look transient.
type recordedFailure struct {
	Message       string                 `json:"message"`
	ProviderError *recordedProviderError `json:"provider_error,omitempty"`
}

type recordedProviderError struct {
	Provider     string             `json:"provider"`
	Code         apierror.ErrorCode `json:"code"`
	Status       int                `json:"status,omitempty"`
	Message      string             `json:"message"`
	RequestID    string             `json:"request_id,omitempty"`
	RetryAfterNS int64              `json:"retry_after_ns,omitempty"`
}

type recordingKind string

const (
	recordingText   recordingKind = "text"
	recordingTool   recordingKind = "tool"
	recordingEvents recordingKind = "events"
)

const RecorderDefaultRunID = "0"

// RecorderMiddleware returns a middleware that caches LLM responses on disk.
// The cache key is SHA256(model + prompt). Recordings are stored as JSON files
// in dir. Typically dir is "testdata/llm_recordings" relative to the test file.
func RecorderMiddleware(dir string, model string, mode RecordMode) Middleware {
	return RecorderMiddlewareWithConfig(RecorderConfig{Dir: dir, Model: model, Mode: mode})
}

// RecorderConfig makes every cassette-affecting dependency explicit. Clock is
// injectable so recording metadata is byte-stable under replay/fixture tests;
// nil selects a real clock for live recording.
type RecorderConfig struct {
	Dir               string
	Model             string
	TransportIdentity TransportIdentity
	Mode              RecordMode
	RunID             string
	Clock             clock.Clock
}

func RecorderMiddlewareWithConfig(cfg RecorderConfig) Middleware {
	recorderClock := cfg.Clock
	if recorderClock == nil {
		recorderClock = clock.New()
	}
	runID, configErr := canonicalRecorderRunID(cfg.RunID)
	if strings.TrimSpace(cfg.Dir) == "" {
		configErr = errors.Join(configErr, fmt.Errorf("recorder directory is empty"))
	}
	if strings.TrimSpace(cfg.Model) == "" {
		configErr = errors.Join(configErr, fmt.Errorf("recorder model is empty"))
	}
	if cfg.Mode != RecordReplayOnly && cfg.Mode != RecordAlways && cfg.Mode != RecordOnMiss {
		configErr = errors.Join(configErr, fmt.Errorf("invalid recorder mode %d", cfg.Mode))
	}
	return func(next Client) Client {
		return &recorderMW{
			next:              next,
			dir:               cfg.Dir,
			model:             cfg.Model,
			transportIdentity: cfg.TransportIdentity,
			mode:              cfg.Mode,
			runID:             runID,
			clock:             recorderClock,
			configErr:         configErr,
		}
	}
}

// RecorderMiddlewareWithRunID records the same semantic prompt hash under a
// run/sample namespace. Run id "0" is the default and intentionally uses the
// legacy dir layout so existing cassettes replay unchanged. Non-zero ids write
// under dir/runs/<id>/..., which lets a harness loop collect multiple live
// examples for the exact same calls without overwriting prior samples.
func RecorderMiddlewareWithRunID(dir string, model string, mode RecordMode, runID string) Middleware {
	return RecorderMiddlewareWithConfig(RecorderConfig{Dir: dir, Model: model, Mode: mode, RunID: runID})
}

type recorderMW struct {
	next              Client
	dir               string
	model             string
	transportIdentity TransportIdentity
	mode              RecordMode
	runID             string
	clock             clock.Clock
	configErr         error
	usageMu           sync.Mutex
	lastUsage         *Usage
	// liveMu keeps each recorded request paired with the provider's
	// LastCallUsage value. Providers expose usage as last-call state, so live
	// recording cannot safely overlap calls on one client.
	liveMu sync.Mutex
	// writeMu serializes recording writes: parallel-safe tool batches drive
	// concurrent LLM/tool calls through ONE recorder, and two goroutines
	// hitting the same hash raced on the cassette file (review B8).
	writeMu sync.Mutex
	seenMu  sync.Mutex
	seen    map[string]struct{}
}

func (m *recorderMW) Call(ctx context.Context, prompt string) (string, error) {
	return m.CallWithOptions(ctx, prompt, RequestOptions{})
}

func (m *recorderMW) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error) {
	m.setLastUsage(nil)
	if m.configErr != nil {
		return "", fmt.Errorf("recorder: invalid configuration: %w", m.configErr)
	}
	keyInput, err := keyWithOptions(prompt, opts)
	if err != nil {
		return "", fmt.Errorf("recorder: encode call request: %w", err)
	}
	hash := hashKey(m.model, keyInput)
	path := m.recordingPath(hash + ".json")
	span := oteltrace.SpanFromContext(ctx)

	// Try replay.
	if m.mode != RecordAlways {
		resp, replayErr := m.replay(path, hash, recordingText)
		if replayErr == nil {
			span.SetAttributes(attribute.String("llm.recorder", "replay"))
			m.dumpDebugKey("replay", hash, keyInput)
			m.setLastUsage(resp.Usage)
			return resp.Response, recordedError(resp)
		}
		if m.mode == RecordReplayOnly || !errors.Is(replayErr, os.ErrNotExist) {
			return "", m.replayError("call", hash, replayErr)
		}
	}

	// Call the real provider.
	if err := m.claim(path); err != nil {
		return "", err
	}
	m.liveMu.Lock()
	defer m.liveMu.Unlock()
	span.SetAttributes(attribute.String("llm.recorder", "record"))
	resp, callErr := CallWithOptions(ctx, m.next, prompt, opts)
	if writeErr := m.record(path, hash, keyInput, resp, callErr); writeErr != nil {
		return resp, errors.Join(callErr, fmt.Errorf("recorder: persist call %s: %w", hash, writeErr))
	}
	return resp, callErr
}

func (m *recorderMW) Stream(ctx context.Context, prompt string, onChunk func(text string) error) (string, error) {
	return m.StreamWithOptions(ctx, prompt, RequestOptions{}, onChunk)
}

func (m *recorderMW) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(text string) error) (string, error) {
	m.setLastUsage(nil)
	if m.configErr != nil {
		return "", fmt.Errorf("recorder: invalid configuration: %w", m.configErr)
	}
	keyInput, err := keyWithOptions(prompt, opts)
	if err != nil {
		return "", fmt.Errorf("recorder: encode stream request: %w", err)
	}
	hash := hashKey(m.model, keyInput)
	path := m.recordingPath(hash + ".json")
	span := oteltrace.SpanFromContext(ctx)

	if m.mode != RecordAlways {
		rec, replayErr := m.replay(path, hash, recordingText)
		if replayErr == nil {
			span.SetAttributes(attribute.String("llm.recorder", "replay"))
			m.dumpDebugKey("replay", hash, keyInput)
			m.setLastUsage(rec.Usage)
			if onChunk != nil && rec.Response != "" {
				if callbackErr := onChunk(rec.Response); callbackErr != nil {
					return rec.Response, callbackErr
				}
			}
			return rec.Response, recordedError(rec)
		}
		if m.mode == RecordReplayOnly || !errors.Is(replayErr, os.ErrNotExist) {
			return "", m.replayError("stream", hash, replayErr)
		}
	}

	if err := m.claim(path); err != nil {
		return "", err
	}
	m.liveMu.Lock()
	defer m.liveMu.Unlock()
	span.SetAttributes(attribute.String("llm.recorder", "record"))
	var callbackErr error
	full, streamErr := StreamWithOptions(ctx, m.next, prompt, opts, func(text string) error {
		if onChunk == nil {
			return nil
		}
		callbackErr = onChunk(text)
		return callbackErr
	})
	if callbackErr != nil {
		m.release(path)
		return full, callbackErr
	}
	if writeErr := m.record(path, hash, keyInput, full, streamErr); writeErr != nil {
		return full, errors.Join(streamErr, fmt.Errorf("recorder: persist stream %s: %w", hash, writeErr))
	}
	return full, streamErr
}

func (m *recorderMW) CallCached(ctx context.Context, system, user string) (string, error) {
	return m.CallCachedWithOptions(ctx, system, user, RequestOptions{})
}

func (m *recorderMW) CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (string, error) {
	m.setLastUsage(nil)
	if m.configErr != nil {
		return "", fmt.Errorf("recorder: invalid configuration: %w", m.configErr)
	}
	combined := system + "\n\n" + user
	keyInput, err := keyWithOptions(combined, opts)
	if err != nil {
		return "", fmt.Errorf("recorder: encode cached request: %w", err)
	}
	hash := hashKey(m.model, keyInput)
	path := m.recordingPath(hash + ".json")
	span := oteltrace.SpanFromContext(ctx)

	if m.mode != RecordAlways {
		rec, replayErr := m.replay(path, hash, recordingText)
		if replayErr == nil {
			span.SetAttributes(attribute.String("llm.recorder", "replay"))
			m.setLastUsage(rec.Usage)
			return rec.Response, recordedError(rec)
		}
		if m.mode == RecordReplayOnly || !errors.Is(replayErr, os.ErrNotExist) {
			return "", m.replayError("cached_call", hash, replayErr)
		}
	}

	if err := m.claim(path); err != nil {
		return "", err
	}
	m.liveMu.Lock()
	defer m.liveMu.Unlock()
	span.SetAttributes(attribute.String("llm.recorder", "record"))
	resp, callErr := CallCachedWithOptions(ctx, m.next, system, user, opts)
	if writeErr := m.record(path, hash, keyInput, resp, callErr); writeErr != nil {
		return resp, errors.Join(callErr, fmt.Errorf("recorder: persist cached call %s: %w", hash, writeErr))
	}
	return resp, callErr
}

// CallWithTools records and replays multi-turn tool conversations.
// The cache key is derived from model + system + serialized messages + tool names.
func (m *recorderMW) CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (Message, error) {
	return m.CallWithToolsOptions(ctx, system, messages, tools, RequestOptions{})
}

func (m *recorderMW) CallWithToolsOptions(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error) {
	m.setLastUsage(nil)
	if m.configErr != nil {
		return Message{}, fmt.Errorf("recorder: invalid configuration: %w", m.configErr)
	}
	conversationKey, err := toolConversationKey(system, messages, tools)
	if err != nil {
		return Message{}, fmt.Errorf("recorder: encode tool conversation: %w", err)
	}
	keyInput, err := keyWithOptions(conversationKey, opts)
	if err != nil {
		return Message{}, fmt.Errorf("recorder: encode tool options: %w", err)
	}
	hash := hashKey(m.model, keyInput)
	path := m.recordingPath("tools_" + hash + ".json")
	span := oteltrace.SpanFromContext(ctx)

	if m.mode != RecordAlways {
		rec, replayErr := m.replay(path, hash, recordingTool)
		if replayErr == nil {
			var msg Message
			if err := json.Unmarshal(rec.ToolResponse, &msg); err != nil {
				return Message{}, fmt.Errorf("recorder: decode tool response %s: %w", hash, err)
			}
			span.SetAttributes(attribute.String("llm.recorder", "replay"))
			m.dumpDebugKey("replay", hash, keyInput)
			m.setLastUsage(rec.Usage)
			return msg, recordedError(rec)
		}
		if m.mode == RecordReplayOnly || !errors.Is(replayErr, os.ErrNotExist) {
			m.dumpDebugKey("miss", hash, keyInput)
			return Message{}, m.replayError("tool_call", hash, replayErr)
		}
	}

	if err := m.claim(path); err != nil {
		return Message{}, err
	}
	m.liveMu.Lock()
	defer m.liveMu.Unlock()
	span.SetAttributes(attribute.String("llm.recorder", "record"))
	resp, callErr := CallWithToolsOptions(ctx, m.next, system, messages, tools, opts)
	if writeErr := m.recordToolCall(path, hash, keyInput, resp, callErr); writeErr != nil {
		return resp, errors.Join(callErr, fmt.Errorf("recorder: persist tool call %s: %w", hash, writeErr))
	}
	m.dumpDebugKey("record", hash, keyInput)

	return resp, callErr
}

// dumpDebugKey writes the recorder's key input next to the cassette
// dir for byte-diffing record vs replay divergence. Gated by
// MIND_RECORDER_DEBUG=1; pairs "_miss_<hash>.key.txt" (replay branch)
// with "_record_<hash>.key.txt" or "_replay_<hash>.key.txt" so a single
// `diff` reveals exactly which turn drifted. Test-suite-only diagnostic;
// silently no-ops outside debug mode.
func (m *recorderMW) dumpDebugKey(phase, hash, keyInput string) {
	if !recorderDebugEnabled() {
		return
	}
	dumpPath := m.recordingPath("_" + phase + "_" + hash + ".key.txt")
	if err := os.WriteFile(dumpPath, []byte(keyInput), 0o644); err == nil && phase == "miss" {
		wool.Get(context.Background()).In("llm.recorder").Warn("recorder miss: dumped key input", wool.Field("hash", hash), wool.Field("path", dumpPath))
	}
}

// StreamEvents records and replays normalized event streams. The recording key
// includes the full tool descriptors so cassette replay notices schema changes.
func (m *recorderMW) StreamEvents(ctx context.Context, req StreamRequest) (<-chan StreamEvent, error) {
	m.setLastUsage(nil)
	if m.configErr != nil {
		return nil, fmt.Errorf("recorder: invalid configuration: %w", m.configErr)
	}
	keyInput, err := eventStreamKey(req)
	if err != nil {
		return nil, fmt.Errorf("recorder: encode event stream request: %w", err)
	}
	hash := hashKey(m.model, keyInput)
	path := m.recordingPath("events_" + hash + ".json")
	span := oteltrace.SpanFromContext(ctx)

	if m.mode != RecordAlways {
		rec, replayErr := m.replay(path, hash, recordingEvents)
		if replayErr == nil {
			span.SetAttributes(attribute.String("llm.recorder", "replay"))
			m.setLastUsage(rec.Usage)
			out := make(chan StreamEvent, len(rec.Events))
			go func() {
				defer close(out)
				for _, event := range rec.Events {
					if err := sendRecorderEvent(ctx, out, event); err != nil {
						return
					}
				}
			}()
			return out, nil
		}
		if m.mode == RecordReplayOnly || !errors.Is(replayErr, os.ErrNotExist) {
			return nil, m.replayError("event_stream", hash, replayErr)
		}
	}

	if err := m.claim(path); err != nil {
		return nil, err
	}
	m.liveMu.Lock()
	span.SetAttributes(attribute.String("llm.recorder", "record"))
	source, err := StreamEventsFromClient(ctx, m.next, req)
	if err != nil {
		m.liveMu.Unlock()
		m.release(path)
		return nil, err
	}
	out := make(chan StreamEvent, 16)
	go func() {
		defer close(out)
		defer m.liveMu.Unlock()
		var events []StreamEvent
		for event := range source {
			events = append(events, event)
		}
		if writeErr := m.recordEvents(path, hash, keyInput, events); writeErr != nil {
			_ = sendRecorderEvent(ctx, out, StreamEvent{
				Type:      StreamEventError,
				Error:     fmt.Sprintf("recorder: persist event stream %s: %v", hash, writeErr),
				Synthetic: true,
			})
			return
		}
		for _, event := range events {
			if err := sendRecorderEvent(ctx, out, event); err != nil {
				return
			}
		}
	}()
	return out, nil
}

func (m *recorderMW) recordToolCall(path, hash, keyInput string, resp Message, callErr error) error {
	toolResp, err := json.Marshal(resp)
	if err != nil {
		return fmt.Errorf("marshal tool response: %w", err)
	}
	rec := m.newRecording(hash, keyInput)
	rec.ToolResponse = toolResp
	rec.Error = captureRecordedFailure(callErr)
	return m.writeRecording(path, rec)
}

// toolConversationKey produces a deterministic string from the conversation state.
// encoding/json sorts string map keys; unlike the previous helper, this
// function propagates unsupported-value errors instead of hashing empty data.
func deterministicJSON(v any) ([]byte, error) {
	return json.Marshal(v)
}

func toolConversationKey(system string, messages []Message, tools []ToolDef) (string, error) {
	key, err := json.Marshal(struct {
		System   string    `json:"system"`
		Messages []Message `json:"messages"`
		Tools    []ToolDef `json:"tools"`
	}{
		System: system, Messages: messages, Tools: tools,
	})
	if err != nil {
		return "", err
	}
	return string(key), nil
}

func eventStreamKey(req StreamRequest) (string, error) {
	key, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	return string(key), nil
}

func keyWithOptions(base string, opts RequestOptions) (string, error) {
	if opts == (RequestOptions{}) {
		return base, nil
	}
	encoded, err := deterministicJSON(opts)
	if err != nil {
		return "", err
	}
	return base + "\x00options:" + string(encoded), nil
}

func (m *recorderMW) Unwrap() Client { return m.next }

func (m *recorderMW) LastCallUsage() *Usage {
	m.usageMu.Lock()
	defer m.usageMu.Unlock()
	if m.lastUsage == nil {
		return nil
	}
	usage := *m.lastUsage
	return &usage
}

func (m *recorderMW) setLastUsage(usage *Usage) {
	m.usageMu.Lock()
	defer m.usageMu.Unlock()
	if usage == nil {
		m.lastUsage = nil
		return
	}
	copy := *usage
	m.lastUsage = &copy
}

func (m *recorderMW) replay(path, hash string, kind recordingKind) (recording, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return recording{}, fmt.Errorf("read %s: %w", path, err)
	}
	var rec recording
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&rec); err != nil {
		return recording{}, fmt.Errorf("decode %s: %w", path, err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return recording{}, fmt.Errorf("decode %s: trailing JSON content", path)
	}
	if err := m.validateRecording(rec, hash, kind); err != nil {
		return recording{}, fmt.Errorf("validate %s: %w", path, err)
	}
	if err := m.claim(path); err != nil {
		return recording{}, err
	}
	return rec, nil
}

func (m *recorderMW) record(path, hash, prompt, response string, callErr error) error {
	rec := m.newRecording(hash, prompt)
	rec.Response = response
	if response != "" {
		typedResponse := []byte(response)
		if !json.Valid(typedResponse) {
			typedResponse, _ = json.Marshal(response)
		}
		rec.TypedResponse = typedResponse
	}
	rec.Error = captureRecordedFailure(callErr)
	return m.writeRecording(path, rec)
}

func (m *recorderMW) recordEvents(path, hash, keyInput string, events []StreamEvent) error {
	if len(events) == 0 {
		return fmt.Errorf("event stream completed without events")
	}
	rec := m.newRecording(hash, keyInput)
	rec.Events = events
	return m.writeRecording(path, rec)
}

func (m *recorderMW) newRecording(hash, keyInput string) recording {
	preview := keyInput
	if len(preview) > 200 {
		preview = preview[:200] + "..."
	}
	usage := latestClientUsage(m.next)
	m.setLastUsage(usage)
	return recording{
		Model:             m.model,
		TransportIdentity: m.transportIdentity,
		RunID:             m.runID,
		PromptHash:        hash,
		PromptPreview:     preview,
		Request:           keyInput,
		Usage:             usage,
		RecordedAt:        m.clock.Now().UTC().Format(time.RFC3339Nano),
	}
}

func latestClientUsage(client Client) *Usage {
	for client != nil {
		if provider, ok := client.(UsageProvider); ok {
			if usage := provider.LastCallUsage(); usage != nil {
				copy := *usage
				return &copy
			}
		}
		unwrapper, ok := client.(interface{ Unwrap() Client })
		if !ok {
			return nil
		}
		client = unwrapper.Unwrap()
	}
	return nil
}

func (m *recorderMW) writeRecording(path string, rec recording) (retErr error) {
	m.writeMu.Lock()
	defer m.writeMu.Unlock()
	defer func() {
		if retErr != nil {
			m.release(path)
		}
	}()

	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create directory %s: %w", dir, err)
	}
	data, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal recording: %w", err)
	}
	temporary, err := os.CreateTemp(dir, "."+filepath.Base(path)+".tmp-*")
	if err != nil {
		return fmt.Errorf("create temporary recording: %w", err)
	}
	temporaryPath := temporary.Name()
	defer func() { _ = os.Remove(temporaryPath) }()
	if err := temporary.Chmod(0o644); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("chmod temporary recording: %w", err)
	}
	if _, err := temporary.Write(data); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("write temporary recording: %w", err)
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return fmt.Errorf("sync temporary recording: %w", err)
	}
	if err := temporary.Close(); err != nil {
		return fmt.Errorf("close temporary recording: %w", err)
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		return fmt.Errorf("commit recording %s: %w", path, err)
	}
	return nil
}

func (m *recorderMW) validateRecording(rec recording, hash string, kind recordingKind) error {
	if rec.Model != m.model {
		return fmt.Errorf("model = %q, want %q", rec.Model, m.model)
	}
	if rec.TransportIdentity != "" && rec.TransportIdentity != m.transportIdentity {
		return fmt.Errorf("transport_identity = %q, want %q", rec.TransportIdentity, m.transportIdentity)
	}
	if rec.RunID != m.runID {
		return fmt.Errorf("run_id = %q, want %q", rec.RunID, m.runID)
	}
	if rec.PromptHash != hash {
		return fmt.Errorf("prompt_hash = %q, want %q", rec.PromptHash, hash)
	}
	if _, err := time.Parse(time.RFC3339Nano, rec.RecordedAt); err != nil {
		return fmt.Errorf("recorded_at %q: %w", rec.RecordedAt, err)
	}
	switch kind {
	case recordingText:
		if rec.ToolResponse != nil || rec.Events != nil {
			return fmt.Errorf("text recording contains a non-text response")
		}
	case recordingTool:
		if rec.ToolResponse == nil || rec.Events != nil || rec.Response != "" || rec.TypedResponse != nil {
			return fmt.Errorf("tool recording has an invalid response shape")
		}
	case recordingEvents:
		if len(rec.Events) == 0 || rec.ToolResponse != nil || rec.Response != "" || rec.TypedResponse != nil {
			return fmt.Errorf("event recording has an invalid response shape")
		}
	default:
		return fmt.Errorf("unknown recording kind %q", kind)
	}
	return nil
}

func (m *recorderMW) claim(path string) error {
	m.seenMu.Lock()
	defer m.seenMu.Unlock()
	if m.seen == nil {
		m.seen = make(map[string]struct{})
	}
	if _, exists := m.seen[path]; exists {
		return fmt.Errorf("recorder: duplicate request %s cannot be represented by a content-addressed cassette", path)
	}
	m.seen[path] = struct{}{}
	return nil
}

func (m *recorderMW) release(path string) {
	m.seenMu.Lock()
	delete(m.seen, path)
	m.seenMu.Unlock()
}

func recordedError(rec recording) error {
	if rec.Error == nil {
		return nil
	}
	if rec.Error.ProviderError == nil {
		return errors.New(rec.Error.Message)
	}
	provider := rec.Error.ProviderError
	return replayedFailure{
		message: rec.Error.Message,
		cause: &apierror.ProviderError{
			Provider:   provider.Provider,
			Code:       provider.Code,
			Status:     provider.Status,
			Message:    provider.Message,
			RequestID:  provider.RequestID,
			RetryAfter: time.Duration(provider.RetryAfterNS),
		},
	}
}

func captureRecordedFailure(err error) *recordedFailure {
	if err == nil {
		return nil
	}
	failure := &recordedFailure{Message: err.Error()}
	var provider *apierror.ProviderError
	if errors.As(err, &provider) {
		failure.ProviderError = &recordedProviderError{
			Provider:     provider.Provider,
			Code:         provider.Code,
			Status:       provider.Status,
			Message:      provider.Message,
			RequestID:    provider.RequestID,
			RetryAfterNS: int64(provider.RetryAfter),
		}
	}
	return failure
}

// replayedFailure retains the exact recorded text while exposing the typed
// provider cause through errors.Is/errors.As to the production retry policy.
type replayedFailure struct {
	message string
	cause   error
}

func (e replayedFailure) Error() string { return e.message }
func (e replayedFailure) Unwrap() error { return e.cause }

// hashKey is the exact model + request hash used by recorder filenames.
func hashKey(model, prompt string) string {
	h := sha256.New()
	h.Write([]byte(model))
	h.Write([]byte("\x00"))
	h.Write([]byte(prompt))
	return fmt.Sprintf("%x", h.Sum(nil))[:16]
}

func (m *recorderMW) recordingPath(name string) string {
	return filepath.Join(m.recordingRoot(), name)
}

func (m *recorderMW) recordingRoot() string {
	if m.runID == RecorderDefaultRunID {
		return m.dir
	}
	return filepath.Join(m.dir, "runs", m.runID)
}

func (m *recorderMW) replayError(operation, hash string, cause error) error {
	if m.mode == RecordReplayOnly && errors.Is(cause, os.ErrNotExist) {
		return &ReplayMissError{
			Operation: operation, Hash: hash, Occurrence: 1,
			Root: m.recordingRoot(), Cause: cause,
		}
	}
	return fmt.Errorf("recorder: replay %s %s: %w", operation, hash, cause)
}

func canonicalRecorderRunID(runID string) (string, error) {
	if runID == "" {
		return RecorderDefaultRunID, nil
	}
	if strings.TrimSpace(runID) != runID {
		return "", fmt.Errorf("run id %q contains surrounding whitespace", runID)
	}
	for index, r := range runID {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
		case (r == '-' || r == '_') && index > 0:
		default:
			return "", fmt.Errorf("run id %q contains invalid character %q", runID, r)
		}
	}
	return runID, nil
}

func sendRecorderEvent(ctx context.Context, out chan<- StreamEvent, event StreamEvent) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case out <- event:
		return nil
	}
}
