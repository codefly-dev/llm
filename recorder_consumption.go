package llm

// ARCHITECTURE: a cassette is qualification evidence, not a best-effort cache.
// Request lookup proves that every call made by the current workflow was
// recorded; this file proves the inverse, that every recording owned by the
// selected run was exercised. Together those checks make workflow drift fail
// closed without coupling callers to the recorder's concrete middleware type.

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// RecorderConsumptionVerifier is implemented by the recorder middleware in a
// Client chain. Callers normally use VerifyRecorderConsumed so outer tracing,
// cost, or policy middleware remains transparent.
type RecorderConsumptionVerifier interface {
	VerifyRecorderConsumed() error
}

// VerifyRecorderConsumed finds every recorder in a Client middleware chain and
// proves that replay consumed its exact recording set. Asking a chain without
// a recorder to verify fails closed: otherwise a wiring regression could turn
// a qualification check into a silent no-op.
func VerifyRecorderConsumed(client Client) error {
	found := false
	var result error
	for current := client; current != nil; {
		if verifier, ok := current.(RecorderConsumptionVerifier); ok {
			found = true
			result = errors.Join(result, verifier.VerifyRecorderConsumed())
		}
		unwrapper, ok := current.(interface{ Unwrap() Client })
		if !ok {
			break
		}
		current = unwrapper.Unwrap()
	}
	if !found {
		return errors.New("llm recorder: consumption verification requested for a client without recorder middleware")
	}
	return result
}

// VerifyRecorderConsumed checks one recorder's selected run namespace. Record
// mode deliberately has no replay-consumption contract. Replay and on-miss
// modes reject stale, malformed, misnamed, or otherwise unowned objects.
func (m *recorderMW) VerifyRecorderConsumed() error {
	if m == nil {
		return errors.New("llm recorder: nil recorder middleware")
	}
	if m.configErr != nil {
		return fmt.Errorf("llm recorder: invalid configuration: %w", m.configErr)
	}
	if m.mode == RecordAlways {
		return nil
	}

	// A live on-miss call may still be committing its recording. Serialize the
	// inventory snapshot with writes, then copy the claimed paths under their
	// own lock. Callers invoke this after the workflow has joined all calls.
	m.writeMu.Lock()
	defer m.writeMu.Unlock()
	m.seenMu.Lock()
	consumed := make(map[string]struct{}, len(m.seen))
	for path := range m.seen {
		consumed[path] = struct{}{}
	}
	m.seenMu.Unlock()

	root := m.recordingRoot()
	entries, err := os.ReadDir(root)
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return fmt.Errorf("llm recorder: list replay recordings in %s: %w", root, err)
	}
	expected := make(map[string]struct{}, len(entries))
	for _, entry := range entries {
		path := filepath.Join(root, entry.Name())
		if entry.IsDir() {
			// Named runs are independent cassette samples. The default run owns
			// only flat files and must not consume its sibling run namespaces.
			if m.runID == RecorderDefaultRunID && entry.Name() == "runs" {
				continue
			}
			return fmt.Errorf("llm recorder: unowned entry object %s", path)
		}
		kind, hash, err := recorderIdentityFromName(entry.Name())
		if err != nil {
			return fmt.Errorf("llm recorder: unowned entry object %s: %w", path, err)
		}
		rec, err := readRecording(path)
		if err != nil {
			return fmt.Errorf("llm recorder: inspect replay recording %s: %w", path, err)
		}
		if err := m.validateRecording(rec, hash, kind); err != nil {
			return fmt.Errorf("llm recorder: invalid content-addressed recording %s: %w", path, err)
		}
		expected[path] = struct{}{}
	}

	paths := make(map[string]struct{}, len(expected)+len(consumed))
	for path := range expected {
		paths[path] = struct{}{}
	}
	for path := range consumed {
		paths[path] = struct{}{}
	}
	ordered := make([]string, 0, len(paths))
	for path := range paths {
		ordered = append(ordered, path)
	}
	sort.Strings(ordered)
	for _, path := range ordered {
		_, wasConsumed := consumed[path]
		_, wasRecorded := expected[path]
		if wasConsumed != wasRecorded {
			return fmt.Errorf(
				"llm recorder: replay consumption mismatch for %s: consumed %d, recorded %d",
				filepath.Base(path), boolCount(wasConsumed), boolCount(wasRecorded),
			)
		}
	}
	return nil
}

func readRecording(path string) (recording, error) {
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
	return rec, nil
}

func recorderIdentityFromName(name string) (recordingKind, string, error) {
	if filepath.Base(name) != name || filepath.Ext(name) != ".json" {
		return "", "", fmt.Errorf("invalid recording name %q", name)
	}
	stem := strings.TrimSuffix(name, ".json")
	kind := recordingText
	switch {
	case strings.HasPrefix(stem, "tools_"):
		kind, stem = recordingTool, strings.TrimPrefix(stem, "tools_")
	case strings.HasPrefix(stem, "events_"):
		kind, stem = recordingEvents, strings.TrimPrefix(stem, "events_")
	}
	if len(stem) != 16 || strings.ToLower(stem) != stem {
		return "", "", fmt.Errorf("invalid recording hash %q", stem)
	}
	if _, err := hex.DecodeString(stem); err != nil {
		return "", "", fmt.Errorf("invalid recording hash %q: %w", stem, err)
	}
	return kind, stem, nil
}

func boolCount(value bool) int {
	if value {
		return 1
	}
	return 0
}
