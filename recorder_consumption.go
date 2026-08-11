package llm

// ARCHITECTURE: a cassette is qualification evidence, not a best-effort cache.
// Request lookup proves that every call made by the current workflow was
// recorded; this file proves the inverse, that every recording owned by the
// selected run was exercised. A model router may create several recorder
// clients over one directory, so verification aggregates their claims before
// judging that shared namespace.

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

// VerifyRecorderConsumed finds the recorders in one or more Client middleware
// chains and proves replay consumed each selected cassette namespace exactly.
// Multiple clients are required when model routing shares one cassette
// directory. Asking clients without a recorder to verify fails closed so a
// wiring regression cannot turn qualification into a silent no-op.
func VerifyRecorderConsumed(clients ...Client) error {
	recorders := recorderMiddlewareSet(clients)
	if len(recorders) == 0 {
		return errors.New("llm recorder: consumption verification requested without recorder middleware")
	}
	groups := make(map[string][]*recorderMW)
	for _, recorder := range recorders {
		groups[recorder.recordingRoot()] = append(groups[recorder.recordingRoot()], recorder)
	}
	roots := make([]string, 0, len(groups))
	for root := range groups {
		roots = append(roots, root)
	}
	sort.Strings(roots)
	var result error
	for _, root := range roots {
		result = errors.Join(result, verifyRecorderGroup(root, groups[root]))
	}
	return result
}

func recorderMiddlewareSet(clients []Client) []*recorderMW {
	seen := make(map[*recorderMW]struct{})
	var recorders []*recorderMW
	for _, client := range clients {
		for current := client; current != nil; {
			if recorder, ok := current.(*recorderMW); ok {
				if _, exists := seen[recorder]; !exists {
					seen[recorder] = struct{}{}
					recorders = append(recorders, recorder)
				}
			}
			unwrapper, ok := current.(interface{ Unwrap() Client })
			if !ok {
				break
			}
			current = unwrapper.Unwrap()
		}
	}
	return recorders
}

func verifyRecorderGroup(root string, recorders []*recorderMW) error {
	mode := recorders[0].mode
	runID := recorders[0].runID
	for _, recorder := range recorders {
		if recorder.configErr != nil {
			return fmt.Errorf("llm recorder: invalid configuration: %w", recorder.configErr)
		}
		if recorder.mode != mode || recorder.runID != runID {
			return fmt.Errorf("llm recorder: inconsistent recorder configuration for shared root %s", root)
		}
	}
	if mode == RecordAlways {
		return nil
	}

	// Verification is an end-of-run operation after the caller has joined all
	// requests. Each snapshot still takes the recorder's normal locks so race
	// instrumentation and accidental overlap remain safe without introducing a
	// cross-recorder lock order.
	consumed := make(map[string]struct{})
	for _, recorder := range recorders {
		recorder.writeMu.Lock()
		recorder.seenMu.Lock()
		for path := range recorder.seen {
			consumed[path] = struct{}{}
		}
		recorder.seenMu.Unlock()
		recorder.writeMu.Unlock()
	}

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
			if runID == RecorderDefaultRunID && entry.Name() == "runs" {
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
		if err := validateRecordingForRun(rec, hash, kind, runID); err != nil {
			return fmt.Errorf("llm recorder: invalid content-addressed recording %s: %w", path, err)
		}
		if err := validateRecordingTransport(rec, recorders); err != nil {
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

// validateRecordingTransport checks a known current model against every
// recorder that may serve it. A recording for a removed model remains valid
// stale evidence and is reported by the consumption mismatch below.
func validateRecordingTransport(rec recording, recorders []*recorderMW) error {
	modelKnown := false
	for _, recorder := range recorders {
		if recorder.model != rec.Model {
			continue
		}
		modelKnown = true
		if rec.TransportIdentity == "" || rec.TransportIdentity == recorder.transportIdentity {
			return nil
		}
	}
	if modelKnown {
		return fmt.Errorf("transport_identity %q does not match the current recorder", rec.TransportIdentity)
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
