package llm

// ToolRecorder wraps a ToolExecutor and records every Call(name, args)
// result to disk. On replay it returns the recorded content without
// dispatching the real tool, turning tool execution into an on-disk
// lookup just like RecorderMiddleware does for LLM calls.
//
// This closes the last non-determinism gap in cassette-based tests: LLM
// calls were already deterministic (pkg/llm/recorder.go), but tool
// execution hit live Neo4j / Postgres / file system every turn — 20s
// on xarray exploration. With a recorded ToolExecutor the same test
// runs from disk in sub-second.
//
// ── Cache key ─────────────────────────────────────────────────────────
//
// sha256(tool_name + "\n" + canonicalize(args)). Args are canonicalized
// by encoding them with json.Marshal after sorting any map keys — Go's
// map iteration order is nondeterministic, and the whole point is a
// stable hash. Same inputs → same path on disk.
//
// ── Mode semantics (reuse of RecordMode) ──────────────────────────────
//
//   RecordReplayOnly — miss = error. Default for CI. Guarantees the
//                      test can't silently drift into live calls.
//   RecordAlways    — skip replay, always call real, overwrite on disk.
//                      Use when the tool's underlying graph has changed
//                      and every recording should be refreshed.
//   RecordOnMiss    — replay exact hits and call the real tool only when
//                     its recording is absent. Corrupt recordings fail.
//
// ── What NOT to wrap with this ────────────────────────────────────────
//
// Mutation tools (create, edit, symbol_patch, shell_exec, …) must NOT be
// recorded: replaying a file write doesn't actually write the file. The
// wrapped executor's ConcurrentToolExecutor.IsReadOnly metadata, or the
// explicit predicate passed to NewToolRecorder, tells the wrapper which
// tools are safe to cache. Non-read-only tools bypass both replay and
// record and hit the inner executor directly.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/codefly-dev/core/wool"
)

// ToolRecorder is a ToolExecutor middleware that caches tool results on
// disk. Construct via NewToolRecorder.
type ToolRecorder struct {
	inner    ToolExecutor
	dir      string
	mode     RecordMode
	readOnly func(name string) bool
}

// NewToolRecorder wraps inner and records to dir under the given mode.
// readOnly decides which tools are safe to cache when inner does not expose
// ConcurrentToolExecutor. If nil, a small allowlist of known read-only tool
// names is used; unknown tools are treated as live-only.
func NewToolRecorder(inner ToolExecutor, dir string, mode RecordMode, readOnly func(name string) bool) *ToolRecorder {
	if readOnly == nil {
		readOnly = defaultReadOnlyPredicate
	}
	return &ToolRecorder{inner: inner, dir: dir, mode: mode, readOnly: readOnly}
}

// Call implements ToolExecutor.
func (r *ToolRecorder) Call(ctx context.Context, name string, args map[string]any) (string, error) {
	// Mutations always bypass the cassette layer — replaying a write is
	// meaningless. Tool-loops that use write tools should not use a
	// ToolRecorder in the first place, but this guard keeps a mixed
	// toolbox safe by default. Go through IsReadOnly so the tool's
	// read-only metadata is honored.
	if !r.IsReadOnly(name) {
		return r.inner.Call(ctx, name, args)
	}

	hash := hashToolCall(name, args)
	path := filepath.Join(r.dir, "tool_"+hash+".json")

	if r.mode != RecordAlways {
		content, replayErr := r.replay(path)
		if replayErr == nil {
			return content, nil
		}
		if r.mode == RecordReplayOnly || !errors.Is(replayErr, os.ErrNotExist) {
			return "", fmt.Errorf("tool recorder: no recording for %s (hash=%s) in %s: %w — set MIND_RECORD=heal to fill gaps or MIND_RECORD=1 to re-record", name, hash, r.dir, replayErr)
		}
	}

	content, err := r.inner.Call(ctx, name, args)
	if err != nil {
		return content, err
	}
	if writeErr := r.record(path, name, args, content); writeErr != nil {
		// Non-fatal — the real call succeeded, we just couldn't persist.
		wool.Get(ctx).In("llm.tool_recorder").Warn("failed to write tool recording",
			wool.Field("path", path), wool.ErrField(writeErr))
	}
	return content, nil
}

// IsReadOnly implements ConcurrentToolExecutor when the inner does.
// Keeps the recorder transparent to the loop's concurrency scheduling.
func (r *ToolRecorder) IsReadOnly(name string) bool {
	if c, ok := r.inner.(ConcurrentToolExecutor); ok {
		return c.IsReadOnly(name)
	}
	return r.readOnly(name)
}

// ── Internals ────────────────────────────────────────────────────────

type toolRecord struct {
	Name    string `json:"name"`
	Args    any    `json:"args"`
	Content string `json:"content"`
	Hash    string `json:"hash"`
}

// hashToolCall computes a stable sha256 hex over (name, canonicalized-args).
// Canonicalization: marshal args with keys sorted so map iteration order
// doesn't affect the hash.
func hashToolCall(name string, args map[string]any) string {
	h := sha256.New()
	h.Write([]byte(name))
	h.Write([]byte("\n"))
	h.Write([]byte(canonicalJSON(args)))
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// canonicalJSON serializes m with sorted keys recursively. For nested
// maps, sorts their keys too. For slices, preserves order (order matters
// for tool-call semantics like "top-5 results").
func canonicalJSON(v any) string {
	switch m := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(m))
		for k := range m {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		var b []byte
		b = append(b, '{')
		for i, k := range keys {
			if i > 0 {
				b = append(b, ',')
			}
			kj, _ := json.Marshal(k)
			b = append(b, kj...)
			b = append(b, ':')
			b = append(b, canonicalJSON(m[k])...)
		}
		b = append(b, '}')
		return string(b)
	case []any:
		var b []byte
		b = append(b, '[')
		for i, e := range m {
			if i > 0 {
				b = append(b, ',')
			}
			b = append(b, canonicalJSON(e)...)
		}
		b = append(b, ']')
		return string(b)
	default:
		raw, _ := json.Marshal(m)
		return string(raw)
	}
}

func (r *ToolRecorder) replay(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	var rec toolRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return "", err
	}
	return rec.Content, nil
}

func (r *ToolRecorder) record(path, name string, args map[string]any, content string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(toolRecord{
		Name:    name,
		Args:    args,
		Content: content,
		Hash:    filepath.Base(path),
	}, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	return os.WriteFile(path, data, 0o644)
}

// defaultReadOnlyPredicate is the fallback for bare ToolExecutor test
// doubles that do not expose ConcurrentToolExecutor. Production registries
// should carry read-only flags per tool, so unknown names stay live-only
// here to avoid accidental replay of mutation tools.
func defaultReadOnlyPredicate(name string) bool {
	switch name {
	case "read", "read_file", "list_files", "glob", "grep", "search",
		"web_search", "web_fetch",
		"code_status", "code_diff", "code_history",
		"knowledge_query", "semantic_search", "graph_query",
		"localize_symbol", "localize_files", "localize_tests":
		return true
	}
	return false
}
