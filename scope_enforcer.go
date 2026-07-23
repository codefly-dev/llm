// ARCHITECTURE: ScopeGuard enforces Objective.Scope.Write in the tool loop.
//
// Reference: docs/reference/loop/README.md §3.1 (tool safety) & §14.10.
//
// An objective declares a write scope (glob patterns). The LLM may attempt
// writes outside that scope — either mistake or jailbreak. ScopeGuard wraps
// the ToolExecutor and:
//  1. Detects write-family tool calls (write, edit, apply_edit, create_file,
//     delete_file, move_file, shell_exec).
//  2. Extracts the target path from the tool arguments.
//  3. Matches the path against AllowedWritePaths (globs, with ** support).
//  4. On violation: invokes OnViolation callback (typically emits a spine
//     ScopeViolation event) and returns a structured error. The loop sees
//     the error and can surface it to the planner.
//
// shell_exec is denied unconditionally unless AllowShell is set — the command
// string is free-form and can't be reliably path-checked.
package llm

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
)

// ScopeGuard is a ToolExecutor decorator that enforces write-scope policy.
type ScopeGuard struct {
	// Inner is the executor we're wrapping. Required.
	Inner ToolExecutor

	// AllowedWritePaths is the list of glob patterns the LLM may write to.
	// Empty slice = deny all writes.
	// A leading "/" or absolute path is treated as repo-relative.
	AllowedWritePaths []string

	// AllowedWriteSymbols, when non-empty, additionally restricts symbol-
	// level writes to specific identifier names. The Mind agent's
	// structured edit tools (rename_symbol, replace_symbol, etc.) populate
	// a "symbol" argument that ScopeGuard checks against this list. Empty
	// list = path-only enforcement (the historical behaviour).
	//
	// This is the second of LOOM's two scope dimensions: paths (geometry)
	// + symbols (semantics). Paths catch "write to the wrong file"; symbols
	// catch "write to the wrong function within an allowed file."
	AllowedWriteSymbols []string

	// AllowShell permits shell_exec calls. By default shell_exec is blocked
	// because the command string is free-form and can't be path-checked.
	AllowShell bool

	// WriteTools is the list of tool names treated as write operations.
	// If empty, DefaultWriteTools is used.
	WriteTools []string

	// PathExtractor maps tool name → argument key holding the target path.
	// If nil, DefaultPathExtractors is used.
	PathExtractor map[string]string

	// SymbolExtractor maps tool name → argument key holding the target
	// symbol identifier. If nil, DefaultSymbolExtractors is used. Only
	// consulted when AllowedWriteSymbols is non-empty.
	SymbolExtractor map[string]string

	// OnViolation is invoked whenever a write is blocked. Typically used to
	// emit a spine event. Optional.
	OnViolation func(ctx context.Context, tool string, path string, reason string)
}

// DefaultWriteTools is the canonical list of tools that mutate the repo.
// Callers can override via ScopeGuard.WriteTools.
var DefaultWriteTools = []string{
	"write",
	"write_file",
	"edit",
	"apply_edit",
	"create_file",
	"delete_file",
	"move_file",
	"shell_exec",
	// Structured semantic edits (D9 — see pkg/astedit). These are checked
	// against both AllowedWritePaths AND, when configured, AllowedWriteSymbols.
	"rename_symbol",
	"replace_symbol",
	"insert_after",
	"delete_symbol",
}

// DefaultPathExtractors maps tool name → arg key holding the target path.
// Used by ScopeGuard when its PathExtractor map is nil.
var DefaultPathExtractors = map[string]string{
	"write":       "path",
	"write_file":  "path",
	"edit":        "path",
	"apply_edit":  "path",
	"create_file": "path",
	"delete_file": "path",
	"move_file":   "path",
	// Structured semantic edits also take a "path" arg.
	"rename_symbol":  "path",
	"replace_symbol": "path",
	"insert_after":   "path",
	"delete_symbol":  "path",
}

// DefaultSymbolExtractors maps tool name → arg key holding the target
// symbol identifier for symbol-scoped enforcement. The four structured
// edit tools (rename_symbol, replace_symbol, insert_after, delete_symbol)
// all use "symbol" or "old" as the canonical argument name; we accept
// both via per-tool entries here.
var DefaultSymbolExtractors = map[string]string{
	"rename_symbol":  "old",
	"replace_symbol": "symbol",
	"insert_after":   "anchor_symbol",
	"delete_symbol":  "symbol",
}

// ScopeViolationError is returned from Call when a write is blocked.
// The loop surfaces the error text to the LLM which can re-plan.
type ScopeViolationError struct {
	Tool   string
	Path   string
	Reason string
}

func (e *ScopeViolationError) Error() string {
	return fmt.Sprintf("scope violation: tool %q blocked from %q (%s)", e.Tool, e.Path, e.Reason)
}

// Call executes the tool after checking scope. Read-only tools pass through.
func (g *ScopeGuard) Call(ctx context.Context, name string, args map[string]any) (string, error) {
	if !g.isWriteTool(name) {
		return g.Inner.Call(ctx, name, args)
	}

	// shell_exec: deny unless explicitly allowed; if allowed, pass through
	// without path-matching (the command string is free-form).
	if name == "shell_exec" {
		if !g.AllowShell {
			reason := "shell_exec blocked: AllowShell is false for this scope"
			g.notify(ctx, name, "", reason)
			return "", &ScopeViolationError{Tool: name, Path: "", Reason: reason}
		}
		return g.Inner.Call(ctx, name, args)
	}

	path, ok := g.extractPath(name, args)
	if !ok {
		// Write tool without a path argument we recognise → refuse.
		reason := "no path argument extracted"
		g.notify(ctx, name, "", reason)
		return "", &ScopeViolationError{Tool: name, Path: "", Reason: reason}
	}

	if !g.allowed(path) {
		reason := fmt.Sprintf("path not in allowed write scope (%d patterns)", len(g.AllowedWritePaths))
		g.notify(ctx, name, path, reason)
		return "", &ScopeViolationError{Tool: name, Path: path, Reason: reason}
	}

	// Symbol-scope enforcement: when AllowedWriteSymbols is configured,
	// structured edit tools whose argument names a symbol must target a
	// symbol on the allowlist. Path-only tools (write, edit, apply_edit)
	// are not affected — those land via the path check above.
	if len(g.AllowedWriteSymbols) > 0 {
		if symbol, hasSymbol := g.extractSymbol(name, args); hasSymbol {
			if !g.symbolAllowed(symbol) {
				reason := fmt.Sprintf("symbol %q not in allowed write-symbols (%d allowed)", symbol, len(g.AllowedWriteSymbols))
				g.notify(ctx, name, path, reason)
				return "", &ScopeViolationError{Tool: name, Path: path, Reason: reason}
			}
		}
	}

	return g.Inner.Call(ctx, name, args)
}

// extractSymbol returns the target symbol identifier from args using the
// configured SymbolExtractor map (or DefaultSymbolExtractors). Returns
// false when the tool has no symbol-extractor entry — meaning it's not a
// structured edit tool and the symbol-scope check doesn't apply.
func (g *ScopeGuard) extractSymbol(name string, args map[string]any) (string, bool) {
	extractors := g.SymbolExtractor
	if extractors == nil {
		extractors = DefaultSymbolExtractors
	}
	key, ok := extractors[name]
	if !ok {
		return "", false
	}
	v, ok := args[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	return s, ok && s != ""
}

// symbolAllowed returns true if symbol matches any AllowedWriteSymbols
// entry. A leading "*." prefix matches any symbol with that suffix
// ("*.handler" matches both "User.handler" and "Admin.handler"); otherwise
// exact-string match.
func (g *ScopeGuard) symbolAllowed(symbol string) bool {
	for _, allowed := range g.AllowedWriteSymbols {
		if allowed == symbol {
			return true
		}
		if strings.HasPrefix(allowed, "*.") {
			suffix := allowed[1:]
			if strings.HasSuffix(symbol, suffix) {
				return true
			}
		}
	}
	return false
}

// IsReadOnly delegates to the inner executor if it supports concurrency hints.
// Write tools are never read-only; the guard itself doesn't promote anything.
func (g *ScopeGuard) IsReadOnly(name string) bool {
	if g.isWriteTool(name) {
		return false
	}
	if inner, ok := g.Inner.(ConcurrentToolExecutor); ok {
		return inner.IsReadOnly(name)
	}
	return false
}

func (g *ScopeGuard) isWriteTool(name string) bool {
	list := g.WriteTools
	if len(list) == 0 {
		list = DefaultWriteTools
	}
	return slices.Contains(list, name)
}

func (g *ScopeGuard) extractPath(name string, args map[string]any) (string, bool) {
	extractors := g.PathExtractor
	if extractors == nil {
		extractors = DefaultPathExtractors
	}
	key, ok := extractors[name]
	if !ok {
		return "", false
	}
	v, ok := args[key]
	if !ok {
		return "", false
	}
	s, ok := v.(string)
	if !ok {
		return "", false
	}
	return cleanPath(s), s != ""
}

// allowed returns true if path matches any AllowedWritePaths glob.
func (g *ScopeGuard) allowed(path string) bool {
	clean := cleanPath(path)
	for _, pat := range g.AllowedWritePaths {
		if matchGlob(cleanPath(pat), clean) {
			return true
		}
	}
	return false
}

func (g *ScopeGuard) notify(ctx context.Context, tool, path, reason string) {
	if g.OnViolation != nil {
		g.OnViolation(ctx, tool, path, reason)
	}
}

// cleanPath strips a leading "./" or "/" so relative and absolute paths
// compare uniformly. The path is also filepath.Clean'd.
func cleanPath(p string) string {
	p = strings.TrimSpace(p)
	if p == "" {
		return ""
	}
	p = filepath.ToSlash(filepath.Clean(p))
	p = strings.TrimPrefix(p, "./")
	p = strings.TrimPrefix(p, "/")
	return p
}

// matchGlob supports * within a path segment and ** across segments.
// Implementation is intentionally minimal: we don't depend on bmatcuk/doublestar.
func matchGlob(pattern, name string) bool {
	if pattern == name {
		return true
	}
	if pattern == "**" || pattern == "**/*" {
		return true
	}
	// Handle ** anywhere in the pattern: split and match each side.
	if strings.Contains(pattern, "**") {
		parts := strings.SplitN(pattern, "**", 2)
		prefix := strings.TrimSuffix(parts[0], "/")
		suffix := strings.TrimPrefix(parts[1], "/")
		if prefix != "" && !strings.HasPrefix(name, prefix) {
			return false
		}
		rest := name
		if prefix != "" {
			rest = strings.TrimPrefix(name, prefix)
			rest = strings.TrimPrefix(rest, "/")
		}
		if suffix == "" {
			return true
		}
		// The suffix itself may contain more globs; recurse.
		// Try every possible split boundary inside rest.
		if matchGlob(suffix, rest) {
			return true
		}
		// Advance the match window one segment at a time.
		for i := 0; i < len(rest); i++ {
			if rest[i] == '/' && matchGlob(suffix, rest[i+1:]) {
				return true
			}
		}
		return false
	}
	// No **: use filepath.Match (supports single * within a segment).
	ok, err := filepath.Match(pattern, name)
	if err != nil {
		return false
	}
	return ok
}
