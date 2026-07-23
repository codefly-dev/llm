package llm

import (
	"context"
	"errors"
	"testing"
)

type fakeExec struct {
	calls   []string
	reply   string
	err     error
	readset map[string]bool
}

func (f *fakeExec) Call(_ context.Context, name string, _ map[string]any) (string, error) {
	f.calls = append(f.calls, name)
	return f.reply, f.err
}

func (f *fakeExec) IsReadOnly(name string) bool {
	if f.readset == nil {
		return false
	}
	return f.readset[name]
}

func TestScopeGuard_AllowsWriteInsideScope(t *testing.T) {
	inner := &fakeExec{reply: "ok"}
	g := &ScopeGuard{
		Inner:             inner,
		AllowedWritePaths: []string{"pkg/api/**/*.go"},
	}
	out, err := g.Call(context.Background(), "write", map[string]any{
		"path":    "pkg/api/handler.go",
		"content": "package api",
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if out != "ok" {
		t.Fatalf("wrong reply: %q", out)
	}
	if len(inner.calls) != 1 || inner.calls[0] != "write" {
		t.Fatalf("expected inner write, got %v", inner.calls)
	}
}

func TestScopeGuard_BlocksWriteOutsideScope(t *testing.T) {
	inner := &fakeExec{reply: "should not run"}
	var violations []string
	g := &ScopeGuard{
		Inner:             inner,
		AllowedWritePaths: []string{"pkg/api/**/*.go"},
		OnViolation: func(_ context.Context, tool, path, _ string) {
			violations = append(violations, tool+":"+path)
		},
	}
	_, err := g.Call(context.Background(), "write", map[string]any{
		"path":    "pkg/billing/charge.go",
		"content": "malicious",
	})
	if err == nil {
		t.Fatal("expected scope violation")
	}
	var sv *ScopeViolationError
	if !errors.As(err, &sv) {
		t.Fatalf("expected *ScopeViolationError, got %T", err)
	}
	if sv.Tool != "write" || sv.Path != "pkg/billing/charge.go" {
		t.Fatalf("wrong violation: %+v", sv)
	}
	if len(inner.calls) != 0 {
		t.Fatalf("inner should not have been called: %v", inner.calls)
	}
	if len(violations) != 1 {
		t.Fatalf("OnViolation called %d times, want 1", len(violations))
	}
}

func TestScopeGuard_ReadToolsPassThrough(t *testing.T) {
	inner := &fakeExec{reply: "hello", readset: map[string]bool{"read": true, "search": true}}
	g := &ScopeGuard{
		Inner:             inner,
		AllowedWritePaths: []string{"pkg/api/**/*.go"},
	}
	for _, tool := range []string{"read", "search", "list_symbols"} {
		out, err := g.Call(context.Background(), tool, map[string]any{"path": "anywhere/at/all.go"})
		if err != nil {
			t.Fatalf("read tool %q should pass through: %v", tool, err)
		}
		if out != "hello" {
			t.Fatalf("tool %q returned %q", tool, out)
		}
	}
	// IsReadOnly should delegate.
	if !g.IsReadOnly("read") {
		t.Fatal("IsReadOnly(read) should delegate to inner")
	}
	if g.IsReadOnly("write") {
		t.Fatal("IsReadOnly(write) should be false")
	}
}

func TestScopeGuard_ShellExecBlockedByDefault(t *testing.T) {
	inner := &fakeExec{reply: "should not run"}
	g := &ScopeGuard{
		Inner:             inner,
		AllowedWritePaths: []string{"**"},
	}
	_, err := g.Call(context.Background(), "shell_exec", map[string]any{"command": "rm -rf /"})
	if err == nil {
		t.Fatal("expected shell_exec block")
	}
	if len(inner.calls) != 0 {
		t.Fatal("inner should not have been called")
	}
}

func TestScopeGuard_ShellExecAllowedWhenOptIn(t *testing.T) {
	inner := &fakeExec{reply: "exit 0"}
	g := &ScopeGuard{
		Inner:             inner,
		AllowedWritePaths: []string{"**"},
		AllowShell:        true,
	}
	out, err := g.Call(context.Background(), "shell_exec", map[string]any{"command": "ls"})
	if err != nil || out != "exit 0" {
		t.Fatalf("expected pass-through, got out=%q err=%v", out, err)
	}
}

func TestScopeGuard_EmptyAllowedDeniesAllWrites(t *testing.T) {
	inner := &fakeExec{}
	g := &ScopeGuard{Inner: inner, AllowedWritePaths: nil}
	_, err := g.Call(context.Background(), "edit", map[string]any{"path": "anything.go"})
	if err == nil {
		t.Fatal("empty allowed scope should block all writes")
	}
	if len(inner.calls) != 0 {
		t.Fatal("inner should not have been called")
	}
}

func TestScopeGuard_HandlesDotSlashAndAbsolutePaths(t *testing.T) {
	inner := &fakeExec{reply: "ok"}
	g := &ScopeGuard{
		Inner:             inner,
		AllowedWritePaths: []string{"pkg/api/**"},
	}
	// Leading "./" should normalize.
	if _, err := g.Call(context.Background(), "write", map[string]any{"path": "./pkg/api/a.go"}); err != nil {
		t.Fatalf("./-prefix should match: %v", err)
	}
	// Leading "/" treated as repo-relative.
	if _, err := g.Call(context.Background(), "write", map[string]any{"path": "/pkg/api/b.go"}); err != nil {
		t.Fatalf("/-prefix should match: %v", err)
	}
}

func TestMatchGlob(t *testing.T) {
	cases := []struct {
		pattern, name string
		want          bool
	}{
		{"**", "a/b/c.go", true},
		{"**/*.go", "a/b/c.go", true},
		{"pkg/**/*.go", "pkg/api/handler.go", true},
		{"pkg/**/*.go", "pkg/api/deep/nested.go", true},
		{"pkg/api/**", "pkg/api/a.go", true},
		{"pkg/api/**", "pkg/billing/a.go", false},
		{"pkg/*.go", "pkg/foo.go", true},
		{"pkg/*.go", "pkg/sub/foo.go", false},
		{"exactfile.go", "exactfile.go", true},
		{"exactfile.go", "other.go", false},
		{"pkg/api/handler.go", "pkg/api/handler.go", true},
	}
	for _, c := range cases {
		got := matchGlob(c.pattern, c.name)
		if got != c.want {
			t.Errorf("matchGlob(%q, %q) = %v, want %v", c.pattern, c.name, got, c.want)
		}
	}
}

func TestScopeGuard_UnknownWriteToolNoPathKey(t *testing.T) {
	// A registered write tool with no path extractor configured → blocked.
	inner := &fakeExec{}
	g := &ScopeGuard{
		Inner:             inner,
		AllowedWritePaths: []string{"**"},
		WriteTools:        []string{"mystery_write"},
		PathExtractor:     map[string]string{},
	}
	_, err := g.Call(context.Background(), "mystery_write", map[string]any{"foo": "bar"})
	if err == nil {
		t.Fatal("expected block when no path extractor is configured")
	}
}
