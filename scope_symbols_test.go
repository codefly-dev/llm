package llm

import (
	"context"
	"errors"
	"testing"
)

func TestScopeGuard_SymbolAllowedExact(t *testing.T) {
	inner := &fakeExec{reply: "ok"}
	g := &ScopeGuard{
		Inner:               inner,
		AllowedWritePaths:   []string{"**"},
		AllowedWriteSymbols: []string{"HandleRequest", "User.id"},
	}
	out, err := g.Call(context.Background(), "rename_symbol", map[string]any{
		"path": "pkg/api/handler.go",
		"old":  "HandleRequest",
		"new":  "ServeRequest",
	})
	if err != nil {
		t.Fatalf("symbol on allowlist should pass: %v", err)
	}
	if out != "ok" {
		t.Fatalf("wrong reply: %q", out)
	}
}

func TestScopeGuard_SymbolBlockedNotInAllowlist(t *testing.T) {
	inner := &fakeExec{reply: "should not run"}
	var blocked []string
	g := &ScopeGuard{
		Inner:               inner,
		AllowedWritePaths:   []string{"**"},
		AllowedWriteSymbols: []string{"HandleRequest"},
		OnViolation: func(_ context.Context, tool, _, reason string) {
			blocked = append(blocked, tool+":"+reason)
		},
	}
	_, err := g.Call(context.Background(), "rename_symbol", map[string]any{
		"path": "pkg/api/handler.go",
		"old":  "ProcessPayment", // not on allowlist
		"new":  "PayProcessor",
	})
	if err == nil {
		t.Fatal("expected scope violation on disallowed symbol")
	}
	var sv *ScopeViolationError
	if !errors.As(err, &sv) {
		t.Fatalf("expected *ScopeViolationError, got %T", err)
	}
	if len(blocked) != 1 {
		t.Fatalf("OnViolation fired %d times, want 1", len(blocked))
	}
	if len(inner.calls) != 0 {
		t.Fatal("inner should not have been called")
	}
}

func TestScopeGuard_SymbolWildcardSuffix(t *testing.T) {
	inner := &fakeExec{reply: "ok"}
	g := &ScopeGuard{
		Inner:               inner,
		AllowedWritePaths:   []string{"**"},
		AllowedWriteSymbols: []string{"*.handler"}, // matches User.handler, Admin.handler, ...
	}
	cases := []struct {
		symbol string
		wantOK bool
	}{
		{"User.handler", true},
		{"Admin.handler", true},
		{"User.payload", false},
	}
	for _, c := range cases {
		_, err := g.Call(context.Background(), "rename_symbol", map[string]any{
			"path": "pkg/api/x.go",
			"old":  c.symbol,
			"new":  "x",
		})
		if (err == nil) != c.wantOK {
			t.Errorf("symbol %q: err=%v, wantOK=%v", c.symbol, err, c.wantOK)
		}
	}
}

func TestScopeGuard_SymbolEmptyAllowlistFallsBackToPathOnly(t *testing.T) {
	// AllowedWriteSymbols is empty → symbol check is skipped entirely.
	// Path-only enforcement applies, so any symbol is fine as long as the
	// path is in scope. This preserves backward compatibility with the
	// path-only ScopeGuard usage we shipped on Day 17.
	inner := &fakeExec{reply: "ok"}
	g := &ScopeGuard{
		Inner:             inner,
		AllowedWritePaths: []string{"pkg/**"},
	}
	out, err := g.Call(context.Background(), "rename_symbol", map[string]any{
		"path": "pkg/api/x.go",
		"old":  "AnythingGoes",
		"new":  "Y",
	})
	if err != nil || out != "ok" {
		t.Fatalf("symbol-empty allowlist should pass-through: out=%q err=%v", out, err)
	}
}

func TestScopeGuard_SymbolForUnknownToolPasses(t *testing.T) {
	// A tool not in DefaultSymbolExtractors (say, the plain "edit" tool)
	// has no symbol arg; the check is skipped and we fall through to the
	// path check. Regression guard so non-structured edits still work.
	inner := &fakeExec{reply: "ok"}
	g := &ScopeGuard{
		Inner:               inner,
		AllowedWritePaths:   []string{"**"},
		AllowedWriteSymbols: []string{"OnlyThisOne"},
	}
	out, err := g.Call(context.Background(), "edit", map[string]any{
		"path": "pkg/api/x.go",
	})
	if err != nil || out != "ok" {
		t.Fatalf("non-structured tools should not be symbol-checked: out=%q err=%v", out, err)
	}
}

func TestScopeGuard_SymbolAndPathEnforced(t *testing.T) {
	// In-scope path BUT off-list symbol → blocked.
	// Out-of-scope path → blocked at path check before symbol is even
	// looked at (which is intentional — paths fence the file, symbols
	// fence within).
	inner := &fakeExec{reply: "should not run"}
	g := &ScopeGuard{
		Inner:               inner,
		AllowedWritePaths:   []string{"pkg/api/**"},
		AllowedWriteSymbols: []string{"HandleRequest"},
	}
	// Off-symbol within in-scope path: block.
	_, err := g.Call(context.Background(), "rename_symbol", map[string]any{
		"path": "pkg/api/x.go",
		"old":  "Other",
		"new":  "Y",
	})
	if err == nil {
		t.Fatal("off-symbol should be blocked")
	}
	// Out-of-scope path AND off-symbol: blocked at path layer.
	_, err = g.Call(context.Background(), "rename_symbol", map[string]any{
		"path": "pkg/billing/x.go",
		"old":  "HandleRequest",
		"new":  "Y",
	})
	if err == nil {
		t.Fatal("out-of-scope path should be blocked even with allowed symbol")
	}
}
