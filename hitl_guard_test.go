package llm

import (
	"context"
	"errors"
	"testing"
)

func TestHITLGuard_NonDestructivePassesThrough(t *testing.T) {
	inner := &fakeExec{reply: "ok"}
	g := &HITLGuard{Inner: inner, Approver: AlwaysDeny("")}
	out, err := g.Call(context.Background(), "read", map[string]any{"path": "x.go"})
	if err != nil || out != "ok" {
		t.Fatalf("non-destructive tool should pass through: out=%q err=%v", out, err)
	}
}

func TestHITLGuard_DestructiveDeniedByDefault(t *testing.T) {
	inner := &fakeExec{reply: "should not run"}
	var denied []string
	g := &HITLGuard{
		Inner: inner,
		// No Approver → default-deny.
		OnDenied: func(_ context.Context, tool string, _ map[string]any, _ string) {
			denied = append(denied, tool)
		},
	}
	_, err := g.Call(context.Background(), "edit", map[string]any{"path": "main.go"})
	var hitlErr *ToolHITLDeniedError
	if !errors.As(err, &hitlErr) {
		t.Fatalf("expected *ToolHITLDeniedError, got %T (%v)", err, err)
	}
	if hitlErr.Tool != "edit" {
		t.Fatalf("wrong tool: %v", hitlErr.Tool)
	}
	if len(inner.calls) != 0 {
		t.Fatal("inner should not have been called")
	}
	if len(denied) != 1 {
		t.Fatalf("OnDenied fired %d times, want 1", len(denied))
	}
}

func TestHITLGuard_ApprovedCallPassesThrough(t *testing.T) {
	inner := &fakeExec{reply: "edited"}
	g := &HITLGuard{Inner: inner, Approver: AlwaysApprove()}
	out, err := g.Call(context.Background(), "edit", map[string]any{"path": "main.go"})
	if err != nil || out != "edited" {
		t.Fatalf("approved call should pass through: out=%q err=%v", out, err)
	}
	if len(inner.calls) != 1 {
		t.Fatal("inner should have been called once")
	}
}

func TestHITLGuard_ShellExecPatternTriggersApproval(t *testing.T) {
	inner := &fakeExec{reply: "should not run"}
	g := &HITLGuard{Inner: inner, Approver: AlwaysDeny("nope")}
	_, err := g.Call(context.Background(), "shell_exec", map[string]any{"command": "sudo rm -rf /"})
	var hitlErr *ToolHITLDeniedError
	if !errors.As(err, &hitlErr) {
		t.Fatalf("expected HITL denial for destructive shell, got %v", err)
	}
	if len(inner.calls) != 0 {
		t.Fatal("inner should not have run for destructive shell_exec")
	}
}

func TestHITLGuard_ShellExecSafeCommandPassesThrough(t *testing.T) {
	inner := &fakeExec{reply: "hi"}
	g := &HITLGuard{Inner: inner, Approver: AlwaysDeny("")}
	out, err := g.Call(context.Background(), "shell_exec", map[string]any{"command": "echo hi"})
	if err != nil || out != "hi" {
		t.Fatalf("safe shell should pass through: out=%q err=%v", out, err)
	}
}

func TestHITLGuard_ScriptedApprover_ApprovesMatchingRule(t *testing.T) {
	inner := &fakeExec{reply: "ok"}
	g := &HITLGuard{
		Inner: inner,
		Approver: &ScriptedApprover{Rules: []ApprovalRule{
			{Tool: "edit", Contains: "pkg/", Approve: true, Reason: "scoped package edit"},
			{Tool: "edit", Contains: "main.go", Approve: false, Reason: "main needs review"},
			{Tool: "*", Approve: false, Reason: "unknown tool"},
		}},
	}
	// Scoped edit: approved.
	if _, err := g.Call(context.Background(), "edit", map[string]any{"path": "pkg/hitl.go"}); err != nil {
		t.Fatalf("scoped edit should be approved: %v", err)
	}
	// Main edit: denied.
	if _, err := g.Call(context.Background(), "edit", map[string]any{"path": "main.go"}); err == nil {
		t.Fatal("main edit should be denied")
	}
}

func TestHITLGuard_CustomRequiresApproval(t *testing.T) {
	inner := &fakeExec{reply: "ok"}
	g := &HITLGuard{
		Inner: inner,
		RequiresApproval: func(name string, _ map[string]any) bool {
			return name == "read" // treat read as destructive in this config
		},
		Approver: AlwaysDeny("reads need approval"),
	}
	// edit would normally need approval, but the custom predicate owns policy.
	if _, err := g.Call(context.Background(), "edit", map[string]any{"path": "a.go"}); err != nil {
		t.Fatalf("edit should pass under custom policy: %v", err)
	}
	// read is now destructive.
	if _, err := g.Call(context.Background(), "read", map[string]any{"path": "a.go"}); err == nil {
		t.Fatal("read should be denied")
	}
}

func TestHITLGuard_ComposesWithScopeGuard(t *testing.T) {
	// Inner → ScopeGuard → HITLGuard.
	// Test: in-scope write approved by HITL = passes. In-scope write denied by HITL = blocked.
	inner := &fakeExec{reply: "wrote"}
	scoped := &ScopeGuard{Inner: inner, AllowedWritePaths: []string{"pkg/**/*.go"}}
	gated := &HITLGuard{
		Inner: scoped,
		RequiresApproval: func(name string, _ map[string]any) bool {
			return name == "write" // treat write as needing approval here
		},
		Approver: &ScriptedApprover{Rules: []ApprovalRule{
			{Tool: "write", Contains: "test.go", Approve: true, Reason: "test files are safe"},
			{Tool: "*", Approve: false, Reason: "default deny"},
		}},
	}
	// Approved + in-scope: wrote.
	out, err := gated.Call(context.Background(), "write", map[string]any{"path": "pkg/api/handler_test.go"})
	if err != nil || out != "wrote" {
		t.Fatalf("should have written: out=%q err=%v", out, err)
	}
	// HITL denied: blocked before ScopeGuard.
	_, err = gated.Call(context.Background(), "write", map[string]any{"path": "pkg/api/handler.go"})
	if err == nil {
		t.Fatal("expected denial for non-test file")
	}
	// ScopeGuard still blocks out-of-scope even when HITL would approve.
	_, err = gated.Call(context.Background(), "write", map[string]any{"path": "/etc/passwd_test.go"})
	if err == nil {
		t.Fatal("expected scope block for out-of-repo path")
	}
}
