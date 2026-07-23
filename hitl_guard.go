// ARCHITECTURE: HITLGuard is the tool-level human-in-the-loop interceptor.
//
// Reference: docs/reference/loop/README.md §3.1 (tool safety) & §14.10.
//
// Some tool calls are destructive and one-way: source edits, a deployment
// kickoff, `rm -rf`, `curl` to an external endpoint. Even if they're inside
// the objective's write scope, we want a human (or a scripted auto-replier)
// to sign off before executing.
//
// HITLGuard sits between the loop and the real executor:
//  1. For each tool call, RequiresApproval decides whether the call needs
//     sign-off.
//  2. If it does, Approver is asked. Approver returns (approved, reason).
//  3. Approved calls pass through to Inner. Denied calls emit an optional
//     HITL event via OnDenied and return a ToolHITLDeniedError.
//
// Compose with ScopeGuard: the runner chain is Inner → ScopeGuard → HITLGuard.
// ScopeGuard enforces geometry (paths); HITLGuard enforces policy (approval).
package llm

import (
	"context"
	"fmt"
	"slices"
	"strings"
)

// HITLGuard is a ToolExecutor decorator that requires approval before
// destructive tool calls.
type HITLGuard struct {
	// Inner is the executor we're wrapping. Required.
	Inner ToolExecutor

	// RequiresApproval, when non-nil, overrides the default destructive-tool
	// detection. The default policy matches DefaultDestructiveTools and
	// inspects shell_exec command text for banned patterns.
	RequiresApproval func(name string, args map[string]any) bool

	// Approver decides whether a call may proceed. Must be non-nil; default
	// behaviour is deny (fail-safe). In tests, wire a scripted approver; in
	// production, wire an async HITL queue.
	Approver HITLApprover

	// OnDenied is invoked after a denial, before the error returns. Typically
	// emits a spine HITLRequest / HITLResponse pair so the denial is auditable.
	OnDenied func(ctx context.Context, tool string, args map[string]any, reason string)
}

// HITLApprover decides whether a destructive tool call may proceed.
// Implementations can block on a human response, consult a policy engine,
// or play back a scripted answer for deterministic testing.
type HITLApprover interface {
	Approve(ctx context.Context, tool string, args map[string]any) (approved bool, reason string, err error)
}

// HITLApproverFunc adapts an ordinary function to HITLApprover.
type HITLApproverFunc func(ctx context.Context, tool string, args map[string]any) (bool, string, error)

// Approve implements HITLApprover.
func (f HITLApproverFunc) Approve(ctx context.Context, tool string, args map[string]any) (bool, string, error) {
	return f(ctx, tool, args)
}

// DefaultDestructiveTools is the canonical list of tools that trigger HITL
// approval by default. Callers can override via RequiresApproval.
var DefaultDestructiveTools = []string{
	"create",
	"edit",
	"insert_after",
	"symbol_patch",
	"deploy",
	"publish",
	"release",
	"delete_file",
	"move_file",
	"rollback",
}

// DestructiveShellPatterns are substrings inside shell_exec command args
// that trigger approval even when shell_exec itself is allowed.
var DestructiveShellPatterns = []string{
	"rm -rf",
	"sudo ",
	"git push",
	"docker push",
	"terraform apply",
	"helm uninstall",
	"aws s3 rm",
	"gh pr merge",
	":(){",
	"mkfs",
}

// ToolHITLDeniedError is returned when the approver rejects a call.
type ToolHITLDeniedError struct {
	Tool   string
	Reason string
}

func (e *ToolHITLDeniedError) Error() string {
	if e.Reason != "" {
		return fmt.Sprintf("tool %q denied by HITL: %s", e.Tool, e.Reason)
	}
	return fmt.Sprintf("tool %q denied by HITL", e.Tool)
}

// Call checks for approval before dispatching to Inner.
func (g *HITLGuard) Call(ctx context.Context, name string, args map[string]any) (string, error) {
	if !g.requiresApproval(name, args) {
		return g.Inner.Call(ctx, name, args)
	}

	if g.Approver == nil {
		reason := "no HITL approver configured (default-deny)"
		g.notify(ctx, name, args, reason)
		return "", &ToolHITLDeniedError{Tool: name, Reason: reason}
	}

	approved, reason, err := g.Approver.Approve(ctx, name, args)
	if err != nil {
		g.notify(ctx, name, args, err.Error())
		return "", fmt.Errorf("hitl approver: %w", err)
	}
	if !approved {
		g.notify(ctx, name, args, reason)
		return "", &ToolHITLDeniedError{Tool: name, Reason: reason}
	}
	return g.Inner.Call(ctx, name, args)
}

// IsReadOnly delegates to the inner executor if available. HITLGuard never
// claims read-only status itself because the gate is about policy, not I/O.
func (g *HITLGuard) IsReadOnly(name string) bool {
	if inner, ok := g.Inner.(ConcurrentToolExecutor); ok {
		return inner.IsReadOnly(name)
	}
	return false
}

func (g *HITLGuard) requiresApproval(name string, args map[string]any) bool {
	if g.RequiresApproval != nil {
		return g.RequiresApproval(name, args)
	}
	if slices.Contains(DefaultDestructiveTools, name) {
		return true
	}
	if name == "shell_exec" {
		if cmd, ok := args["command"].(string); ok && containsDestructivePattern(cmd) {
			return true
		}
	}
	return false
}

func (g *HITLGuard) notify(ctx context.Context, tool string, args map[string]any, reason string) {
	if g.OnDenied != nil {
		g.OnDenied(ctx, tool, args, reason)
	}
}

// AlwaysApprove is a convenience approver that always approves. Useful in
// scripted tests where policy is not under test.
func AlwaysApprove() HITLApprover {
	return HITLApproverFunc(func(_ context.Context, _ string, _ map[string]any) (bool, string, error) {
		return true, "scripted always-approve", nil
	})
}

// AlwaysDeny is a convenience approver for negative tests.
func AlwaysDeny(reason string) HITLApprover {
	r := reason
	if r == "" {
		r = "scripted always-deny"
	}
	return HITLApproverFunc(func(_ context.Context, _ string, _ map[string]any) (bool, string, error) {
		return false, r, nil
	})
}

// ScriptedApprover approves calls whose tool+args match a table of rules.
// Each rule matches a tool name and optionally a substring in the first
// string argument found. Useful for deterministic testing of mixed flows.
type ScriptedApprover struct {
	Rules []ApprovalRule
}

// ApprovalRule is one row of a ScriptedApprover.
type ApprovalRule struct {
	Tool     string // tool name; "*" matches any
	Contains string // substring to match in any string arg (empty = don't check)
	Approve  bool
	Reason   string
}

// Approve implements HITLApprover.
func (s *ScriptedApprover) Approve(_ context.Context, tool string, args map[string]any) (bool, string, error) {
	for _, r := range s.Rules {
		if r.Tool != "*" && r.Tool != tool {
			continue
		}
		if r.Contains != "" && !anyArgContains(args, r.Contains) {
			continue
		}
		return r.Approve, r.Reason, nil
	}
	return false, fmt.Sprintf("no matching rule for %s", tool), nil
}

func anyArgContains(args map[string]any, needle string) bool {
	for _, v := range args {
		if s, ok := v.(string); ok && strings.Contains(s, needle) {
			return true
		}
	}
	return false
}

func containsDestructivePattern(cmd string) bool {
	for _, p := range DestructiveShellPatterns {
		if strings.Contains(cmd, p) {
			return true
		}
	}
	return false
}
