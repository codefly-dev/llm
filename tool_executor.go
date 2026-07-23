package llm

import "context"

// ToolExecutor is the small call-style tool seam used by LLM-side decorators
// such as HITLGuard, ScopeGuard, and ToolRecorder.
type ToolExecutor interface {
	Call(ctx context.Context, name string, args map[string]any) (content string, err error)
}

// ConcurrentToolExecutor extends ToolExecutor with read-only awareness so
// wrappers can preserve concurrency metadata when the inner executor exposes it.
type ConcurrentToolExecutor interface {
	ToolExecutor
	IsReadOnly(name string) bool
}
