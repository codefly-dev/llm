// ARCHITECTURE: replay misses are deterministic certification failures, not
// provider failures. This typed error crosses the recorder boundary so retry,
// circuit-breaking, and orchestration layers can fail closed without treating
// a missing local recording as evidence that the live provider is unhealthy.
package llm

import (
	"errors"
	"fmt"
	"strings"
)

// ReplayMissError identifies the exact native LLM request that replay-only
// mode could not satisfy. Native recordings are content-addressed and reject
// duplicate consumption, so their occurrence is always one.
type ReplayMissError struct {
	Operation  string
	Hash       string
	Occurrence int
	Root       string
	Cause      error
}

func (e *ReplayMissError) Error() string {
	if e == nil {
		return "llm recorder replay miss"
	}
	identity := fmt.Sprintf(
		"operation=%s hash=%s occurrence=%d root=%s",
		emptyReplayMissField(e.Operation),
		emptyReplayMissField(e.Hash),
		e.Occurrence,
		emptyReplayMissField(e.Root),
	)
	if e.Cause != nil {
		return fmt.Sprintf("llm recorder replay miss (%s): %v", identity, e.Cause)
	}
	return "llm recorder replay miss (" + identity + ")"
}

func emptyReplayMissField(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return "unknown"
	}
	return value
}

// Unwrap preserves errors.Is(err, fs.ErrNotExist) for callers that still need
// the underlying storage diagnosis.
func (e *ReplayMissError) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.Cause
}

// IsExternalReplayMiss is a dependency-neutral marker consumed by Mind's
// external-boundary policy without making this standalone module import Mind.
func (e *ReplayMissError) IsExternalReplayMiss() bool { return e != nil }

// IsReplayMiss reports whether err contains a deterministic native-recorder
// miss. It remains false for corrupt recordings and live-provider failures.
func IsReplayMiss(err error) bool {
	var miss *ReplayMissError
	return errors.As(err, &miss)
}
