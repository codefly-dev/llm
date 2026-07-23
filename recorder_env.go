package llm

import (
	"os"
	"strings"
)

// recorderDebugEnabled reports whether the recorder should dump miss/record
// key-inputs for cassette drift diagnosis. Strict "1" only — a deliberate
// opt-in because each miss writes a file. MIND_RECORDER_DEBUG=1.
func recorderDebugEnabled() bool {
	return os.Getenv("MIND_RECORDER_DEBUG") == "1"
}

// otelCaptureContentEnabled gates whether OTEL spans capture full
// request/response content (prompts, tool args, results) vs just metadata.
// Off by default — content can be large and may contain secrets.
// MIND_OTEL_CAPTURE_CONTENT=1|true|yes|on.
func otelCaptureContentEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("MIND_OTEL_CAPTURE_CONTENT"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}
