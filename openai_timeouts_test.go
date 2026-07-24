package llm

// ARCHITECTURE: OpenAI uses a different SDK transport from Anthropic, so its
// request deadline is certified independently. The server below injects the
// otherwise hard-to-produce failure mode from the live coding-loop trace: an
// accepted Responses API request that never produces a response.

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/benbjohnson/clock"
	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func TestOpenAIResponsesPerAttemptTimeout(t *testing.T) {
	requestArrived := make(chan struct{}, 1)
	releaseServer := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		select {
		case requestArrived <- struct{}{}:
		default:
		}
		select {
		case <-r.Context().Done():
		case <-releaseServer:
		}
	}))
	defer srv.Close()
	defer close(releaseServer)

	client := &OpenAI{
		sdk: openai.NewClient(
			option.WithAPIKey("timeout-certificate"),
			option.WithBaseURL(srv.URL),
			option.WithMaxRetries(0),
		),
		model: OpenAIModelGPT56Luna,
		clock: clock.New(),
		timeouts: TimeoutConfig{
			PerAttempt:  150 * time.Millisecond,
			StreamStall: -1,
		},
	}

	start := time.Now()
	_, err := client.CallWithTools(
		context.Background(),
		"Return a tool call.",
		[]Message{{Role: "user", Content: "wait forever"}},
		[]ToolDef{{
			Name: "finish",
			Parameters: []ToolParam{{
				Name: "result", Type: "string", Required: true,
			}},
		}},
	)
	elapsed := time.Since(start)

	if err == nil {
		t.Fatal("expected per-attempt timeout error, got clean success")
	}
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("err = %v, want CodeTimeout ProviderError", err)
	}
	if !IsRetryable(err) {
		t.Fatalf("per-attempt timeout must be retryable, got %v", err)
	}
	if elapsed > 3*time.Second {
		t.Fatalf("attempt budget took %s to fire, want <3s", elapsed)
	}
	select {
	case <-requestArrived:
	default:
		t.Fatal("timeout test never reached the OpenAI transport")
	}
}
