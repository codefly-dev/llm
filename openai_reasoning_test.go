package llm

import (
	"testing"

	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
)

// reasoning_effort is allowed only for the current GPT-5.6 roster and is
// omitted for unknown models and ReasoningNone.
func TestOpenAIReasoningEffortGating(t *testing.T) {
	cases := []struct {
		model  string
		effort ReasoningEffort
		want   shared.ReasoningEffort
	}{
		{"gpt-5.6-sol", ReasoningHigh, shared.ReasoningEffortHigh},
		{"gpt-5.6-terra", ReasoningMedium, shared.ReasoningEffortMedium},
		{"gpt-5.6-luna", ReasoningLow, shared.ReasoningEffortLow},
		{"gpt-5.6-sol", ReasoningNone, ""},
		{"unknown-model", ReasoningHigh, ""},
	}
	for _, tc := range cases {
		if got := openAIReasoningEffort(tc.model, tc.effort); got != tc.want {
			t.Errorf("openAIReasoningEffort(%q, %q) = %q, want %q", tc.model, tc.effort, got, tc.want)
		}
	}
}

// On a reasoning model, applying effort must drop temperature (rejected by
// those models); on a non-reasoning model, temperature flows as usual.
func TestApplyOpenAIOptionsEffortDropsTemperature(t *testing.T) {
	temp := 0.7
	p := openai.ChatCompletionNewParams{Model: shared.ChatModel(OpenAIModelGPT56Sol)}
	applyOpenAIOptions(&p, RequestOptions{ReasoningEffort: ReasoningHigh, Temperature: &temp})
	if p.ReasoningEffort != shared.ReasoningEffortHigh {
		t.Errorf("reasoning_effort = %q, want high", p.ReasoningEffort)
	}
	if p.Temperature.Valid() {
		t.Error("temperature must be unset on a reasoning model with effort")
	}

	p2 := openai.ChatCompletionNewParams{Model: shared.ChatModel("unknown-model")}
	applyOpenAIOptions(&p2, RequestOptions{ReasoningEffort: ReasoningHigh, Temperature: &temp})
	if p2.ReasoningEffort != "" {
		t.Error("unknown model must not get reasoning_effort")
	}
	if !p2.Temperature.Valid() {
		t.Error("temperature should flow on a non-reasoning model")
	}
}
