package llm

import "strings"

// ErrorComplexity classifies how hard a failure is to resolve.
type ErrorComplexity int

const (
	ErrorSimple  ErrorComplexity = iota // compile error, missing import
	ErrorMedium                         // test failure with stack trace
	ErrorComplex                        // repeated failures, logic errors
)

// ModelRouter routes LLM calls to cheap/strong/expert clients based on
// iteration number, error complexity, and stuck-loop detection.
type ModelRouter struct {
	Cheap  Client // fast, cheap model (Haiku, GPT-4o-mini)
	Strong Client // default capable model (Sonnet, GPT-4o)
	Expert Client // most capable model (Opus) — may be nil
}

// Route selects the appropriate client for the current iteration state.
// For difficulty-aware routing, use RouteWithDifficulty instead.
func (r *ModelRouter) Route(iteration int, complexity ErrorComplexity, stuck bool) Client {
	if stuck && r.Expert != nil {
		return r.Expert
	}
	if stuck {
		return r.Strong
	}

	switch {
	case iteration == 1:
		return r.Strong
	case complexity == ErrorSimple:
		return r.cheapOrStrong()
	case complexity >= ErrorMedium:
		return r.Strong
	default:
		return r.Strong
	}
}

// RouteWithDifficulty selects the appropriate client considering objective difficulty.
// Difficulty levels: 0=trivial, 1=easy, 2=medium, 3=hard, 4=expert.
// Uses int to avoid circular dependency on pkg/objective.
//
// Priority: stuck > difficulty > error complexity > iteration.
// Stuck always escalates to Expert (or Strong if Expert unavailable).
// Difficulty drives the baseline: trivial/easy→Cheap, medium→Strong, hard/expert→Expert.
// Error complexity and iteration are the fallback when difficulty is not set.
func (r *ModelRouter) RouteWithDifficulty(iteration int, complexity ErrorComplexity, stuck bool, difficulty int) Client {
	// Stuck always escalates.
	if stuck && r.Expert != nil {
		return r.Expert
	}
	if stuck {
		return r.Strong
	}

	// Difficulty-based routing (when explicitly set, difficulty >= 0).
	if difficulty >= 0 {
		switch {
		case difficulty <= 1: // trivial, easy
			return r.cheapOrStrong()
		case difficulty == 2: // medium
			return r.Strong
		default: // hard (3), expert (4+)
			if r.Expert != nil {
				return r.Expert
			}
			return r.Strong
		}
	}

	// Fallback: existing error-complexity + iteration logic.
	return r.Route(iteration, complexity, false)
}

func (r *ModelRouter) cheapOrStrong() Client {
	if r.Cheap != nil {
		return r.Cheap
	}
	return r.Strong
}

// CheapOrStrong returns the cheap client if set, otherwise strong.
func (r *ModelRouter) CheapOrStrong() Client {
	return r.cheapOrStrong()
}

// ClassifyErrors determines the complexity of failures for model routing decisions.
// See also: act.classifyError which categorizes by error nature (for strategy decisions).
func ClassifyErrors(errors []string) ErrorComplexity {
	if len(errors) == 0 {
		return ErrorSimple
	}

	joined := strings.Join(errors, "\n")
	lower := strings.ToLower(joined)

	complexPatterns := []string{
		"panic", "deadlock", "race condition", "segfault",
		"logic error", "infinite loop",
	}
	for _, p := range complexPatterns {
		if strings.Contains(lower, p) {
			return ErrorComplex
		}
	}

	simplePatterns := []string{
		"undefined:", "cannot find", "not found",
		"imported and not used", "declared and not used",
		"missing import", "syntax error",
		"expected", "unexpected",
	}
	allSimple := true
	for _, e := range errors {
		eLower := strings.ToLower(e)
		isSimple := false
		for _, p := range simplePatterns {
			if strings.Contains(eLower, p) {
				isSimple = true
				break
			}
		}
		if !isSimple {
			allSimple = false
			break
		}
	}
	if allSimple {
		return ErrorSimple
	}

	return ErrorMedium
}
