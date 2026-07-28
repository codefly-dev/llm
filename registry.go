package llm

// ARCHITECTURE: immutable executable-provider registry.
//
// Mind supports exactly two production LLM families. Adding one is an explicit
// source change that must include a current model roster and family policy.
// Tests and callers cannot mutate this table at runtime; WithProvider remains
// the explicit seam for infrastructure probes and non-production integrations.
//
// What the registry knows per provider:
//   - canonical name ("anthropic", "openai")
//   - API-key diagnostic env var name and credential kind
//   - default model id when the caller omits a "/model" suffix
//   - factory that constructs the provider Client from (apiKey, model)

import (
	"sort"
)

// ProviderSpec declares everything NewClient needs to build an instance of a
// provider: how to find its API key, what model to default to, and how to
// construct it.
type ProviderSpec struct {
	// Name is the canonical provider identifier used in Model strings
	// ("anthropic/..." → Name "anthropic"). Must be lowercase.
	Name string

	// EnvKey is the provider's conventional API key env-var name. It is used
	// only in diagnostics; key resolution flows through APIKeyStore.
	EnvKey string

	// CredKind is the CredKind used to look up the API key in the
	// credential store.
	CredKind CredKind

	// DefaultModel is the API model id used when the caller passes a bare
	// provider name ("anthropic") without "/model".
	DefaultModel string

	// Factory constructs the provider's Client from a resolved provider API key
	// or gateway transport and model id.
	Factory func(apiKey, model string, transport *providerTransport) Client
}

var registry = map[string]ProviderSpec{
	"anthropic": {
		Name:         "anthropic",
		EnvKey:       "ANTHROPIC_API_KEY",
		CredKind:     CredKindAnthropicAPI,
		DefaultModel: AnthropicModelSonnet,
		Factory: func(key, model string, transport *providerTransport) Client {
			if transport != nil {
				return newAnthropicWithTransport(model, *transport)
			}
			return NewAnthropic(key, model)
		},
	},
	"openai": {
		Name:         "openai",
		EnvKey:       "OPENAI_API_KEY",
		CredKind:     CredKindOpenAIAPI,
		DefaultModel: OpenAIModelGPT56Terra,
		Factory: func(key, model string, transport *providerTransport) Client {
			if transport != nil {
				return newOpenAIWithTransport(model, *transport)
			}
			return NewOpenAI(key, model)
		},
	},
}

// resolveProviderSpec looks up a provider by name. Returns false when the
// name is not registered.
func resolveProviderSpec(name string) (ProviderSpec, bool) {
	s, ok := registry[name]
	return s, ok
}

// RegisteredProviders returns the executable providers in stable order so
// diagnostics and cassette failures are byte-identical across runs.
func RegisteredProviders() []string {
	out := make([]string, 0, len(registry))
	for name := range registry {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// explicitKeyFor returns the caller-supplied API key for a provider when
// the caller set one via the typed Options fields. Registered providers
// not covered here simply fall back to the credential store / env chain.
func explicitKeyFor(providerName string, opts Options) string {
	switch providerName {
	case "anthropic":
		return opts.AnthropicAPIKey
	case "openai":
		return opts.OpenAIAPIKey
	}
	return ""
}
