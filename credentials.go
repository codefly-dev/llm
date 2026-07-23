package llm

import "context"

// CredKind identifies which provider API key to resolve. The string values
// match Mind's credentials.Kind so a host store can adapt without translation.
type CredKind string

const (
	CredKindAnthropicAPI CredKind = "anthropic_api_key"
	CredKindOpenAIAPI    CredKind = "openai_api_key"
)

// APIKeyStore is the module's minimal view of a credential source: resolve a
// provider API key by kind. Hosts (Mind, codefly) adapt their own secret
// store to this interface. Returning ("", nil) means "not set" — the builder
// falls back to any explicitly provided key.
type APIKeyStore interface {
	APIKey(ctx context.Context, kind CredKind) (string, error)
}
