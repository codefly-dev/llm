package llm

import "testing"

func TestRegistry_BuiltinsRegistered(t *testing.T) {
	got := RegisteredProviders()
	if len(got) != 2 || got[0] != "anthropic" || got[1] != "openai" {
		t.Fatalf("registered providers = %v, want exact current families [anthropic openai]", got)
	}
}

func TestRegistry_ResolveProviderSpec(t *testing.T) {
	spec, ok := resolveProviderSpec("anthropic")
	if !ok {
		t.Fatal("anthropic not registered")
	}
	if spec.EnvKey != "ANTHROPIC_API_KEY" || spec.DefaultModel == "" || spec.Factory == nil {
		t.Fatalf("anthropic spec incomplete: %+v", spec)
	}

	if _, ok := resolveProviderSpec("no-such-provider"); ok {
		t.Fatal("nonexistent provider must not resolve")
	}
}

func TestExplicitKeyFor(t *testing.T) {
	opts := Options{
		AnthropicAPIKey: "akey",
		OpenAIAPIKey:    "okey",
	}
	cases := map[string]string{
		"anthropic": "akey",
		"openai":    "okey",
		"unknown":   "",
	}
	for name, want := range cases {
		if got := explicitKeyFor(name, opts); got != want {
			t.Errorf("explicitKeyFor(%q) = %q, want %q", name, got, want)
		}
	}
}
