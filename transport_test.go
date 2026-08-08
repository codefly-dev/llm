package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	"go.opentelemetry.io/otel/attribute"
)

type transportCredentialStore struct {
	values map[CredKind]string
	err    error
}

func (s transportCredentialStore) APIKey(_ context.Context, kind CredKind) (string, error) {
	return s.values[kind], s.err
}

func gatewayProfile(baseURL string) ProviderTransportProfile {
	return ProviderTransportProfile{
		Identity: TransportIdentityGateway,
		BaseURL:  baseURL,
		Authentication: GatewayAuthentication{
			Header:         "x-warden-api-key",
			CredentialKind: CredKind("gateway_api_key"),
		},
		StaticHeaders: map[string]string{
			"X-Gateway-Protocol": "provider-compatible",
		},
		AllowInsecureLocalhost: true,
	}
}

func assertSecretAbsent(t *testing.T, text, secret string) {
	t.Helper()
	representation := secret
	seen := map[string]struct{}{}
	for range 3 {
		if _, ok := seen[representation]; ok {
			break
		}
		seen[representation] = struct{}{}
		if strings.Contains(text, representation) {
			t.Fatalf("secret representation %q appears in %q", representation, text)
		}
		encoded, err := json.Marshal(representation)
		if err != nil {
			t.Fatal(err)
		}
		representation = string(encoded[1 : len(encoded)-1])
	}
}

func TestProviderTransportProfileValidation(t *testing.T) {
	valid := gatewayProfile("https://gateway.example.com/anthropic")
	if err := validateProviderTransportProfile(valid); err != nil {
		t.Fatalf("valid profile: %v", err)
	}

	tests := map[string]ProviderTransportProfile{
		"relative URL": func() ProviderTransportProfile {
			p := valid
			p.BaseURL = "/gateway"
			return p
		}(),
		"URL userinfo": func() ProviderTransportProfile {
			p := valid
			p.BaseURL = "https://user:password@gateway.example.com"
			return p
		}(),
		"URL query": func() ProviderTransportProfile {
			p := valid
			p.BaseURL = "https://gateway.example.com?token=secret"
			return p
		}(),
		"insecure remote URL": func() ProviderTransportProfile {
			p := valid
			p.BaseURL = "http://gateway.example.com"
			return p
		}(),
		"insecure localhost without allowance": func() ProviderTransportProfile {
			p := valid
			p.BaseURL = "http://localhost:8080"
			p.AllowInsecureLocalhost = false
			return p
		}(),
		"direct identity": func() ProviderTransportProfile {
			p := valid
			p.Identity = TransportIdentityDirectProvider
			return p
		}(),
		"empty auth header": func() ProviderTransportProfile {
			p := valid
			p.Authentication.Header = ""
			return p
		}(),
		"unsafe auth header": func() ProviderTransportProfile {
			p := valid
			p.Authentication.Header = "Host"
			return p
		}(),
		"empty credential kind": func() ProviderTransportProfile {
			p := valid
			p.Authentication.CredentialKind = ""
			return p
		}(),
		"duplicate auth header": func() ProviderTransportProfile {
			p := valid
			p.StaticHeaders = map[string]string{"X-Warden-API-Key": "duplicate"}
			return p
		}(),
		"second secret header": func() ProviderTransportProfile {
			p := valid
			p.StaticHeaders = map[string]string{"Authorization": "Bearer duplicate"}
			return p
		}(),
		"case-insensitive duplicate header": func() ProviderTransportProfile {
			p := valid
			p.StaticHeaders = map[string]string{"X-Protocol": "one", "x-protocol": "two"}
			return p
		}(),
		"unsafe header": func() ProviderTransportProfile {
			p := valid
			p.StaticHeaders = map[string]string{"Transfer-Encoding": "chunked"}
			return p
		}(),
		"header injection": func() ProviderTransportProfile {
			p := valid
			p.StaticHeaders = map[string]string{"X-Protocol": "safe\r\nX-Injected: value"}
			return p
		}(),
	}
	for name, profile := range tests {
		t.Run(name, func(t *testing.T) {
			if err := validateProviderTransportProfile(profile); err == nil {
				t.Fatal("invalid profile was accepted")
			}
		})
	}
}

func TestProviderTransportProfileAllowsExplicitLoopbackHTTP(t *testing.T) {
	for _, baseURL := range []string{
		"http://localhost:8080/gateway",
		"http://127.0.0.1:8080/gateway",
		"http://[::1]:8080/gateway",
	} {
		profile := gatewayProfile(baseURL)
		if err := validateProviderTransportProfile(profile); err != nil {
			t.Errorf("%s: %v", baseURL, err)
		}
	}
}

func TestProviderTransportProfileRejectsEmptyGatewayCredential(t *testing.T) {
	const secret = "must-not-leak-from-store-error"
	_, err := NewClient(
		Model("anthropic/"+AnthropicModelHaiku),
		Options{},
		WithTransportProfile(gatewayProfile("https://gateway.example.com")),
		WithCredentialStore(transportCredentialStore{err: errors.New(secret)}),
	)
	if err == nil {
		t.Fatal("empty gateway credential was accepted")
	}
	if strings.Contains(err.Error(), secret) {
		t.Fatalf("credential-store error leaked a secret: %v", err)
	}
}

func TestProviderTransportFormattingRedactsSecrets(t *testing.T) {
	const secret = "gateway-secret-value"
	profile := gatewayProfile("https://user:" + secret + "@gateway.example.com/path?token=" + secret)
	profile.StaticHeaders["X-Protocol"] = secret
	resolved := providerTransport{
		identity:      TransportIdentityGateway,
		baseURL:       profile.BaseURL,
		authHeader:    profile.Authentication.Header,
		credential:    secret,
		staticHeaders: profile.StaticHeaders,
	}
	formatted := fmt.Sprintf("%v\n%+v\n%#v\n%v\n%+v\n%#v", profile, profile, profile, resolved, resolved, resolved)
	if strings.Contains(formatted, secret) {
		t.Fatalf("formatted transport configuration leaked credential: %s", formatted)
	}
}

func TestDirectProviderConstructionKeepsDirectIdentity(t *testing.T) {
	if got := NewAnthropic("anthropic-key", AnthropicModelHaiku).TransportIdentity(); got != TransportIdentityDirectProvider {
		t.Fatalf("Anthropic identity = %q", got)
	}
	if got := NewOpenAI("openai-key", OpenAIModelGPT56Terra).TransportIdentity(); got != TransportIdentityDirectProvider {
		t.Fatalf("OpenAI identity = %q", got)
	}
	client, err := NewClient(
		Model("anthropic/"+AnthropicModelHaiku),
		Options{AnthropicAPIKey: "anthropic-key"},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := ClientTransportIdentity(client); got != TransportIdentityDirectProvider {
		t.Fatalf("wrapped direct identity = %q", got)
	}
}

func TestGatewayProfilePreservesReplayWithoutLiveCredentials(t *testing.T) {
	client, err := NewClient(
		Model("anthropic/"+AnthropicModelHaiku),
		Options{},
		WithTransportProfile(gatewayProfile("https://gateway.example.com")),
		WithRecorder(t.TempDir(), RecordReplayOnly),
	)
	if err != nil {
		t.Fatal(err)
	}
	if got := ClientTransportIdentity(client); got != TransportIdentityGateway {
		t.Fatalf("replay transport identity = %q", got)
	}
}

func TestAnthropicGatewayTransportPreservesProviderCapabilities(t *testing.T) {
	const credential = "warden-anthropic-secret"
	var requestCount atomic.Int32
	var sawCacheControl atomic.Bool

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requestCount.Add(1)
		if r.URL.Path != "/compatible/v1/messages" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("X-Warden-API-Key"); got != credential {
			t.Errorf("gateway credential = %q", got)
		}
		if got := r.Header.Get("X-Api-Key"); got != "" {
			t.Errorf("vendor credential header was also sent: %q", got)
		}
		if got := r.Header.Get("Authorization"); got != "" {
			t.Errorf("unexpected Authorization header: %q", got)
		}
		if got := r.Header.Get("X-Gateway-Protocol"); got != "provider-compatible" {
			t.Errorf("static protocol header = %q", got)
		}

		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode request: %v", err)
			http.Error(w, "invalid request", http.StatusBadRequest)
			return
		}
		if system, ok := body["system"].([]any); ok && len(system) > 0 {
			block, _ := system[0].(map[string]any)
			if _, ok := block["cache_control"]; ok {
				sawCacheControl.Store(true)
			}
		}
		if streaming, _ := body["stream"].(bool); streaming {
			sseHeaders(w)
			writeSSE(w, "message_start", `{"type":"message_start","message":{"id":"msg_gateway","type":"message","role":"assistant","model":"claude-haiku-4-5-20251001","content":[],"stop_reason":null,"usage":{"input_tokens":7,"output_tokens":1,"cache_read_input_tokens":2,"cache_creation_input_tokens":3}}}`)
			writeSSE(w, "content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`)
			writeSSE(w, "content_block_delta", sseTextDelta("gateway "))
			writeSSE(w, "content_block_delta", sseTextDelta("stream"))
			writeSSE(w, "content_block_stop", `{"type":"content_block_stop","index":0}`)
			writeSSE(w, "message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}`)
			writeSSE(w, "message_stop", `{"type":"message_stop"}`)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_tools",
			"type":"message",
			"role":"assistant",
			"model":"claude-haiku-4-5-20251001",
			"content":[{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"query":"gateway"}}],
			"stop_reason":"tool_use",
			"usage":{"input_tokens":11,"output_tokens":4,"cache_read_input_tokens":5,"cache_creation_input_tokens":2}
		}`))
	}))
	defer server.Close()

	t.Setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1")
	t.Setenv("ANTHROPIC_API_KEY", "vendor-secret-that-must-not-be-used")
	profile := gatewayProfile(server.URL + "/compatible")
	client, err := NewClient(
		Model("anthropic/"+AnthropicModelHaiku),
		Options{},
		WithTransportProfile(profile),
		WithCredentialStore(transportCredentialStore{values: map[CredKind]string{
			profile.Authentication.CredentialKind: credential,
		}}),
	)
	if err != nil {
		t.Fatal(err)
	}

	var chunks []string
	full, err := client.Stream(t.Context(), "stream through gateway", func(chunk string) error {
		chunks = append(chunks, chunk)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if full != "gateway stream" || strings.Join(chunks, "") != full {
		t.Fatalf("stream = %q, chunks = %v", full, chunks)
	}

	cached, err := CallCachedWithOptions(t.Context(), client, "stable system", "fresh user", RequestOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if cached != "gateway stream" || !sawCacheControl.Load() {
		t.Fatalf("cached response = %q, cache control seen = %t", cached, sawCacheControl.Load())
	}

	message, err := CallWithToolsOptions(
		t.Context(),
		client,
		"use tools",
		[]Message{{Role: "user", Content: "look it up"}},
		[]ToolDef{{Name: "lookup", Parameters: []ToolParam{{Name: "query", Type: "string", Required: true}}}},
		RequestOptions{},
	)
	if err != nil {
		t.Fatal(err)
	}
	if len(message.ToolCalls) != 1 || message.ToolCalls[0].Name != "lookup" ||
		message.ToolCalls[0].Args["query"] != "gateway" {
		t.Fatalf("tool response = %+v", message)
	}

	usage, ok := AsUsageProvider(client)
	if !ok || usage.LastCallUsage() == nil {
		t.Fatal("gateway client lost usage accounting")
	}
	lastUsage := usage.LastCallUsage()
	if lastUsage.InputTokens != 11 || lastUsage.OutputTokens != 4 || lastUsage.CachedTokens != 5 {
		t.Fatalf("usage = %+v", lastUsage)
	}
	if got := ClientTransportIdentity(client); got != TransportIdentityGateway {
		t.Fatalf("transport identity = %q", got)
	}
	meta, ok := AsProviderMeta(client)
	if !ok || meta.Provider() != "anthropic" || meta.Model() != AnthropicModelHaiku {
		t.Fatalf("provider metadata = %v, %t", meta, ok)
	}
	if requestCount.Load() != 3 {
		t.Fatalf("requests = %d, want 3", requestCount.Load())
	}
}

func TestOpenAIGatewayTransportUsesCompatibleBaseAndGatewayAuth(t *testing.T) {
	const credential = "warden-openai-secret"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/compatible/v1/chat/completions" {
			t.Errorf("path = %q", r.URL.Path)
		}
		if got := r.Header.Get("X-Warden-API-Key"); got != credential {
			t.Errorf("gateway credential = %q", got)
		}
		for _, header := range []string{"Authorization", "OpenAI-Organization", "OpenAI-Project"} {
			if got := r.Header.Get(header); got != "" {
				t.Errorf("%s should be empty, got %q", header, got)
			}
		}
		if got := r.Header.Get("X-Gateway-Protocol"); got != "provider-compatible" {
			t.Errorf("static protocol header = %q", got)
		}
		sseHeaders(w)
		writeSSE(w, "", `{"id":"chatcmpl_gateway","object":"chat.completion.chunk","created":1,"model":"gpt-5.6-terra","choices":[{"index":0,"delta":{"content":"openai gateway"},"finish_reason":null}]}`)
		writeSSE(w, "", `{"id":"chatcmpl_gateway","object":"chat.completion.chunk","created":1,"model":"gpt-5.6-terra","choices":[],"usage":{"prompt_tokens":6,"completion_tokens":3,"total_tokens":9,"prompt_tokens_details":{"cached_tokens":2},"completion_tokens_details":{"reasoning_tokens":1}}}`)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	t.Setenv("OPENAI_BASE_URL", "https://invalid.example/%zz")
	t.Setenv("OPENAI_API_KEY", "vendor-secret-that-must-not-be-used")
	t.Setenv("OPENAI_ORG_ID", "environment-org")
	t.Setenv("OPENAI_PROJECT_ID", "environment-project")
	t.Setenv("OPENAI_WEBHOOK_SECRET", "environment-webhook-secret")
	profile := gatewayProfile(server.URL + "/compatible/v1")
	client, err := NewClient(
		Model("openai/"+OpenAIModelGPT56Terra),
		Options{},
		WithTransportProfile(profile),
		WithCredentialStore(transportCredentialStore{values: map[CredKind]string{
			profile.Authentication.CredentialKind: credential,
		}}),
	)
	if err != nil {
		t.Fatal(err)
	}

	response, err := client.Stream(t.Context(), "stream through gateway", nil)
	if err != nil {
		t.Fatal(err)
	}
	if response != "openai gateway" {
		t.Fatalf("response = %q", response)
	}
	usage, ok := AsUsageProvider(client)
	if !ok || usage.LastCallUsage() == nil {
		t.Fatal("gateway client lost usage accounting")
	}
	lastUsage := usage.LastCallUsage()
	if lastUsage.InputTokens != 6 || lastUsage.OutputTokens != 3 ||
		lastUsage.CachedTokens != 2 || lastUsage.ReasoningTokens != 1 {
		t.Fatalf("usage = %+v", lastUsage)
	}
}

func TestGatewayTransportPreservesHostedWebSearchCapability(t *testing.T) {
	profile := gatewayProfile("http://127.0.0.1:1")
	store := transportCredentialStore{values: map[CredKind]string{
		profile.Authentication.CredentialKind: "gateway-credential",
	}}

	anthropicClient, err := NewClient(
		Model("anthropic/"+AnthropicModelHaiku),
		Options{},
		WithTransportProfile(profile),
		WithCredentialStore(store),
	)
	if err != nil {
		t.Fatal(err)
	}
	if FindWebSearcher(anthropicClient) == nil {
		t.Fatal("Anthropic gateway lost hosted web search")
	}

	openAIClient, err := NewClient(
		Model("openai/"+OpenAIModelGPT56Terra),
		Options{},
		WithTransportProfile(profile),
		WithCredentialStore(store),
	)
	if err != nil {
		t.Fatal(err)
	}
	if searcher := FindWebSearcher(openAIClient); searcher != nil {
		t.Fatalf("OpenAI gateway advertised unsupported hosted web search through %T", searcher)
	}
}

func TestWithTransportProfileSnapshotsStaticHeaders(t *testing.T) {
	const credential = "snapshot-gateway-credential"
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("X-Gateway-Protocol"); got != "original" {
			t.Errorf("static header = %q, want original snapshot", got)
		}
		sseHeaders(w)
		writeSSE(w, "", `{"id":"chatcmpl_snapshot","object":"chat.completion.chunk","created":1,"model":"gpt-5.6-terra","choices":[{"index":0,"delta":{"content":"snapshot"},"finish_reason":null}]}`)
		_, _ = fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()

	profile := gatewayProfile(server.URL + "/v1")
	profile.StaticHeaders["X-Gateway-Protocol"] = "original"
	transportOption := WithTransportProfile(profile)
	profile.StaticHeaders["X-Gateway-Protocol"] = "mutated"

	client, err := NewClient(
		Model("openai/"+OpenAIModelGPT56Terra),
		Options{},
		transportOption,
		WithCredentialStore(transportCredentialStore{values: map[CredKind]string{
			profile.Authentication.CredentialKind: credential,
		}}),
	)
	if err != nil {
		t.Fatal(err)
	}
	response, err := client.Stream(t.Context(), "use captured profile", nil)
	if err != nil {
		t.Fatal(err)
	}
	if response != "snapshot" {
		t.Fatalf("response = %q", response)
	}
}

func TestGatewayErrorRedactionPreservesSafeTransportCause(t *testing.T) {
	cause := &url.Error{
		Op:  "Post",
		URL: "https://gateway.example.com/v1/messages",
		Err: context.DeadlineExceeded,
	}
	err := redactTransportError(&ProviderError{
		Provider: "anthropic",
		Code:     CodeTimeout,
		Message:  cause.Error(),
		Wrapped:  cause,
	}, "gateway-credential")

	var unwrapped *url.Error
	if !errors.As(err, &unwrapped) {
		t.Fatalf("gateway error no longer unwraps to its transport cause: %T", err)
	}
	if unwrapped != cause {
		t.Fatal("safe transport cause identity was not preserved")
	}
}

func TestGatewaySecretsAreRedactedFromErrorsAndCassettes(t *testing.T) {
	const credential = "gateway-credential-\"quoted\\slash<&>"
	encodedCredential, err := json.Marshal(credential)
	if err != nil {
		t.Fatal(err)
	}
	gatewayEncodedCredential := strings.ReplaceAll(
		string(encodedCredential[1:len(encodedCredential)-1]),
		`\"`,
		`\u0022`,
	)
	responseBody := `{"type":"error","error":{"type":"authentication_error","message":"rejected ` +
		gatewayEncodedCredential + `"}}`
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Request-ID", credential)
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(responseBody))
	}))
	defer server.Close()

	profile := gatewayProfile(server.URL)
	cassetteDir := t.TempDir()
	client, err := NewClient(
		Model("anthropic/"+AnthropicModelHaiku),
		Options{},
		WithTransportProfile(profile),
		WithCredentialStore(transportCredentialStore{values: map[CredKind]string{
			profile.Authentication.CredentialKind: credential,
		}}),
		WithRecorder(cassetteDir, RecordAlways),
	)
	if err != nil {
		t.Fatal(err)
	}
	_, err = client.Call(t.Context(), "trigger classified gateway error")
	if err == nil {
		t.Fatal("expected gateway error")
	}
	if !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("classified error = %v", err)
	}
	formattedError := fmt.Sprintf("%v\n%+v\n%#v", err, err, err)
	assertSecretAbsent(t, formattedError, credential)
	assertSecretAbsent(t, formattedError, gatewayEncodedCredential)

	var sdkErr *anthropic.Error
	if !errors.As(err, &sdkErr) {
		t.Fatalf("gateway error no longer unwraps to the Anthropic SDK error: %T", err)
	}
	for name, diagnostic := range map[string]string{
		"error":         sdkErr.Error(),
		"raw JSON":      sdkErr.RawJSON(),
		"request dump":  string(sdkErr.DumpRequest(false)),
		"response dump": string(sdkErr.DumpResponse(false)),
	} {
		t.Run(name, func(t *testing.T) {
			assertSecretAbsent(t, diagnostic, credential)
			assertSecretAbsent(t, diagnostic, gatewayEncodedCredential)
		})
	}

	cassetteFiles := 0
	err = filepath.WalkDir(cassetteDir, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() {
			return walkErr
		}
		cassetteFiles++
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		assertSecretAbsent(t, string(data), credential)
		assertSecretAbsent(t, string(data), gatewayEncodedCredential)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if cassetteFiles == 0 {
		t.Fatal("gateway error did not produce a cassette")
	}
}

func TestDirectProviderSecretsAreRedactedFromErrorsAndCassettes(t *testing.T) {
	const credential = `direct-provider-credential-"quoted\slash<&>`
	providerErr := &ProviderError{
		Provider:  "openai",
		Code:      CodeUnauthorized,
		Status:    http.StatusUnauthorized,
		Message:   "provider rejected " + credential,
		RequestID: credential,
		Wrapped:   errors.New("direct provider transport echoed " + credential),
	}
	inner := &recorderInfraClient{err: providerErr}
	cassetteDir := t.TempDir()
	client := RecorderMiddleware(cassetteDir, "openai/test-model", RecordAlways)(
		protectProviderErrors(inner, credential, nil),
	)

	_, err := client.Call(t.Context(), "trigger classified direct-provider error")
	if err == nil {
		t.Fatal("expected direct-provider error")
	}
	if !errors.Is(err, ErrUnauthorized) {
		t.Fatalf("classified error = %v", err)
	}
	var classified *ProviderError
	if !errors.As(err, &classified) {
		t.Fatalf("direct-provider error no longer exposes ProviderError: %T", err)
	}
	assertSecretAbsent(t, fmt.Sprintf("%v\n%+v\n%#v", err, err, err), credential)

	cassetteFiles := 0
	err = filepath.WalkDir(cassetteDir, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() {
			return walkErr
		}
		cassetteFiles++
		data, readErr := os.ReadFile(path)
		if readErr != nil {
			return readErr
		}
		assertSecretAbsent(t, string(data), credential)
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if cassetteFiles != 1 {
		t.Fatalf("direct-provider error cassette files = %d, want 1", cassetteFiles)
	}

	replayInner := &recorderInfraClient{response: Message{Content: "must not run"}}
	replay := RecorderMiddleware(cassetteDir, "openai/test-model", RecordReplayOnly)(replayInner)
	_, replayErr := replay.Call(t.Context(), "trigger classified direct-provider error")
	if !errors.Is(replayErr, ErrUnauthorized) {
		t.Fatalf("replay lost direct-provider classification: %v", replayErr)
	}
	if replayInner.calls != 0 {
		t.Fatalf("replay called direct provider %d times", replayInner.calls)
	}
}

func TestGatewayIdentityIsIncludedInTraceAttributes(t *testing.T) {
	middleware := &otelMW{
		model:             "anthropic/" + AnthropicModelHaiku,
		transportIdentity: TransportIdentityGateway,
	}
	attrs := middleware.baseLLMAttrs("chat")
	attrMap := make(map[attribute.Key]attribute.Value, len(attrs))
	for _, attr := range attrs {
		attrMap[attr.Key] = attr.Value
	}
	if got := attrMap["llm.transport.identity"].AsString(); got != string(TransportIdentityGateway) {
		t.Fatalf("trace transport identity = %q", got)
	}
}
