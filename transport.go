package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"sort"
	"strings"

	anthropic "github.com/anthropics/anthropic-sdk-go"
	openai "github.com/openai/openai-go"
)

// TransportIdentity identifies whether provider traffic goes to the provider
// itself or through a provider-compatible gateway.
type TransportIdentity string

const (
	TransportIdentityDirectProvider TransportIdentity = "direct_provider"
	TransportIdentityGateway        TransportIdentity = "gateway"
)

// GatewayAuthentication selects the request header and host credential-store
// entry used to authenticate a provider-compatible gateway.
type GatewayAuthentication struct {
	// Header is the single request header carrying the gateway credential.
	Header string
	// CredentialKind is resolved through the APIKeyStore passed to
	// WithCredentialStore.
	CredentialKind CredKind
}

// ProviderTransportProfile configures a built-in provider to use a compatible
// gateway through the provider's official SDK transport.
type ProviderTransportProfile struct {
	// Identity must be TransportIdentityGateway.
	Identity TransportIdentity
	// BaseURL ends immediately before the provider SDK's endpoint path, such
	// as /v1/messages for Anthropic or /chat/completions for OpenAI.
	BaseURL string
	// Authentication replaces the provider's credential header.
	Authentication GatewayAuthentication
	// StaticHeaders contains non-secret protocol-negotiation headers.
	StaticHeaders map[string]string
	// AllowInsecureLocalhost permits HTTP only for localhost and loopback IPs.
	AllowInsecureLocalhost bool
}

func (p ProviderTransportProfile) String() string {
	headerNames := make([]string, 0, len(p.StaticHeaders))
	for name := range p.StaticHeaders {
		headerNames = append(headerNames, name)
	}
	sort.Strings(headerNames)
	return fmt.Sprintf(
		"ProviderTransportProfile{Identity:%q BaseURL:%q Authentication:{Header:%q CredentialKind:%q} StaticHeaders:%v AllowInsecureLocalhost:%t}",
		p.Identity,
		diagnosticBaseURL(p.BaseURL),
		p.Authentication.Header,
		p.Authentication.CredentialKind,
		headerNames,
		p.AllowInsecureLocalhost,
	)
}

func (p ProviderTransportProfile) GoString() string { return p.String() }

// TransportMeta exposes a client's provider transport identity for diagnostics.
type TransportMeta interface {
	TransportIdentity() TransportIdentity
}

// ClientTransportIdentity returns the transport identity exposed by a client
// or one of its wrapped clients. It returns an empty identity when unavailable.
func ClientTransportIdentity(client Client) TransportIdentity {
	for client != nil {
		if meta, ok := client.(TransportMeta); ok {
			return meta.TransportIdentity()
		}
		unwrapper, ok := client.(interface{ Unwrap() Client })
		if !ok {
			return ""
		}
		client = unwrapper.Unwrap()
	}
	return ""
}

type providerTransport struct {
	identity      TransportIdentity
	baseURL       string
	authHeader    string
	credential    string
	staticHeaders map[string]string
}

func (t providerTransport) String() string {
	headerNames := make([]string, 0, len(t.staticHeaders))
	for name := range t.staticHeaders {
		headerNames = append(headerNames, name)
	}
	sort.Strings(headerNames)
	return fmt.Sprintf(
		"providerTransport{identity:%q baseURL:%q authHeader:%q credential:[REDACTED] staticHeaders:%v}",
		t.identity,
		diagnosticBaseURL(t.baseURL),
		t.authHeader,
		headerNames,
	)
}

func (t providerTransport) GoString() string { return t.String() }

func resolveProviderTransport(profile ProviderTransportProfile, store APIKeyStore) (*providerTransport, error) {
	if err := validateProviderTransportProfile(profile); err != nil {
		return nil, err
	}
	credential := resolveAPIKey(store, "", profile.Authentication.CredentialKind, "")
	if strings.TrimSpace(credential) == "" {
		return nil, fmt.Errorf("provider transport profile gateway credential is empty")
	}
	if !validHeaderValue(credential) {
		return nil, fmt.Errorf("provider transport profile gateway credential is invalid")
	}
	baseURL, _ := validateProviderBaseURL(profile.BaseURL, profile.AllowInsecureLocalhost)
	staticHeaders, _ := validateStaticHeaders(profile.StaticHeaders, profile.Authentication.Header)
	return &providerTransport{
		identity:      profile.Identity,
		baseURL:       baseURL,
		authHeader:    profile.Authentication.Header,
		credential:    credential,
		staticHeaders: staticHeaders,
	}, nil
}

func validateProviderTransportProfile(profile ProviderTransportProfile) error {
	if profile.Identity != TransportIdentityGateway {
		return fmt.Errorf("provider transport profile identity must be gateway")
	}
	if _, err := validateProviderBaseURL(profile.BaseURL, profile.AllowInsecureLocalhost); err != nil {
		return err
	}
	authHeader := strings.TrimSpace(profile.Authentication.Header)
	if authHeader != profile.Authentication.Header || !validHeaderName(authHeader) {
		return fmt.Errorf("provider transport profile authentication header is invalid")
	}
	if unsafeStaticHeader(strings.ToLower(authHeader)) {
		return fmt.Errorf("provider transport profile authentication header is unsafe")
	}
	if profile.Authentication.CredentialKind == "" {
		return fmt.Errorf("provider transport profile credential kind is empty")
	}
	_, err := validateStaticHeaders(profile.StaticHeaders, authHeader)
	return err
}

func validateProviderBaseURL(raw string, allowInsecureLocalhost bool) (string, error) {
	if raw == "" || strings.TrimSpace(raw) != raw {
		return "", fmt.Errorf("provider transport profile base URL is invalid")
	}
	parsed, err := url.Parse(raw)
	if err != nil || !parsed.IsAbs() || parsed.Opaque != "" || parsed.Host == "" ||
		parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" {
		return "", fmt.Errorf("provider transport profile base URL is invalid")
	}
	switch strings.ToLower(parsed.Scheme) {
	case "https":
	case "http":
		if !allowInsecureLocalhost || !isLoopbackHost(parsed.Hostname()) {
			return "", fmt.Errorf("provider transport profile requires HTTPS outside explicitly allowed localhost")
		}
	default:
		return "", fmt.Errorf("provider transport profile base URL scheme is invalid")
	}
	return parsed.String(), nil
}

func isLoopbackHost(host string) bool {
	host = strings.TrimSuffix(strings.ToLower(host), ".")
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func validateStaticHeaders(headers map[string]string, authHeader string) (map[string]string, error) {
	out := make(map[string]string, len(headers))
	seen := make(map[string]struct{}, len(headers))
	authHeader = strings.ToLower(authHeader)
	for name, value := range headers {
		trimmedName := strings.TrimSpace(name)
		canonicalName := strings.ToLower(trimmedName)
		if trimmedName != name || !validHeaderName(name) || !validHeaderValue(value) {
			return nil, fmt.Errorf("provider transport profile static header is invalid")
		}
		if _, exists := seen[canonicalName]; exists {
			return nil, fmt.Errorf("provider transport profile contains duplicate static headers")
		}
		seen[canonicalName] = struct{}{}
		if canonicalName == authHeader || secretHeader(canonicalName) {
			return nil, fmt.Errorf("provider transport profile contains duplicate secret headers")
		}
		if unsafeStaticHeader(canonicalName) {
			return nil, fmt.Errorf("provider transport profile contains an unsafe static header")
		}
		out[name] = value
	}
	return out, nil
}

func validHeaderName(name string) bool {
	if name == "" {
		return false
	}
	for i := 0; i < len(name); i++ {
		c := name[i]
		if ('a' <= c && c <= 'z') || ('A' <= c && c <= 'Z') || ('0' <= c && c <= '9') {
			continue
		}
		switch c {
		case '!', '#', '$', '%', '&', '\'', '*', '+', '-', '.', '^', '_', '`', '|', '~':
		default:
			return false
		}
	}
	return true
}

func validHeaderValue(value string) bool {
	for i := 0; i < len(value); i++ {
		if value[i] == '\t' {
			continue
		}
		if value[i] < 0x20 || value[i] == 0x7f {
			return false
		}
	}
	return true
}

func secretHeader(name string) bool {
	switch name {
	case "authorization", "proxy-authorization", "cookie", "set-cookie", "api-key", "token", "secret":
		return true
	}
	return strings.HasSuffix(name, "-api-key") ||
		strings.HasSuffix(name, "-token") ||
		strings.HasSuffix(name, "-secret")
}

func unsafeStaticHeader(name string) bool {
	switch name {
	case "connection", "content-length", "host", "keep-alive", "proxy-connection",
		"te", "trailer", "transfer-encoding", "upgrade":
		return true
	default:
		return false
	}
}

func diagnosticBaseURL(raw string) string {
	parsed, err := url.Parse(raw)
	if err != nil || !parsed.IsAbs() || parsed.Host == "" {
		return "<invalid>"
	}
	parsed.User = nil
	parsed.RawQuery = ""
	parsed.Fragment = ""
	return parsed.String()
}

func redactCredential(text, credential string) string {
	if text == "" || credential == "" {
		return text
	}
	forms := []string{credential}
	if encoded, err := json.Marshal(credential); err == nil && len(encoded) >= 2 {
		forms = append(forms, string(encoded[1:len(encoded)-1]))
	}
	sort.SliceStable(forms, func(i, j int) bool {
		return len(forms[i]) > len(forms[j])
	})
	seen := make(map[string]struct{}, len(forms))
	for _, form := range forms {
		if form == "" {
			continue
		}
		if _, ok := seen[form]; ok {
			continue
		}
		seen[form] = struct{}{}
		text = strings.ReplaceAll(text, form, "[REDACTED]")
	}
	return text
}

func redactHTTPHeader(header http.Header, credential string) http.Header {
	if header == nil {
		return nil
	}
	safe := header.Clone()
	for name, values := range safe {
		for i, value := range values {
			values[i] = redactCredential(value, credential)
		}
		safe[name] = values
	}
	return safe
}

func redactHTTPRequest(request *http.Request, credential string) *http.Request {
	if request == nil {
		return nil
	}
	safe := request.Clone(request.Context())
	safe.Header = redactHTTPHeader(request.Header, credential)
	safe.Trailer = redactHTTPHeader(request.Trailer, credential)
	safe.Host = redactCredential(request.Host, credential)
	if request.URL != nil {
		redactedURL, err := url.Parse(redactCredential(request.URL.String(), credential))
		if err == nil {
			safe.URL = redactedURL
		} else {
			safe.URL = &url.URL{Path: "[REDACTED]"}
		}
	}
	return safe
}

func redactHTTPResponse(response *http.Response, credential string) *http.Response {
	if response == nil {
		return nil
	}
	safe := new(http.Response)
	*safe = *response
	safe.Header = redactHTTPHeader(response.Header, credential)
	safe.Trailer = redactHTTPHeader(response.Trailer, credential)
	safe.Status = redactCredential(response.Status, credential)
	safe.Request = redactHTTPRequest(response.Request, credential)
	safe.Body = http.NoBody
	safe.ContentLength = 0
	safe.Header.Del("Content-Length")
	return safe
}

func redactJSONCredential(raw, credential string) (string, error) {
	var value any
	decoder := json.NewDecoder(strings.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&value); err != nil {
		return "", err
	}
	var redactValue func(any) any
	redactValue = func(value any) any {
		switch value := value.(type) {
		case string:
			return strings.ReplaceAll(value, credential, "[REDACTED]")
		case []any:
			for i, item := range value {
				value[i] = redactValue(item)
			}
		case map[string]any:
			safe := make(map[string]any, len(value))
			for name, item := range value {
				safe[strings.ReplaceAll(name, credential, "[REDACTED]")] = redactValue(item)
			}
			return safe
		}
		return value
	}
	encoded, err := json.Marshal(redactValue(value))
	return string(encoded), err
}

func redactAnthropicError(source *anthropic.Error, credential string) error {
	safe := *source
	raw := source.RawJSON()
	if raw != "" {
		redactedRaw, err := redactJSONCredential(raw, credential)
		if err != nil {
			return errors.New("anthropic SDK error redacted")
		}
		var rebuilt anthropic.Error
		if err := json.Unmarshal([]byte(redactedRaw), &rebuilt); err != nil {
			return errors.New("anthropic SDK error redacted")
		}
		safe = rebuilt
	}
	safe.StatusCode = source.StatusCode
	safe.Request = redactHTTPRequest(source.Request, credential)
	safe.Response = redactHTTPResponse(source.Response, credential)
	safe.RequestID = redactCredential(source.RequestID, credential)
	return &safe
}

func redactOpenAIError(source *openai.Error, credential string) error {
	safe := *source
	raw := source.RawJSON()
	if raw != "" {
		redactedRaw, err := redactJSONCredential(raw, credential)
		if err != nil {
			return errors.New("openai SDK error redacted")
		}
		var rebuilt openai.Error
		if err := json.Unmarshal([]byte(redactedRaw), &rebuilt); err != nil {
			return errors.New("openai SDK error redacted")
		}
		safe = rebuilt
	}
	safe.StatusCode = source.StatusCode
	safe.Request = redactHTTPRequest(source.Request, credential)
	safe.Response = redactHTTPResponse(source.Response, credential)
	return &safe
}

func redactWrappedTransportError(err error, credential string) error {
	if err == nil {
		return nil
	}
	message := redactCredential(err.Error(), credential)
	var anthropicErr *anthropic.Error
	if errors.As(err, &anthropicErr) {
		return redactAnthropicError(anthropicErr, credential)
	}
	var openAIErr *openai.Error
	if errors.As(err, &openAIErr) {
		return redactOpenAIError(openAIErr, credential)
	}
	if message == err.Error() {
		return err
	}
	switch {
	case errors.Is(err, ErrStreamStalled):
		return fmt.Errorf("%w: %s", ErrStreamStalled, message)
	case errors.Is(err, context.DeadlineExceeded):
		return fmt.Errorf("%w: %s", context.DeadlineExceeded, message)
	case errors.Is(err, context.Canceled):
		return fmt.Errorf("%w: %s", context.Canceled, message)
	default:
		return errors.New(message)
	}
}

func redactTransportError(err error, credential string) error {
	if err == nil || credential == "" {
		return err
	}
	redact := func(text string) string {
		return redactCredential(text, credential)
	}
	var providerErr *ProviderError
	if errors.As(err, &providerErr) {
		safe := *providerErr
		safe.Message = redact(providerErr.Message)
		safe.RequestID = redact(providerErr.RequestID)
		safe.Wrapped = redactWrappedTransportError(providerErr.Wrapped, credential)
		var anthropicErr *anthropic.Error
		var openAIErr *openai.Error
		if errors.As(providerErr.Wrapped, &anthropicErr) || errors.As(providerErr.Wrapped, &openAIErr) {
			safe.Message = safe.Wrapped.Error()
		}
		return &safe
	}
	return redactWrappedTransportError(err, credential)
}

type transportRedactionClient struct {
	next       Client
	credential string
}

func redactTransportErrors(next Client, credential string) Client {
	redacted := &transportRedactionClient{next: next, credential: credential}
	if searcher := FindWebSearcher(next); searcher != nil {
		return &transportRedactionWebSearchClient{
			transportRedactionClient: redacted,
			searcher:                 searcher,
		}
	}
	return redacted
}

func (c *transportRedactionClient) Call(ctx context.Context, prompt string) (string, error) {
	response, err := c.next.Call(ctx, prompt)
	return response, redactTransportError(err, c.credential)
}

func (c *transportRedactionClient) CallWithOptions(ctx context.Context, prompt string, opts RequestOptions) (string, error) {
	response, err := CallWithOptions(ctx, c.next, prompt, opts)
	return response, redactTransportError(err, c.credential)
}

func (c *transportRedactionClient) Stream(ctx context.Context, prompt string, onChunk func(string) error) (string, error) {
	response, err := c.next.Stream(ctx, prompt, onChunk)
	return response, redactTransportError(err, c.credential)
}

func (c *transportRedactionClient) StreamWithOptions(ctx context.Context, prompt string, opts RequestOptions, onChunk func(string) error) (string, error) {
	response, err := StreamWithOptions(ctx, c.next, prompt, opts, onChunk)
	return response, redactTransportError(err, c.credential)
}

func (c *transportRedactionClient) CallCached(ctx context.Context, system, user string) (string, error) {
	response, err := CallWithCaching(ctx, c.next, system, user)
	return response, redactTransportError(err, c.credential)
}

func (c *transportRedactionClient) CallCachedWithOptions(ctx context.Context, system, user string, opts RequestOptions) (string, error) {
	response, err := CallCachedWithOptions(ctx, c.next, system, user, opts)
	return response, redactTransportError(err, c.credential)
}

func (c *transportRedactionClient) CallWithTools(ctx context.Context, system string, messages []Message, tools []ToolDef) (Message, error) {
	response, err := CallWithToolsOptions(ctx, c.next, system, messages, tools, RequestOptions{})
	return response, redactTransportError(err, c.credential)
}

func (c *transportRedactionClient) CallWithToolsOptions(ctx context.Context, system string, messages []Message, tools []ToolDef, opts RequestOptions) (Message, error) {
	response, err := CallWithToolsOptions(ctx, c.next, system, messages, tools, opts)
	return response, redactTransportError(err, c.credential)
}

func (c *transportRedactionClient) StreamEvents(ctx context.Context, req StreamRequest) (<-chan StreamEvent, error) {
	eventClient, ok := AsEventStreamClient(c.next)
	if !ok {
		return nil, fmt.Errorf("llm: client %T does not implement event streaming", c.next)
	}
	events, err := eventClient.StreamEvents(ctx, req)
	if err != nil {
		return nil, redactTransportError(err, c.credential)
	}
	safeEvents := make(chan StreamEvent, cap(events))
	go func() {
		defer close(safeEvents)
		for event := range events {
			if event.Error != "" {
				event.Error = redactCredential(event.Error, c.credential)
			}
			select {
			case safeEvents <- event:
			case <-ctx.Done():
				return
			}
		}
	}()
	return safeEvents, nil
}

func (c *transportRedactionClient) Unwrap() Client { return c.next }

type transportRedactionWebSearchClient struct {
	*transportRedactionClient
	searcher HostedWebSearcher
}

func (c *transportRedactionWebSearchClient) WebSearch(ctx context.Context, query string, maxResults int) (WebSearchResult, error) {
	result, err := c.searcher.WebSearch(ctx, query, maxResults)
	return result, redactTransportError(err, c.credential)
}

var (
	_ Client                  = (*transportRedactionClient)(nil)
	_ OptionedClient          = (*transportRedactionClient)(nil)
	_ CacheableClient         = (*transportRedactionClient)(nil)
	_ OptionedCacheableClient = (*transportRedactionClient)(nil)
	_ ToolClient              = (*transportRedactionClient)(nil)
	_ OptionedToolClient      = (*transportRedactionClient)(nil)
	_ EventStreamClient       = (*transportRedactionClient)(nil)
	_ HostedWebSearcher       = (*transportRedactionWebSearchClient)(nil)
)
