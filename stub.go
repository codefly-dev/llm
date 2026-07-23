package llm

import (
	"context"
	"fmt"
)

// StubError returns a Client that always returns an error.
//
// It exists for explicit negative-path and forbidden-boundary assertions only.
// Successful model behavior must come from a provider-backed cassette.
func StubError(msg string) Client {
	return &stubErrorClient{err: fmt.Errorf("%s", msg)}
}

type stubErrorClient struct{ err error }

func (s *stubErrorClient) Call(_ context.Context, _ string) (string, error) { return "", s.err }
func (s *stubErrorClient) Stream(_ context.Context, _ string, _ func(string) error) (string, error) {
	return "", s.err
}
func (s *stubErrorClient) CallCached(_ context.Context, _, _ string) (string, error) {
	return "", s.err
}
func (s *stubErrorClient) CallWithTools(_ context.Context, _ string, _ []Message, _ []ToolDef) (Message, error) {
	return Message{}, s.err
}
func (s *stubErrorClient) CallWithToolsOptions(_ context.Context, _ string, _ []Message, _ []ToolDef, _ RequestOptions) (Message, error) {
	return Message{}, s.err
}
