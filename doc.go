// Package llm owns Mind's provider-facing language-model API.
//
// Use this package for provider clients, model identifiers, messages, native
// tool-calling schemas, request options, streaming blocks, usage/cost tracking,
// middleware, record/replay, and provider-specific helpers.
//
// Do not add harness or supervisor runtime contracts here. Those belong in
// pkg/spec so the production harness, daemon, trace, and tests share one stable
// contract surface without importing provider implementation code.
package llm
