package llm

import (
	"bytes"
	"fmt"
	"io/fs"
	"text/template"
)

// Prompt loads and renders a prompt template from an embedded filesystem.
// The name should include the path relative to the embed root, e.g. "prompts/finalize.v1.md".
// Data is passed to Go's text/template engine — use nil for static prompts.
func Prompt(fsys fs.FS, name string, data any) (string, error) {
	content, err := fs.ReadFile(fsys, name)
	if err != nil {
		return "", fmt.Errorf("load prompt %s: %w", name, err)
	}

	// If no template data, return raw content.
	if data == nil {
		return string(content), nil
	}

	tmpl, err := template.New(name).Parse(string(content))
	if err != nil {
		return "", fmt.Errorf("parse prompt template %s: %w", name, err)
	}

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("execute prompt template %s: %w", name, err)
	}

	return buf.String(), nil
}

// MustPrompt is like Prompt but panics on error. Use for embed.FS prompts
// that are guaranteed to exist at compile time.
func MustPrompt(fsys fs.FS, name string, data any) string {
	s, err := Prompt(fsys, name, data)
	if err != nil {
		panic(err)
	}
	return s
}
