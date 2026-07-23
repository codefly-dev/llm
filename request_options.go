package llm

func defaultMaxTokens(opts RequestOptions, fallback int64) int64 {
	if opts.MaxTokens > 0 {
		return int64(opts.MaxTokens)
	}
	if fallback > 0 {
		return fallback
	}
	return 8192
}
