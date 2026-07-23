package llm

// ARCHITECTURE: the Block / BlockSchema contract lives in pkg/spec.
// This file provides TWO example schemas. They are illustrative — callers
// with different prompt formats implement their own BlockSchema directly.
// Neither the markers nor the Kind strings are hard-coded anywhere; the
// caller supplies both. Every kind in every emitted Block is a string that
// came from the caller's config.

import (
	"strings"
)

// XMLTagSchema parses XML-style tagged regions: `<tag>content</tag>`.
//
// Callers list the tags they want to recognize. A recognized tag emits
// Block events with Kind equal to the tag name. Text between or outside
// recognized tags is emitted with Kind = OutsideKind — or dropped entirely
// when OutsideKind is "".
//
// The schema tolerates fragmentation: markers split across Feed calls are
// held back until enough bytes arrive to disambiguate. No nesting, no
// attributes — if the prompt format needs more, write a different schema.
type XMLTagSchema struct {
	// Tags is the set of recognized tag names. A tag not in this list is
	// treated as text (and either dropped or emitted via OutsideKind).
	Tags []string

	// OutsideKind is the Kind emitted for text outside any recognized tag.
	// Empty = drop outside text silently.
	OutsideKind string

	buf          strings.Builder // unprocessed bytes
	openTag      string          // "" when outside any recognized tag
	openBlockTxt strings.Builder // accumulated content of the open tag
	outsideTxt   strings.Builder // accumulated outside text (only when OutsideKind != "")
}

// Feed consumes a delta and emits any blocks completed or updated by it.
func (s *XMLTagSchema) Feed(delta string) []Block {
	if delta == "" {
		return nil
	}
	s.buf.WriteString(delta)
	return s.pump()
}

// Close flushes any in-progress block. Any open tag is emitted Done=true.
// Any accumulated outside text (when OutsideKind != "") is flushed Done=true.
func (s *XMLTagSchema) Close() []Block {
	out := s.pump()
	remaining := s.buf.String()
	s.buf.Reset()

	if s.openTag != "" {
		if remaining != "" {
			s.openBlockTxt.WriteString(remaining)
			out = append(out, Block{Kind: s.openTag, Delta: remaining, Text: s.openBlockTxt.String()})
		}
		out = append(out, Block{Kind: s.openTag, Text: s.openBlockTxt.String(), Done: true})
		s.openTag = ""
		s.openBlockTxt.Reset()
		return out
	}

	if s.OutsideKind != "" {
		if remaining != "" {
			s.outsideTxt.WriteString(remaining)
			out = append(out, Block{Kind: s.OutsideKind, Delta: remaining, Text: s.outsideTxt.String()})
		}
		if s.outsideTxt.Len() > 0 {
			out = append(out, Block{Kind: s.OutsideKind, Text: s.outsideTxt.String(), Done: true})
			s.outsideTxt.Reset()
		}
	}
	return out
}

// pump consumes s.buf until nothing more can be safely processed, leaving
// only bytes that could still be part of a partial marker. After each step
// the buffer is advanced by `consumed` regardless of `wait` — the steps
// emit only the bytes they actually consumed, so the same bytes must never
// be re-processed on the next Feed.
func (s *XMLTagSchema) pump() []Block {
	var out []Block
	for {
		str := s.buf.String()
		if str == "" {
			return out
		}

		var (
			emitted  []Block
			consumed int
			wait     bool
		)
		if s.openTag == "" {
			emitted, consumed, wait = s.stepOutside(str)
		} else {
			emitted, consumed, wait = s.stepInside(str)
		}
		out = append(out, emitted...)

		if consumed > 0 {
			s.buf.Reset()
			s.buf.WriteString(str[consumed:])
		}
		if wait {
			return out
		}
	}
}

// stepOutside handles one iteration while outside any recognized tag.
func (s *XMLTagSchema) stepOutside(str string) (blocks []Block, consumed int, wait bool) {
	idx := strings.IndexByte(str, '<')
	if idx == -1 {
		if s.OutsideKind != "" {
			s.outsideTxt.WriteString(str)
			blocks = append(blocks, Block{Kind: s.OutsideKind, Delta: str, Text: s.outsideTxt.String()})
		}
		return blocks, len(str), true
	}
	if idx > 0 {
		prefix := str[:idx]
		if s.OutsideKind != "" {
			s.outsideTxt.WriteString(prefix)
			blocks = append(blocks, Block{Kind: s.OutsideKind, Delta: prefix, Text: s.outsideTxt.String()})
		}
		return blocks, idx, false
	}

	// str starts with '<'. Try to match a known opener.
	for _, tag := range s.Tags {
		opener := "<" + tag + ">"
		if strings.HasPrefix(str, opener) {
			if s.OutsideKind != "" && s.outsideTxt.Len() > 0 {
				blocks = append(blocks, Block{Kind: s.OutsideKind, Text: s.outsideTxt.String(), Done: true})
				s.outsideTxt.Reset()
			}
			s.openTag = tag
			s.openBlockTxt.Reset()
			return blocks, len(opener), false
		}
	}

	// Could it still become a known tag with more data?
	for _, tag := range s.Tags {
		opener := "<" + tag + ">"
		if len(str) < len(opener) && strings.HasPrefix(opener, str) {
			return blocks, 0, true
		}
	}

	// '<' but no tag will match — treat as outside text and advance one byte.
	if s.OutsideKind != "" {
		s.outsideTxt.WriteString("<")
		blocks = append(blocks, Block{Kind: s.OutsideKind, Delta: "<", Text: s.outsideTxt.String()})
	}
	return blocks, 1, false
}

// stepInside handles one iteration while inside a recognized tag, watching
// for the matching closer.
func (s *XMLTagSchema) stepInside(str string) (blocks []Block, consumed int, wait bool) {
	closer := "</" + s.openTag + ">"
	if idx := strings.Index(str, closer); idx >= 0 {
		if idx > 0 {
			safe := str[:idx]
			s.openBlockTxt.WriteString(safe)
			blocks = append(blocks, Block{Kind: s.openTag, Delta: safe, Text: s.openBlockTxt.String()})
		}
		blocks = append(blocks, Block{Kind: s.openTag, Text: s.openBlockTxt.String(), Done: true})
		s.openTag = ""
		s.openBlockTxt.Reset()
		return blocks, idx + len(closer), false
	}

	// No full closer yet. Emit every byte except a trailing partial closer.
	safeN := len(str)
	for n := 1; n < len(closer) && n <= len(str); n++ {
		if strings.HasPrefix(closer, str[len(str)-n:]) && len(str)-n < safeN {
			safeN = len(str) - n
		}
	}
	if safeN > 0 {
		safe := str[:safeN]
		s.openBlockTxt.WriteString(safe)
		blocks = append(blocks, Block{Kind: s.openTag, Delta: safe, Text: s.openBlockTxt.String()})
	}
	return blocks, safeN, true
}

// MarkdownHeaderSchema parses sectioned markdown-like output:
//
//	SECTION_A_HEADER
//	content for A
//	SECTION_B_HEADER
//	content for B
//
// Headers maps each exact header line (e.g. "## Reasoning") to the Kind
// emitted for that section. A section starts at its header and ends at the
// next recognized header or at stream end. Text before the first header is
// emitted with Kind = PreambleKind — or dropped when PreambleKind is "".
//
// Delta-safe: a header split across deltas is held back until its newline
// arrives, so partial matches never leak into a content section.
type MarkdownHeaderSchema struct {
	Headers      map[string]string
	PreambleKind string // empty = drop preamble

	buf         strings.Builder
	currentKind string // "" before the first recognized header (preamble)
	currentTxt  strings.Builder
}

// Feed consumes a delta and emits blocks completed or updated by it.
func (s *MarkdownHeaderSchema) Feed(delta string) []Block {
	if delta == "" {
		return nil
	}
	s.buf.WriteString(delta)
	return s.pump(false)
}

// Close flushes any in-progress section with Done=true.
func (s *MarkdownHeaderSchema) Close() []Block {
	return s.pump(true)
}

func (s *MarkdownHeaderSchema) pump(flush bool) []Block {
	var out []Block
	for {
		str := s.buf.String()
		if str == "" && !flush {
			return out
		}

		nlIdx := strings.IndexByte(str, '\n')
		if nlIdx < 0 {
			if !flush {
				return out
			}
			out = append(out, s.handleLine(str, true)...)
			s.buf.Reset()
			return out
		}

		line := str[:nlIdx]
		out = append(out, s.handleLine(line, false)...)
		s.buf.Reset()
		s.buf.WriteString(str[nlIdx+1:])
	}
}

// handleLine processes one complete line (or the final tail during Close).
func (s *MarkdownHeaderSchema) handleLine(line string, finalFlush bool) []Block {
	var out []Block

	if kind, ok := s.Headers[strings.TrimRight(line, "\r")]; ok {
		// Close the current section (if any) with Done=true.
		if s.currentTxt.Len() > 0 && (s.currentKind != "" || s.PreambleKind != "") {
			k := s.currentKind
			if k == "" {
				k = s.PreambleKind
			}
			out = append(out, Block{Kind: k, Text: strings.TrimRight(s.currentTxt.String(), "\n"), Done: true})
		}
		s.currentKind = kind
		s.currentTxt.Reset()
		return out
	}

	// Regular content line. Drop if we're in preamble and PreambleKind is "".
	kind := s.currentKind
	if kind == "" {
		if s.PreambleKind == "" {
			return out
		}
		kind = s.PreambleKind
	}
	prefix := line
	if !finalFlush {
		prefix = line + "\n"
	}
	s.currentTxt.WriteString(prefix)
	out = append(out, Block{Kind: kind, Delta: prefix, Text: s.currentTxt.String()})

	if finalFlush {
		out = append(out, Block{Kind: kind, Text: s.currentTxt.String(), Done: true})
		s.currentKind = ""
		s.currentTxt.Reset()
	}
	return out
}
