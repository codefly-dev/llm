package llm

// ARCHITECTURE: channel-based block streaming.
//
// Raw streaming (Client.Stream + onChunk callback) is fine for "dump chunks to
// stdout" but useless for UIs that render different sections differently
// (reasoning hidden, code formatted, answer in a panel). StreamBlocks takes
// that raw stream and a BlockSchema, then emits typed [Block] events on a
// channel in arrival order. The consumer decides what to do with each kind
// — show, hide, syntax-highlight, buffer for post-processing.
//
// Channels were chosen over callbacks because:
//   - Consumer runs at its own pace (a UI at 60fps doesn't block the LLM stream)
//   - Multiple goroutines can fan out from one stream (UI renderer + audit logger)
//   - ctx cancellation propagates naturally (close channel → consumers exit range)
//   - Buffered channels give backpressure for free

import (
	"context"
	"fmt"
)

// StreamResult reports the final outcome of a StreamBlocks run.
//
// Returned by [StreamBlocks] on a synchronous channel so the caller can
// observe the full raw response + any error without racing the Block events.
type StreamResult struct {
	// Full is the complete raw response assembled from every delta. Identical
	// to what Client.Stream returns. Use this for logging/cassettes.
	Full string
	// Err is the first error from Client.Stream or from a channel send that
	// failed due to ctx cancellation. nil on clean completion.
	Err error
}

// StreamBlocks drives Client.Stream through a [BlockSchema], emitting parsed
// [Block] events on the returned channel in arrival order. The channel is
// closed when the stream ends (either cleanly or on error).
//
// The returned `result` channel receives exactly one [StreamResult] after
// streaming completes — it signals "end of stream" with the full text and
// any error. Callers typically range over `blocks` and then read from
// `result` once.
//
// If ctx is cancelled, the underlying stream is cancelled and both channels
// are closed; the StreamResult carries ctx.Err().
//
// bufSize controls the block channel buffer. 0 selects a sensible default
// (16). A larger buffer lets the LLM stream run ahead of the consumer; a
// smaller one applies backpressure.
func StreamBlocks(
	ctx context.Context,
	c Client,
	prompt string,
	schema BlockSchema,
	bufSize int,
) (<-chan Block, <-chan StreamResult) {
	if bufSize <= 0 {
		bufSize = 16
	}
	blocks := make(chan Block, bufSize)
	result := make(chan StreamResult, 1)

	go func() {
		defer close(blocks)
		defer close(result)

		if schema == nil {
			result <- StreamResult{Err: fmt.Errorf("llm: nil BlockSchema")}
			return
		}
		if c == nil {
			result <- StreamResult{Err: fmt.Errorf("llm: nil Client")}
			return
		}

		sendErr := func(send []Block) error {
			for _, b := range send {
				select {
				case <-ctx.Done():
					return ctx.Err()
				case blocks <- b:
				}
			}
			return nil
		}

		var sendFailed error
		onChunk := func(delta string) error {
			if sendFailed != nil {
				return sendFailed
			}
			emitted := schema.Feed(delta)
			if err := sendErr(emitted); err != nil {
				sendFailed = err
				return err
			}
			return nil
		}

		full, err := c.Stream(ctx, prompt, onChunk)
		// Flush the schema even on error — any partial block is surfaced.
		if flushed := schema.Close(); len(flushed) > 0 && sendFailed == nil {
			_ = sendErr(flushed)
		}

		finalErr := err
		if finalErr == nil {
			finalErr = sendFailed
		}
		result <- StreamResult{Full: full, Err: finalErr}
	}()

	return blocks, result
}

// CollectBlocks drains a Block channel into a slice. Convenience for tests
// and non-streaming callers that want the final parsed structure.
func CollectBlocks(blocks <-chan Block) []Block {
	var out []Block
	for b := range blocks {
		out = append(out, b)
	}
	return out
}
