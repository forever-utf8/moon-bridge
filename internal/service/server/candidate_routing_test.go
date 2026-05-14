package server

import (
	"encoding/json"
	"testing"

	deepseekv4 "moonbridge/internal/extension/deepseek_v4"
	"moonbridge/internal/format"
	"moonbridge/internal/protocol/anthropic"
	"moonbridge/internal/service/provider"
	"moonbridge/internal/service/stats"
	"moonbridge/internal/session"
)

func TestRequestHasImage(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  bool
	}{
		{"empty input", "", false},
		{"null input", "null", false},
		{"string input", `"hello"`, false},
		{"array without image", `[{"type":"input_text","text":"hello"}]`, false},
		{"array with input_image", `[{"type":"input_text","text":"hello"},{"type":"input_image","image_url":"data:image/png;base64,abc"}]`, true},
		{"array with image", `[{"type":"text","text":"hello"},{"type":"image","image_url":"data:image/png;base64,abc"}]`, true},
		{"array with image_url", `[{"type":"image_url","image_url":"https://example.com/img.png"}]`, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := requestHasImage(json.RawMessage(tt.input))
			if got != tt.want {
				t.Errorf("requestHasImage(%q) = %v, want %v", tt.input, got, tt.want)
			}
		})
	}
}

func TestHasModalityImage(t *testing.T) {
	tests := []struct {
		name       string
		modalities []string
		want       bool
	}{
		{"nil list", nil, false},
		{"empty list", []string{}, false},
		{"only text", []string{"text"}, false},
		{"with image", []string{"text", "image"}, true},
		{"only image", []string{"image"}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := hasModalityImage(tt.modalities)
			if got != tt.want {
				t.Errorf("hasModalityImage(%v) = %v, want %v", tt.modalities, got, tt.want)
			}
		})
	}
}

func TestFilterCandidatesByInputNoProviderMgr(t *testing.T) {
	srv := &Server{providerMgr: nil}
	candidates := []provider.ProviderCandidate{
		{ProviderKey: "p1", UpstreamModel: "model-a"},
	}
	filtered, _ := srv.filterCandidatesByInput(candidates, json.RawMessage(`[{"type":"input_image","image_url":"data:image/png;base64,abc"}]`))
	if len(filtered) != 1 {
		t.Fatalf("without providerMgr, should return unchanged: got %d", len(filtered))
	}
}

func TestComputeCostWithProviderPricingNilStats(t *testing.T) {
	cost := computeCostWithProviderPricing(nil, nil, "model", "model", "provider", stats.BillingUsage{})
	if cost != 0 {
		t.Fatalf("nil stats should return 0, got %f", cost)
	}
}

func TestPrependCachedThinkingSkipsAssistantTextAndFallsBackForToolUse(t *testing.T) {
	sess := session.New()
	state := deepseekv4.NewState()
	sess.InitExtensions(map[string]any{
		"deepseek_v4": state,
	})

	req := &anthropic.MessageRequest{
		Messages: []anthropic.Message{
			{
				Role: "assistant",
				Content: []anthropic.ContentBlock{
					{Type: "text", Text: "plain assistant text"},
				},
			},
			{
				Role: "assistant",
				Content: []anthropic.ContentBlock{
					{Type: "tool_use", ID: "call-miss", Name: "exec_command", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}

	prependCachedThinking(req, sess)

	if len(req.Messages[0].Content) != 1 || req.Messages[0].Content[0].Type != "text" {
		t.Fatalf("assistant text message should remain unchanged, got %+v", req.Messages[0].Content)
	}

	if len(req.Messages[1].Content) < 2 {
		t.Fatalf("tool_use message should receive fallback thinking block, got %+v", req.Messages[1].Content)
	}
	if req.Messages[1].Content[0].Type != "thinking" {
		t.Fatalf("first block should be thinking fallback, got %+v", req.Messages[1].Content[0])
	}
	if req.Messages[1].Content[1].Type != "tool_use" || req.Messages[1].Content[1].ID != "call-miss" {
		t.Fatalf("tool_use block misplaced after fallback prepend, got %+v", req.Messages[1].Content)
	}
}

func TestPrependCachedThinkingChecksAllToolUseBlocks(t *testing.T) {
	sess := session.New()
	state := deepseekv4.NewState()
	state.RememberForToolCalls(
		[]string{"call-hit"},
		format.CoreContentBlock{
			Type:               "reasoning",
			ReasoningText:      "replayed",
			ReasoningSignature: "sig-hit",
		},
	)
	sess.InitExtensions(map[string]any{
		"deepseek_v4": state,
	})

	req := &anthropic.MessageRequest{
		Messages: []anthropic.Message{
			{
				Role: "assistant",
				Content: []anthropic.ContentBlock{
					{Type: "tool_use", ID: "call-miss", Name: "exec_command", Input: json.RawMessage(`{}`)},
					{Type: "tool_use", ID: "call-hit", Name: "exec_command", Input: json.RawMessage(`{}`)},
				},
			},
		},
	}

	prependCachedThinking(req, sess)

	if len(req.Messages[0].Content) < 3 {
		t.Fatalf("assistant message should prepend cached thinking, got %+v", req.Messages[0].Content)
	}
	head := req.Messages[0].Content[0]
	if head.Type != "thinking" || head.Thinking != "replayed" || head.Signature != "sig-hit" {
		t.Fatalf("cached thinking block mismatch, got %+v", head)
	}
}
