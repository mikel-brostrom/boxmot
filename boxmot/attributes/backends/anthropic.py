"""Visual track attributes through Anthropic's Messages API."""

from __future__ import annotations

from typing import Any

from boxmot.attributes.backends.http import HttpAttributeBackend
from boxmot.attributes.errors import AttributeAnalysisError


class AnthropicBackend(HttpAttributeBackend):
    """Send a JPEG storyboard and JSON Schema to a Claude vision model."""

    name = "Anthropic"

    def request_headers(self, api_key: str | None) -> dict[str, str]:
        """Use the Messages API version and its native API-key header."""
        headers = {"anthropic-version": "2026-10-21"}
        if api_key is not None:
            headers["x-api-key"] = api_key
        return headers

    def request_url(self) -> str:
        """Accept an Anthropic-compatible server root or /v1 API base URL."""
        assert self._config.host is not None
        base = self._config.host.rstrip("/")
        return f"{base}/messages" if base.endswith("/v1") else f"{base}/v1/messages"

    def request_payload(self, prompt: str, image_base64: str) -> dict[str, Any]:
        """Request a complete, schema-constrained response to one storyboard."""
        return {
            "model": self._config.model,
            "max_tokens": self._config.max_output_tokens,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": image_base64},
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
            "output_config": {"format": {"type": "json_schema", "schema": self._config.schema}},
        }

    def response_text(self, envelope: Any) -> str | None:
        """Read final assistant text while rejecting refusals, truncation, and tools."""
        if not isinstance(envelope, dict) or envelope.get("type") != "message" or envelope.get("role") != "assistant":
            return None
        if envelope.get("stop_reason") != "end_turn":
            raise AttributeAnalysisError("Anthropic did not finish the attribute response normally.")
        content = envelope.get("content")
        if not isinstance(content, list):
            return None
        parts = []
        for block in content:
            if not isinstance(block, dict):
                return None
            if block.get("type") in ("thinking", "redacted_thinking"):
                continue
            if block.get("type") != "text":
                raise AttributeAnalysisError("Anthropic returned non-text content instead of attributes.")
            text = block.get("text")
            if not isinstance(text, str):
                return None
            parts.append(text)
        return "".join(parts)
