"""Ollama generation protocol for optional visual attribute extraction."""

from __future__ import annotations

from typing import Any

from boxmot.attributes.backends.http import HttpAttributeBackend


class OllamaBackend(HttpAttributeBackend):
    """Send a single labeled storyboard to Ollama's generation endpoint."""

    name = "Ollama"

    def request_payload(self, prompt: str, image_base64: str) -> dict[str, Any]:
        """Request schema-constrained output with Ollama's native image field."""
        payload: dict[str, Any] = {
            "model": self._config.model,
            "prompt": prompt,
            "images": [image_base64],
            "format": self._config.schema,
            "stream": False,
            "options": {"temperature": 0},
        }
        if self._config.think is not None:
            payload["think"] = self._config.think
        return payload

    def response_text(self, envelope: Any) -> str | None:
        """Read final output, ignoring any separate reasoning content."""
        text = envelope.get("response") if isinstance(envelope, dict) else None
        return text if isinstance(text, str) else None

    def request_url(self) -> str:
        """Return the native Ollama generation URL."""
        assert self._config.host is not None
        return f"{self._config.host.rstrip('/')}/api/generate"
