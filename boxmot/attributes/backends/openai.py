"""Visual track attributes through OpenAI's native Responses API."""

from __future__ import annotations

from typing import Any

from boxmot.attributes.backends.http import HttpAttributeBackend
from boxmot.attributes.errors import AttributeAnalysisError


class OpenAIBackend(HttpAttributeBackend):
    """Send a storyboard and authored JSON Schema to an OpenAI vision model."""

    name = "OpenAI"

    def request_url(self) -> str:
        """Accept an API root or /v1 base, including a reverse proxy prefix."""
        assert self._config.host is not None
        base = self._config.host.rstrip("/")
        return f"{base}/responses" if base.endswith("/v1") else f"{base}/v1/responses"

    def request_payload(self, prompt: str, image_base64: str) -> dict[str, Any]:
        """Request a single JSON result without storing it at the provider."""
        output_format: dict[str, Any] = {"type": self._config.response_format}
        if self._config.response_format == "json_schema":
            output_format.update(name="track_attributes", strict=True, schema=self._config.schema)
        return {
            "model": self._config.model,
            "input": [{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{image_base64}",
                        "detail": "auto",
                    },
                ],
            }],
            "stream": False,
            "store": False,
            "max_output_tokens": self._config.max_output_tokens,
            "text": {"format": output_format},
        }

    def response_text(self, envelope: Any) -> str | None:
        """Read completed assistant text while rejecting refusals and truncation."""
        if not isinstance(envelope, dict):
            return None
        if (
            envelope.get("status") != "completed"
            or envelope.get("error") is not None
            or envelope.get("incomplete_details") is not None
        ):
            raise AttributeAnalysisError("OpenAI did not finish the attribute response normally.")
        output = envelope.get("output")
        if not isinstance(output, list) or not output:
            return None
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                return None
            if item.get("type") == "reasoning":
                continue
            if item.get("type") != "message":
                raise AttributeAnalysisError("OpenAI returned a non-message output instead of attributes.")
            if item.get("status") != "completed" or item.get("role") != "assistant":
                raise AttributeAnalysisError("OpenAI did not finish the assistant message normally.")
            content = item.get("content")
            if not isinstance(content, list) or not content:
                return None
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "output_text":
                    raise AttributeAnalysisError("OpenAI returned a refusal or non-text content instead of attributes.")
                text = part.get("text")
                if not isinstance(text, str):
                    return None
                texts.append(text)
        return "".join(texts) if texts else None
