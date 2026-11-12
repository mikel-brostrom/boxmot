"""Visual track attributes through vLLM's OpenAI-compatible HTTP server."""

from __future__ import annotations

from typing import Any

from boxmot.attributes.backends.http import HttpAttributeBackend
from boxmot.attributes.errors import AttributeAnalysisError


class VLLMBackend(HttpAttributeBackend):
    """Send a storyboard and JSON Schema to a separately hosted vision model."""

    name = "vLLM"

    def request_url(self) -> str:
        """Accept a server root or the conventional /v1 API base URL."""
        assert self._config.host is not None
        base = self._config.host.rstrip("/")
        return f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"

    def request_payload(self, prompt: str, image_base64: str) -> dict[str, Any]:
        """Use vision chat with schema-constrained output or provider JSON mode."""
        messages: list[dict[str, Any]] = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            ],
        }]
        response_format: dict[str, Any] = {"type": self._config.response_format}
        if self._config.response_format == "json_schema":
            response_format["json_schema"] = {"name": "track_attributes", "schema": self._config.schema}
        else:
            messages.insert(0, {
                "role": "system",
                "content": "Return only a valid JSON object matching the schema in the user message.",
            })
        payload: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "response_format": response_format,
        }
        if self._config.think is not None:
            payload["chat_template_kwargs"] = {"enable_thinking": self._config.think}
        return payload

    def response_text(self, envelope: Any) -> str | None:
        """Read only final assistant content; reject truncation or refusals."""
        choices = envelope.get("choices") if isinstance(envelope, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return None
        choice = choices[0]
        if choice.get("finish_reason") not in (None, "stop"):
            raise AttributeAnalysisError("vLLM did not finish the attribute response normally.")
        message = choice.get("message")
        if not isinstance(message, dict):
            return None
        if message.get("refusal") or message.get("tool_calls"):
            raise AttributeAnalysisError("vLLM returned a refusal or tool call instead of attributes.")
        return message.get("content")
