"""Anthropic Messages API requests, authentication, and response validation."""

import asyncio
import base64
import json

import cv2
import numpy as np
import pytest

import boxmot.attributes.backends.http as attribute_http
from boxmot.attributes.backends.anthropic import AnthropicBackend
from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.errors import AttributeAnalysisError, TransientAttributeError
from boxmot.attributes.models import AttributeSample
from boxmot.utils.config import ConfigurationError

httpx = pytest.importorskip("httpx")
pytest.importorskip("jsonschema")
pytest.importorskip("referencing")


def _config(**overrides) -> AttributeConfig:
    """Build a schema supported by Claude's structured output API."""
    values = {
        "backend": "anthropic",
        "model": "claude-sonnet-4-6",
        "prompt": "Describe color.",
        "schema": {
            "type": "object",
            "properties": {"color": {"type": "string"}},
            "required": ["color"],
            "additionalProperties": False,
        },
    }
    values.update(overrides)
    return AttributeConfig(**values)


def _response(text='{"color":"red"}', **overrides) -> dict:
    """Wrap model output in a completed assistant message."""
    return {
        "type": "message",
        "role": "assistant",
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        **overrides,
    }


def _transport(monkeypatch, handler) -> None:
    """Capture requests without connecting to a live provider."""
    client_type = httpx.AsyncClient
    monkeypatch.setenv("ANTHROPIC_API_KEY", "private-test-token")
    monkeypatch.setattr(attribute_http, "require_extra", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs)
    )


def _extract(backend: AnthropicBackend) -> dict:
    """Exercise the real storyboard, HTTP, and local validation pipeline."""
    samples = [AttributeSample("sample-0", 0, 0.0), AttributeSample("sample-30", 30, 1.0)]
    images = []
    for color in ((0, 0, 255), (255, 0, 0)):
        success, encoded = cv2.imencode(".jpg", np.full((24, 16, 3), color, dtype=np.uint8))
        assert success
        images.append(encoded.tobytes())

    async def run() -> dict:
        try:
            return await backend.extract(7, samples, images)
        finally:
            await backend.aclose()

    return asyncio.run(run())


@pytest.mark.parametrize("think", [False, None])
def test_messages_request_with_schema_storyboard_and_native_auth(monkeypatch, think) -> None:
    """Use the native Messages wire format, including a required output budget."""
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_response())

    _transport(monkeypatch, handler)
    config = _config(think=think, max_output_tokens=4096)
    backend = AnthropicBackend(config)
    assert _extract(backend) == {"color": "red"}
    request = requests[0]
    assert str(request.url) == "https://api.anthropic.com/v1/messages"
    assert request.headers["x-api-key"] == "private-test-token"
    assert request.headers["anthropic-version"] == "2026-10-21"
    assert request.headers["content-type"] == "application/json"
    assert "authorization" not in request.headers
    assert "anthropic-beta" not in request.headers
    assert "private-test-token" not in request.content.decode()
    payload = json.loads(request.content)
    assert payload["model"] == config.model
    assert payload["max_tokens"] == 4096
    assert payload["stream"] is False
    assert payload["output_config"] == {"format": {"type": "json_schema", "schema": config.schema}}
    assert not {"temperature", "top_p", "top_k", "thinking", "response_format", "chat_template_kwargs"} & payload.keys()
    assert len(payload["messages"]) == 1
    message = payload["messages"][0]
    assert message["role"] == "user"
    image, text = message["content"]
    assert text["type"] == "text" and config.prompt in text["text"]
    assert json.dumps(config.schema) in text["text"]
    assert image["type"] == "image"
    source = image["source"]
    assert source["type"] == "base64" and source["media_type"] == "image/jpeg"
    storyboard = cv2.imdecode(np.frombuffer(base64.b64decode(source["data"]), np.uint8), cv2.IMREAD_COLOR)
    assert storyboard.shape == (480, 640, 3)
    assert storyboard[240, 160, 2] > 240
    assert storyboard[240, 480, 0] > 240
    assert backend._client.is_closed


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("https://api.anthropic.com/", "https://api.anthropic.com/v1/messages"),
        ("https://api.anthropic.com/v1/", "https://api.anthropic.com/v1/messages"),
        ("https://example.com/vision/", "https://example.com/vision/v1/messages"),
        ("https://example.com/vision/v1", "https://example.com/vision/v1/messages"),
    ],
)
def test_custom_api_base_and_key_environment(monkeypatch, host: str, expected: str) -> None:
    """Gateways can select another base URL and API-key environment variable."""
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_response())

    _transport(monkeypatch, handler)
    monkeypatch.setenv("ATTRIBUTE_TEST_TOKEN", "custom-private-token")
    assert _extract(AnthropicBackend(_config(host=host, api_key_env="ATTRIBUTE_TEST_TOKEN"))) == {"color": "red"}
    assert str(requests[0].url) == expected
    assert requests[0].headers["x-api-key"] == "custom-private-token"
    assert "authorization" not in requests[0].headers


@pytest.mark.parametrize("token", [None, "", "   "])
def test_missing_api_key_fails_before_client_creation(monkeypatch, token: str | None) -> None:
    """A hosted backend cannot silently issue unauthenticated requests."""
    monkeypatch.setattr(attribute_http, "require_extra", lambda *args, **kwargs: None)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("Client must not be created."))
    if token is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", token)
    with pytest.raises(ConfigurationError, match="ANTHROPIC_API_KEY.*nonempty"):
        AnthropicBackend(_config())


def test_text_blocks_are_joined_without_returning_thinking(monkeypatch) -> None:
    """Thinking blocks never become attribute values or interfere with final JSON."""
    envelope = _response(content=[
        {"type": "thinking", "thinking": "not JSON", "signature": "opaque"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text", "text": '{"color":'},
        {"type": "text", "text": '"red"}'},
    ])
    _transport(monkeypatch, lambda request: httpx.Response(200, json=envelope))
    assert _extract(AnthropicBackend(_config())) == {"color": "red"}


@pytest.mark.parametrize(
    "envelope",
    [
        None,
        {},
        _response(type="error"),
        _response(role="user"),
        _response(content=None),
        _response(content=[]),
        _response(content=[None]),
        _response(content=[{"type": "thinking", "thinking": '{"color":"red"}'}]),
        _response(None),
        _response(""),
        _response({"color": "red"}),
        _response("```json\n{}\n```"),
        _response('{"color":3}'),
        _response('{"color":"red","track_id":99}'),
        _response('{"color":NaN}'),
        _response(content=[{"type": "text", "text": '{"color":"red"}'}, {"type": "refusal"}]),
        _response(content=[{"type": "tool_use", "name": "anything", "input": {"color": "red"}}]),
    ],
)
def test_invalid_response_never_publishes_attributes(monkeypatch, envelope) -> None:
    """Malformed, refusal, tool, or schema-invalid content cannot replace results."""
    _transport(monkeypatch, lambda request: httpx.Response(200, json=envelope))
    with pytest.raises(AttributeAnalysisError):
        _extract(AnthropicBackend(_config()))


@pytest.mark.parametrize(
    "stop_reason",
    [None, "max_tokens", "refusal", "tool_use", "pause_turn", "stop_sequence", "model_context_window_exceeded"],
)
def test_abnormal_stop_rejects_even_valid_json(monkeypatch, stop_reason: str | None) -> None:
    """Only end_turn permits publishing a structured result."""
    _transport(monkeypatch, lambda request: httpx.Response(200, json=_response(stop_reason=stop_reason)))
    with pytest.raises(AttributeAnalysisError, match="did not finish.*normally"):
        _extract(AnthropicBackend(_config()))


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500, 529])
def test_http_errors_are_classified_without_persisting_provider_body(monkeypatch, status: int) -> None:
    """Overload and rate limits retry, and body details stay out of saved errors."""
    _transport(monkeypatch, lambda request: httpx.Response(
        status, headers={"Retry-After": "3"}, json={"error": {"message": "private-test-token"}}
    ))
    with pytest.raises(AttributeAnalysisError) as error:
        _extract(AnthropicBackend(_config()))
    assert str(error.value) == f"Anthropic returned HTTP {status}."
    transient = status == 429 or status >= 500
    assert isinstance(error.value, TransientAttributeError) is transient
    if transient:
        assert error.value.retry_after_s == 3


def test_transport_errors_do_not_expose_credentials(monkeypatch) -> None:
    """Transport diagnostics may contain headers and must remain sanitized."""
    def handler(request):
        raise httpx.ConnectError("x-api-key: private-test-token", request=request)

    _transport(monkeypatch, handler)
    with pytest.raises(TransientAttributeError, match=r"^Anthropic request failed \(ConnectError\)\.$"):
        _extract(AnthropicBackend(_config()))
