"""OpenAI Responses vision requests and terminal response validation."""

import asyncio
import base64
import json
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np
import pytest

import boxmot.attributes.backends.http as attribute_http
from boxmot.attributes.backends.openai import OpenAIBackend
from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.errors import AttributeAnalysisError, TransientAttributeError
from boxmot.attributes.models import AttributeSample
from boxmot.utils.config import ConfigurationError

httpx = pytest.importorskip("httpx")
pytest.importorskip("jsonschema")
pytest.importorskip("referencing")


def _config(**overrides: Any) -> AttributeConfig:
    """Create a small schema compatible with provider structured output."""
    values = {
        "backend": "openai",
        "model": "test-vision-model",
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


def _response(text: Any = '{"color":"red"}', **overrides: Any) -> dict[str, Any]:
    """Build the raw HTTP envelope, without the SDK's output_text helper."""
    return {
        "status": "completed",
        "output": [{
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text}],
        }],
        **overrides,
    }


def _transport(monkeypatch: pytest.MonkeyPatch, handler: Callable[..., Any]) -> None:
    """Use a real asynchronous HTTP client with an in-memory provider."""
    client_type = httpx.AsyncClient
    monkeypatch.setenv("OPENAI_API_KEY", "private-test-token")
    monkeypatch.setattr(attribute_http, "require_extra", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs)
    )


def _extract(backend: OpenAIBackend) -> dict[str, Any]:
    """Submit two distinguishable frames and close the transport on every path."""
    samples = [AttributeSample("sample-0", 0, 0.0), AttributeSample("sample-30", 30, 1.0)]
    images = []
    for color in ((0, 0, 255), (255, 0, 0)):
        success, encoded = cv2.imencode(".jpg", np.full((24, 16, 3), color, dtype=np.uint8))
        assert success
        images.append(encoded.tobytes())

    async def run() -> dict[str, Any]:
        try:
            return await backend.extract(7, samples, images)
        finally:
            await backend.aclose()

    return asyncio.run(run())


@pytest.mark.parametrize("response_format", ["json_schema", "json_object"])
def test_native_responses_request_and_storyboard(monkeypatch: pytest.MonkeyPatch, response_format: str) -> None:
    """Use Responses multimodal input, local schema validation, and bearer auth."""
    requests = []

    def handler(request: Any) -> Any:
        requests.append(request)
        envelope = _response()
        envelope["output"].insert(0, {"type": "reasoning", "summary": [{"text": "Not JSON"}]})
        return httpx.Response(200, json=envelope)

    _transport(monkeypatch, handler)
    config = _config(response_format=response_format, max_output_tokens=8192)
    backend = OpenAIBackend(config)
    assert _extract(backend) == {"color": "red"}
    request = requests[0]
    assert str(request.url) == "https://api.openai.com/v1/responses"
    assert request.headers["Authorization"] == "Bearer private-test-token"
    assert "private-test-token" not in request.content.decode()
    payload = json.loads(request.content)
    assert payload["model"] == config.model
    assert payload["stream"] is False and payload["store"] is False
    assert payload["max_output_tokens"] == 8192
    output_format = {"type": response_format}
    if response_format == "json_schema":
        output_format.update(name="track_attributes", strict=True, schema=config.schema)
    assert payload["text"] == {"format": output_format}
    assert set(payload) == {"model", "input", "stream", "store", "max_output_tokens", "text"}
    assert len(payload["input"]) == 1
    message = payload["input"][0]
    assert message["role"] == "user"
    text, image = message["content"]
    assert text["type"] == "input_text" and config.prompt in text["text"]
    assert json.dumps(config.schema) in text["text"]
    assert image["type"] == "input_image" and image["detail"] == "auto"
    prefix, encoded = image["image_url"].split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    storyboard = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    assert storyboard.shape == (480, 640, 3)
    assert storyboard[240, 160, 2] > 240
    assert storyboard[240, 480, 0] > 240
    assert backend._client.is_closed


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("https://example.com/", "https://example.com/v1/responses"),
        ("https://example.com/v1/", "https://example.com/v1/responses"),
        ("https://example.com/proxy/", "https://example.com/proxy/v1/responses"),
        ("https://example.com/proxy/v1/", "https://example.com/proxy/v1/responses"),
    ],
)
def test_api_base_and_custom_key(monkeypatch: pytest.MonkeyPatch, host: str, expected: str) -> None:
    """Provider proxies retain their path prefix and configured key variable."""
    requests = []

    def handler(request: Any) -> Any:
        requests.append(request)
        return httpx.Response(200, json=_response())

    _transport(monkeypatch, handler)
    monkeypatch.setenv("ATTRIBUTE_TEST_KEY", "custom-test-key")
    assert _extract(OpenAIBackend(_config(host=host, api_key_env="ATTRIBUTE_TEST_KEY"))) == {"color": "red"}
    assert str(requests[0].url) == expected
    assert requests[0].headers["Authorization"] == "Bearer custom-test-key"


def test_missing_key_fails_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A native hosted backend requires the default API credential."""
    _transport(monkeypatch, lambda request: pytest.fail("No request should be sent"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        OpenAIBackend(_config())


@pytest.mark.parametrize(
    "envelope",
    [
        None,
        {},
        _response(output=[]),
        _response(output=[None]),
        _response(output=[{"type": "reasoning", "summary": []}]),
        _response(output=[{"type": "function_call", "arguments": '{"color":"red"}'}]),
        _response(output=[{"type": "message", "role": "user", "status": "completed", "content": []}]),
        _response(output=[{"type": "message", "role": "assistant", "status": "in_progress", "content": []}]),
        _response(output=[{"type": "message", "role": "assistant", "status": "completed", "content": None}]),
        _response(output=[{
            "type": "message", "role": "assistant", "status": "completed",
            "content": [{"type": "refusal", "refusal": "private-provider-text"}],
        }]),
        _response(status="incomplete", incomplete_details={"reason": "max_output_tokens"}),
        _response(status="failed", error={"message": "private-provider-text"}),
        _response(status="queued"),
        _response(status="cancelled"),
        _response(error={"message": "private-provider-text"}),
        _response(incomplete_details={"reason": "content_filter"}),
        _response(None),
        _response({"color": "red"}),
        _response(""),
        _response("```json\n{}\n```"),
        _response('{"color":3}'),
        _response('{"color":"red","track_id":99}'),
        _response('{"color":NaN}'),
    ],
)
def test_invalid_outputs_are_terminal(monkeypatch: pytest.MonkeyPatch, envelope: Any) -> None:
    """A partial or invalid result never becomes a successful attribute update."""
    _transport(monkeypatch, lambda request: httpx.Response(200, json=envelope))
    with pytest.raises(AttributeAnalysisError) as error:
        _extract(OpenAIBackend(_config()))
    assert not isinstance(error.value, TransientAttributeError)
    assert "private-provider-text" not in str(error.value)


def test_json_mode_still_validates_the_authored_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider JSON mode cannot bypass the local attribute schema."""
    _transport(monkeypatch, lambda request: httpx.Response(200, json=_response('{"color":3}')))
    with pytest.raises(AttributeAnalysisError, match="not of type 'string'"):
        _extract(OpenAIBackend(_config(response_format="json_object")))


@pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
def test_http_failures_and_retry_after(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    """Only transient HTTP failures retry, without including the provider body."""
    _transport(
        monkeypatch,
        lambda request: httpx.Response(status, headers={"Retry-After": "2.5"}, text="private-provider-text"),
    )
    with pytest.raises(AttributeAnalysisError, match=f"OpenAI returned HTTP {status}.") as error:
        _extract(OpenAIBackend(_config()))
    assert isinstance(error.value, TransientAttributeError) is (status == 429 or status >= 500)
    assert "private-provider-text" not in str(error.value)
    if isinstance(error.value, TransientAttributeError):
        assert error.value.retry_after_s == 2.5


def test_transport_error_redacts_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Transport messages cannot expose credentialed URLs or headers in events."""
    def handler(request: Any) -> Any:
        raise httpx.ConnectError("private-test-token", request=request)

    _transport(monkeypatch, handler)
    with pytest.raises(TransientAttributeError, match=r"OpenAI request failed \(ConnectError\).") as error:
        _extract(OpenAIBackend(_config()))
    assert "private-test-token" not in str(error.value)
