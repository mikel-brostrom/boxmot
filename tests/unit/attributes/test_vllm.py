"""vLLM vision chat protocol and shared attribute-session integration."""

import asyncio
import base64
import json
from datetime import datetime, timezone, tzinfo

import cv2
import numpy as np
import pytest
import torch

import boxmot.attributes.backends.http as attribute_http
from boxmot.attributes.backends.vllm import VLLMBackend
from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.errors import AttributeAnalysisError, TransientAttributeError
from boxmot.attributes.models import AttributeSample
from boxmot.attributes.session import AttributeEnricher
from boxmot.structures import Boxes, Frame, Tracks

httpx = pytest.importorskip("httpx")
pytest.importorskip("jsonschema")
pytest.importorskip("referencing")


def _config(**overrides):
    values = {
        "backend": "vllm",
        "model": "Qwen/Qwen3.5-9B",
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


def _response(content='{"color":"red"}', **message_fields):
    return {"choices": [{"finish_reason": "stop", "message": {"content": content, **message_fields}}]}


def _transport(monkeypatch, handler):
    client_type = httpx.AsyncClient
    monkeypatch.setattr(attribute_http, "require_extra", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs)
    )


def _extract(backend):
    samples = [AttributeSample("sample-0", 0, 0.0), AttributeSample("sample-30", 30, 1.0)]
    images = []
    for color in ((0, 0, 255), (255, 0, 0)):
        success, encoded = cv2.imencode(".jpg", np.full((24, 16, 3), color, dtype=np.uint8))
        assert success
        images.append(encoded.tobytes())

    async def run():
        try:
            return await backend.extract(7, samples, images)
        finally:
            await backend.aclose()

    return asyncio.run(run())


@pytest.mark.parametrize("think", [False, True, None])
def test_schema_and_chronological_storyboard_in_vision_chat_request(monkeypatch, think):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_response(reasoning="not JSON", reasoning_content="also not JSON"))

    _transport(monkeypatch, handler)
    config = _config(think=think)
    backend = VLLMBackend(config)
    assert _extract(backend) == {"color": "red"}
    request = requests[0]
    assert str(request.url) == "http://localhost:8000/v1/chat/completions"
    assert "Authorization" not in request.headers
    payload = json.loads(request.content)
    assert payload["model"] == config.model
    assert payload["response_format"] == {
        "type": "json_schema", "json_schema": {"name": "track_attributes", "schema": config.schema}
    }
    assert payload["stream"] is False and payload["temperature"] == 0
    if think is None:
        assert "chat_template_kwargs" not in payload
    else:
        assert payload["chat_template_kwargs"] == {"enable_thinking": think}
    assert len(payload["messages"]) == 1
    message = payload["messages"][0]
    assert message["role"] == "user"
    text, image = message["content"]
    assert text["type"] == "text" and config.prompt in text["text"]
    assert json.dumps(config.schema) in text["text"]
    assert image["type"] == "image_url"
    prefix, encoded = image["image_url"]["url"].split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    storyboard = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    assert storyboard.shape == (480, 640, 3)
    assert storyboard[240, 160, 2] > 240  # First tile remains red in BGR.
    assert storyboard[240, 480, 0] > 240  # Later tile remains blue.
    assert backend._client.is_closed


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("http://localhost:8000/", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
        ("https://example.com/vision/v1/", "https://example.com/vision/v1/chat/completions"),
        ("https://example.com/vision/", "https://example.com/vision/v1/chat/completions"),
    ],
)
def test_api_base_and_bearer_auth(monkeypatch, host, expected):
    monkeypatch.setenv("ATTRIBUTE_TEST_TOKEN", "private-test-token")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_response())

    _transport(monkeypatch, handler)
    assert _extract(VLLMBackend(_config(host=host, api_key_env="ATTRIBUTE_TEST_TOKEN"))) == {"color": "red"}
    assert str(requests[0].url) == expected
    assert requests[0].headers["Authorization"] == "Bearer private-test-token"
    assert "private-test-token" not in requests[0].content.decode()


def test_infistar_json_object_request_with_bearer_auth(monkeypatch):
    """Hosted JSON mode uses vision messages without vLLM-only request options."""
    monkeypatch.setenv("INFISTAR_API_KEY", "private-test-token")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=_response())

    _transport(monkeypatch, handler)
    config = _config(
        host="https://infistar.ai/v1",
        model="test-vision-model",
        api_key_env="INFISTAR_API_KEY",
        response_format="json_object",
        think=None,
    )
    assert _extract(VLLMBackend(config)) == {"color": "red"}
    request = requests[0]
    assert str(request.url) == "https://infistar.ai/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer private-test-token"
    assert "private-test-token" not in request.content.decode()
    payload = json.loads(request.content)
    assert payload["model"] == "test-vision-model"
    assert payload["response_format"] == {"type": "json_object"}
    assert "chat_template_kwargs" not in payload
    system, user = payload["messages"]
    assert system["role"] == "system" and "JSON" in system["content"]
    assert user["role"] == "user"
    text, image = user["content"]
    assert config.prompt in text["text"] and json.dumps(config.schema) in text["text"]
    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_json_object_mode_validates_response_against_local_schema(monkeypatch):
    """A syntactically valid provider JSON response must still satisfy our schema."""
    _transport(monkeypatch, lambda request: httpx.Response(200, json=_response('{"color":3}')))
    with pytest.raises(AttributeAnalysisError, match="not of type 'string'"):
        _extract(VLLMBackend(_config(response_format="json_object", think=None)))


@pytest.mark.parametrize(
    "envelope",
    [
        None,
        {"choices": []},
        {"choices": [None]},
        {"choices": [{"message": None}]},
        {"choices": [{"message": {"reasoning_content": '{"color":"red"}'}}]},
        _response(None),
        _response(""),
        _response({"color": "red"}),
        _response("```json\n{}\n```"),
        _response('{"color":3}'),
        _response('{"color":"red","track_id":99}'),
        _response('{"color":NaN}'),
        _response(refusal="Refused"),
        _response(tool_calls=[{"function": {"name": "anything"}}]),
        {"choices": [{"finish_reason": "length", "message": {"content": '{"color":"red"}'}}]},
    ],
)
def test_invalid_outputs_are_terminal_failures(monkeypatch, envelope):
    _transport(monkeypatch, lambda request: httpx.Response(200, json=envelope))
    with pytest.raises(AttributeAnalysisError) as error:
        _extract(VLLMBackend(_config()))
    assert not isinstance(error.value, TransientAttributeError)


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 503])
def test_http_status_classification(monkeypatch, status):
    _transport(monkeypatch, lambda request: httpx.Response(status))
    with pytest.raises(AttributeAnalysisError, match=str(status)) as error:
        _extract(VLLMBackend(_config()))
    assert isinstance(error.value, TransientAttributeError) is (status == 429 or status >= 500)


@pytest.mark.parametrize("status", [429, 500, 503])
@pytest.mark.parametrize("retry_after", ["0", "10", "2.5"])
def test_transient_http_errors_preserve_retry_after(monkeypatch, status: int, retry_after: str) -> None:
    """Provider retry delays survive HTTP error classification for scheduling."""
    _transport(monkeypatch, lambda request: httpx.Response(status, headers={"Retry-After": retry_after}))
    with pytest.raises(TransientAttributeError) as error:
        _extract(VLLMBackend(_config()))
    assert error.value.retry_after_s == float(retry_after)
    assert str(error.value) == f"vLLM returned HTTP {status}."


@pytest.mark.parametrize("retry_after", [None, "", "invalid-header", "-1", "nan", "inf", "1e999"])
def test_invalid_retry_after_leaves_default_backoff(monkeypatch, retry_after: str | None) -> None:
    """Missing or invalid delays never replace the session's own retry policy."""
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    _transport(monkeypatch, lambda request: httpx.Response(429, headers=headers))
    with pytest.raises(TransientAttributeError) as error:
        _extract(VLLMBackend(_config()))
    assert error.value.retry_after_s is None
    assert str(error.value) == "vLLM returned HTTP 429."


@pytest.mark.parametrize(
    ("retry_after", "expected"),
    [("Wed, 01 Jan 2025 00:00:10 GMT", 10.0), ("Tue, 31 Dec 2024 23:59:59 GMT", 0.0)],
)
def test_retry_after_http_date(monkeypatch, retry_after: str, expected: float) -> None:
    """Absolute HTTP dates become nonnegative delays relative to the current UTC time."""
    class FixedDatetime(datetime):
        """Freeze the HTTP backend's clock for date header checks."""

        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            """Return the test instant without consulting the wall clock."""
            return datetime(2025, 1, 1, tzinfo=timezone.utc)

    monkeypatch.setattr(attribute_http, "datetime", FixedDatetime)
    _transport(monkeypatch, lambda request: httpx.Response(503, headers={"Retry-After": retry_after}))
    with pytest.raises(TransientAttributeError) as error:
        _extract(VLLMBackend(_config()))
    assert error.value.retry_after_s == expected


@pytest.mark.parametrize("backend", ["ollama", "vllm"])
def test_config_selects_backend_and_session_keeps_application_identity(monkeypatch, tmp_path, backend):
    requests = []

    def handler(request):
        requests.append(request)
        envelope = _response() if backend == "vllm" else {"response": '{"color":"red"}'}
        return httpx.Response(200, json=envelope)

    _transport(monkeypatch, handler)
    frame = Frame(torch.zeros((3, 16, 16), dtype=torch.uint8), "clip/30", "clip", 30)
    tracks = Tracks(
        Boxes(torch.tensor([[1, 1, 14, 14]], dtype=torch.float32)),
        torch.tensor([7]), torch.ones(1), torch.tensor([0]), torch.tensor([0]), frame.sample_id,
    )
    with AttributeEnricher(_config(backend=backend, batch_size=1, max_crop_side=8), spool_dir=tmp_path) as session:
        session.update(frame, tracks)
        outcomes = list(session.finish())
        assert len(outcomes) == 1
        result = outcomes[0]
        assert result.status == "completed" and result.values == {"color": "red"}
        assert result.track_id == 7 and result.sequence_id == "clip"
        assert result.run_id == session.run_id and result.generation == 0
        assert result.samples[0].sample_id == "clip/30" and result.samples[0].frame_index == 30
    assert len(requests) == 1
    assert requests[0].url.path == ("/v1/chat/completions" if backend == "vllm" else "/api/generate")
    payload = json.loads(requests[0].content)
    if backend == "vllm":
        image_url = payload["messages"][0]["content"][1]["image_url"]["url"]
        encoded = image_url.split(",", 1)[1]
    else:
        encoded = payload["images"][0]
    image = cv2.imdecode(np.frombuffer(base64.b64decode(encoded), np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (8, 8, 3)  # No padded storyboard enlarges the first bounded crop.
