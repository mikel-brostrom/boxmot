import asyncio
import base64
import json
import traceback

import cv2
import numpy as np
import pytest

import boxmot.attributes.http as attribute_http
from boxmot.attributes.config import AttributeConfig
from boxmot.attributes.errors import AttributeAnalysisError, TransientAttributeError
from boxmot.attributes.models import AttributeSample
from boxmot.attributes.ollama import OllamaBackend
from boxmot.utils.config import ConfigurationError

httpx = pytest.importorskip("httpx")
pytest.importorskip("jsonschema")
pytest.importorskip("referencing")


def _config(**overrides):
    values = {
        "model": "vision",
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


def _input():
    success, encoded = cv2.imencode(".jpg", np.full((24, 16, 3), 120, dtype=np.uint8))
    assert success
    return [AttributeSample("sample-1", 1, 0.1)], [encoded.tobytes()]


def _backend(monkeypatch, handler, config=None):
    client_type = httpx.AsyncClient
    monkeypatch.setattr(attribute_http, "require_extra", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: client_type(transport=httpx.MockTransport(handler), **kwargs)
    )
    return OllamaBackend(config or _config())


@pytest.mark.parametrize("think", [False, True, None])
def test_schema_and_storyboard_request_with_application_owned_provenance(monkeypatch, think):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"response": '{"color": "red"}', "thinking": "not parsed"})

    backend = _backend(monkeypatch, handler, _config(think=think))

    async def run():
        try:
            return await backend.extract(7, *_input())
        finally:
            await backend.aclose()

    assert asyncio.run(run()) == {"color": "red"}
    request = requests[0]
    assert str(request.url) == "http://localhost:11434/api/generate"
    assert "Authorization" not in request.headers
    payload = json.loads(request.content)
    assert payload["format"] == _config().schema
    assert payload["stream"] is False
    if think is None:
        assert "think" not in payload
    else:
        assert payload["think"] is think
    assert payload["options"] == {"temperature": 0}
    assert "Describe color." in payload["prompt"]
    assert json.dumps(_config().schema) in payload["prompt"]
    assert len(payload["images"]) == 1
    assert cv2.imdecode(np.frombuffer(base64.b64decode(payload["images"][0]), np.uint8), cv2.IMREAD_COLOR) is not None
    assert backend._client.is_closed


@pytest.mark.parametrize("status", [429, 500, 503])
def test_retryable_http_statuses(monkeypatch, status):
    backend = _backend(monkeypatch, lambda request: httpx.Response(status))

    async def run():
        try:
            with pytest.raises(TransientAttributeError, match=str(status)):
                await backend.extract(7, *_input())
        finally:
            await backend.aclose()

    asyncio.run(run())


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(404),
        httpx.Response(200, json={"response": ""}),
        httpx.Response(200, json={"thinking": '{"color":"red"}'}),
        httpx.Response(200, json={"response": "```json\n{}\n```"}),
        httpx.Response(200, json={"response": '{"color": 3}'}),
        httpx.Response(200, json={"response": '{"color":"red","track_id":99}'}),
        httpx.Response(200, json={"response": '{"color":NaN}'}),
        httpx.Response(200, content=b"not json"),
    ],
)
def test_terminal_errors_and_ignores_thinking(monkeypatch, response):
    backend = _backend(monkeypatch, lambda request: response)

    async def run():
        try:
            with pytest.raises(AttributeAnalysisError) as error:
                await backend.extract(7, *_input())
            assert not isinstance(error.value, TransientAttributeError)
        finally:
            await backend.aclose()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["timeout", "connection"])
def test_finite_total_timeout_and_transport_failure_are_transient(monkeypatch, failure):
    async def handler(request):
        if failure == "connection":
            raise httpx.ConnectError("service unavailable", request=request)
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={"response": '{"color":"red"}'})

    backend = _backend(monkeypatch, handler, _config(request_timeout_s=0.01))

    async def run():
        try:
            with pytest.raises(TransientAttributeError):
                await backend.extract(7, *_input())
        finally:
            await backend.aclose()

    asyncio.run(run())


def test_schema_is_checked_before_any_request(monkeypatch):
    monkeypatch.setattr(attribute_http, "require_extra", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: pytest.fail("HTTP client constructed for invalid schema")
    )
    with pytest.raises(ConfigurationError, match="Invalid attribute JSON Schema"):
        OllamaBackend(_config(schema={"type": "object", "properties": {"bad": {"type": "invalid"}}}))


def test_local_schema_reference_validates_response(monkeypatch):
    schema = {
        "type": "object",
        "$defs": {"word": {"type": "string"}},
        "properties": {"color": {"$ref": "#/$defs/word"}},
        "required": ["color"],
    }
    backend = _backend(
        monkeypatch,
        lambda request: httpx.Response(200, json={"response": '{"color":"red"}'}),
        _config(schema=schema),
    )

    async def run():
        try:
            return await backend.extract(7, *_input())
        finally:
            await backend.aclose()

    assert asyncio.run(run()) == {"color": "red"}


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {"color": {"$ref": "#/$defs/missing"}}},
        {"type": "object", "$schema": "https://example.com/custom-dialect"},
    ],
)
def test_unresolved_local_references_and_unknown_dialects_fail_before_client_creation(monkeypatch, schema):
    monkeypatch.setattr(attribute_http, "require_extra", lambda *args, **kwargs: None)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("client must not be created"))
    with pytest.raises(ConfigurationError):
        OllamaBackend(_config(schema=schema))


def test_optional_bearer_authentication(monkeypatch):
    monkeypatch.setenv("ATTRIBUTE_TEST_TOKEN", "private-test-token")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"response": '{"color":"red"}'})

    backend = _backend(monkeypatch, handler, _config(api_key_env="ATTRIBUTE_TEST_TOKEN"))

    async def run():
        try:
            return await backend.extract(7, *_input())
        finally:
            await backend.aclose()

    assert asyncio.run(run()) == {"color": "red"}
    assert requests[0].headers["Authorization"] == "Bearer private-test-token"
    assert "private-test-token" not in requests[0].content.decode()


@pytest.mark.parametrize("value", [None, "", "  "])
def test_missing_api_key_fails_before_client_creation(monkeypatch, value):
    monkeypatch.setattr(attribute_http, "require_extra", lambda *args, **kwargs: None)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: pytest.fail("client must not be created"))
    if value is None:
        monkeypatch.delenv("ATTRIBUTE_TEST_TOKEN", raising=False)
    else:
        monkeypatch.setenv("ATTRIBUTE_TEST_TOKEN", value)
    with pytest.raises(ConfigurationError, match="ATTRIBUTE_TEST_TOKEN.*must be set and nonempty"):
        OllamaBackend(_config(api_key_env="ATTRIBUTE_TEST_TOKEN"))


def test_transport_failure_does_not_expose_credentials(monkeypatch):
    monkeypatch.setenv("ATTRIBUTE_TEST_TOKEN", "private-test-token")

    def handler(request):
        raise httpx.ConnectError("private-test-token", request=request)

    backend = _backend(monkeypatch, handler, _config(api_key_env="ATTRIBUTE_TEST_TOKEN"))

    async def run():
        try:
            with pytest.raises(TransientAttributeError, match="ConnectError") as error:
                await backend.extract(7, *_input())
            assert "private-test-token" not in str(error.value)
            assert "private-test-token" not in "".join(traceback.format_exception(error.value))
        finally:
            await backend.aclose()

    asyncio.run(run())
