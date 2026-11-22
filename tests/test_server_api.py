"""Protocol tests against the mock backend.

These do not test model quality — they pin the wire contract: auth, OpenAI
object shapes, SSE framing, usage accounting, clamping and rate limits.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server.config import Settings
from server.main import create_app

KEY = "sk-test-key"


def make_client(tmp_path: Path, **overrides) -> TestClient:
    base = dict(
        backend="mock",
        model_name="extra-code-30m",
        db_path=tmp_path / "keys.db",
        bootstrap_key=KEY,
        api_key_pepper="pepper",
        default_rpm=1000,
        default_tpm=10_000_000,
        default_key_max_tokens=256,
        n_ctx=256,
        default_max_tokens=64,
        max_tokens_cap=128,
        max_concurrent=2,
        queue_timeout_s=2.0,
        generation_timeout_s=15.0,
        chat_template="chatml",
    )
    base.update(overrides)
    settings = Settings(**base)
    app = create_app(settings)
    client = TestClient(app)
    client.headers.update({"Authorization": f"Bearer {KEY}"})
    return client


@pytest.fixture()
def client(tmp_path: Path):
    with make_client(tmp_path) as c:
        yield c


def parse_sse(body: str) -> list[dict]:
    events: list[dict] = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


def test_health_is_public(client: TestClient) -> None:
    client.headers.pop("Authorization")
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["engine"]["backend"] == "mock"
    assert body["model_loaded"] is True


def test_missing_key_is_401(client: TestClient) -> None:
    client.headers.pop("Authorization")
    response = client.get("/v1/models")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_bad_key_is_401(client: TestClient) -> None:
    response = client.get("/v1/models", headers={"Authorization": "Bearer nope"})
    assert response.status_code == 401


def test_models_are_listed(client: TestClient) -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert body["data"][0]["id"] == "extra-code-30m"


def test_unknown_model_is_404(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"model": "gpt-9", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_empty_messages_is_400(client: TestClient) -> None:
    response = client.post("/v1/chat/completions", json={"model": "extra-code-30m", "messages": []})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_chat_completion_shape(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "extra-code-30m",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "say something"},
            ],
            "max_tokens": 32,
            "temperature": 0.2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] in {"stop", "length"}
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens"] > 0
    assert response.headers["x-request-id"]


def test_chat_stream_frames(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "extra-code-30m",
            "messages": [{"role": "user", "content": "stream please"}],
            "max_tokens": 24,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())

    assert body.rstrip().endswith("data: [DONE]")
    events = parse_sse(body)
    assert events[0]["object"] == "chat.completion.chunk"
    assert events[0]["choices"][0]["delta"]["role"] == "assistant"
    text = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e["choices"])
    assert text.strip()
    assert any(e.get("usage") for e in events)
    last_with_choices = [e for e in events if e.get("choices")][-1]
    assert last_with_choices["choices"][0]["finish_reason"] in {"stop", "length"}


def test_completion_non_stream(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "extra-code-30m", "prompt": "def add(a, b):", "max_tokens": 16},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")
    assert isinstance(body["choices"][0]["text"], str)
    assert body["usage"]["prompt_tokens"] > 0


def test_completion_stream(client: TestClient) -> None:
    with client.stream(
        "POST",
        "/v1/completions",
        json={"model": "extra-code-30m", "prompt": "hello", "stream": True, "max_tokens": 12},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())
    assert body.rstrip().endswith("data: [DONE]")
    events = parse_sse(body)
    assert "".join(e["choices"][0]["text"] for e in events if e["choices"]).strip()


def test_multi_prompt_completion(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "extra-code-30m", "prompt": ["one", "two"], "max_tokens": 8},
    )
    assert response.status_code == 200
    assert len(response.json()["choices"]) == 2
    assert [c["index"] for c in response.json()["choices"]] == [0, 1]


def test_multi_prompt_stream_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/completions",
        json={"model": "extra-code-30m", "prompt": ["one", "two"], "stream": True},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_max_tokens_capped_by_key(tmp_path: Path) -> None:
    with make_client(tmp_path, default_key_max_tokens=3, max_tokens_cap=128) as c:
        response = c.post(
            "/v1/chat/completions",
            json={"model": "extra-code-30m", "messages": [{"role": "user", "content": "count"}], "max_tokens": 100},
        )
    assert response.status_code == 200
    assert response.json()["usage"]["completion_tokens"] <= 3


def test_rate_limit_rpm(tmp_path: Path) -> None:
    with make_client(tmp_path, default_rpm=1) as c:
        first = c.post(
            "/v1/chat/completions",
            json={"model": "extra-code-30m", "messages": [{"role": "user", "content": "one"}]},
        )
        second = c.post(
            "/v1/chat/completions",
            json={"model": "extra-code-30m", "messages": [{"role": "user", "content": "two"}]},
        )
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "requests_per_minute"
    assert "Retry-After" in second.headers


def test_tokenize_endpoint(client: TestClient) -> None:
    response = client.post("/v1/tokenize", json={"text": "hello world"})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] > 0
    assert len(body["ids"]) == body["count"]


def test_long_conversation_is_fitted(client: TestClient) -> None:
    messages = [{"role": "user", "content": "x" * 2000}]
    messages += [{"role": "assistant", "content": "y" * 2000} for _ in range(20)]
    messages.append({"role": "user", "content": "last"})
    response = client.post(
        "/v1/chat/completions",
        json={"model": "extra-code-30m", "messages": messages, "max_tokens": 8},
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"]
