import asyncio

import httpx
import openai
import pytest


def _connection_error():
    return openai.APIConnectionError(request=httpx.Request("POST", "http://localhost:1/v1"))


# ---------- parse_stance_response ----------

def test_parse_plain_json():
    import prompts
    text = '{"tweet-0": "1", "tweet-1": 0, "tweet-2": "-1"}'
    assert prompts.parse_stance_response(text, 0, 3) == {0: 1, 1: 0, 2: -1}


def test_parse_json_wrapped_in_prose():
    import prompts
    text = 'Sure! Here are the stances:\n{"tweet-5": 2, "tweet-6": 0}\nLet me know.'
    assert prompts.parse_stance_response(text, 5, 7) == {5: 2, 6: 0}


def test_parse_missing_and_invalid_keys_default_to_minus_one():
    import prompts
    text = '{"tweet-0": "yes", "tweet-2": 1}'
    assert prompts.parse_stance_response(text, 0, 3) == {0: -1, 1: -1, 2: 1}


def test_parse_garbage_returns_all_minus_one():
    import prompts
    assert prompts.parse_stance_response("no json here", 0, 2) == {0: -1, 1: -1}


# ---------- ai_summarize ----------

def test_ai_summarize_success(monkeypatch):
    import prompts

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == "Lllama3TS_unsloth_vllm"
            msg = type("M", (), {"content": "a summary (0, 1)"})
            choice = type("C", (), {"message": msg})
            return type("R", (), {"choices": [choice]})

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setattr(prompts, "_get_summarizer_client", lambda: FakeClient())
    assert prompts.ai_summarize("0-[hello] 1-[world]") == "a summary (0, 1)"


def test_ai_summarize_raises_backend_error(monkeypatch):
    import prompts

    class FakeCompletions:
        def create(self, **kwargs):
            raise _connection_error()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})

    monkeypatch.setattr(prompts, "_get_summarizer_client", lambda: FakeClient())
    with pytest.raises(prompts.BackendError) as exc:
        prompts.ai_summarize("0-[hello]")
    assert exc.value.backend == "summarizer"
    assert "8000" in exc.value.url


# ---------- stance_annotation ----------

class _FakeAsyncCompletions:
    def __init__(self, fail_json_object=False, fail_always=False):
        self.fail_json_object = fail_json_object
        self.fail_always = fail_always
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_always:
            raise _connection_error()
        if self.fail_json_object and "response_format" in kwargs:
            raise openai.BadRequestError(
                "response_format unsupported",
                response=httpx.Response(400, request=httpx.Request("POST", "http://x/v1")),
                body=None,
            )
        msg = type("M", (), {"content": '{"tweet-0": 1}'})
        choice = type("C", (), {"message": msg})
        return type("R", (), {"choices": [choice]})


def _fake_stance_client(completions):
    return type("Client", (), {"chat": type("Chat", (), {"completions": completions})})


def test_stance_annotation_uses_guided_json(monkeypatch):
    import prompts
    fake = _FakeAsyncCompletions()
    monkeypatch.setattr(prompts, "_get_stance_client", lambda: _fake_stance_client(fake))
    result = asyncio.run(prompts.stance_annotation("0-[hi]", "census", ["pro", "anti"], {}))
    assert result == '{"tweet-0": 1}'
    assert fake.calls[0]["response_format"] == {"type": "json_object"}


def test_stance_annotation_falls_back_without_response_format(monkeypatch):
    import prompts
    fake = _FakeAsyncCompletions(fail_json_object=True)
    monkeypatch.setattr(prompts, "_get_stance_client", lambda: _fake_stance_client(fake))
    result = asyncio.run(prompts.stance_annotation("0-[hi]", "census", ["pro", "anti"], {}))
    assert result == '{"tweet-0": 1}'
    assert len(fake.calls) == 2
    assert "response_format" not in fake.calls[1]


def test_stance_annotation_raises_backend_error(monkeypatch):
    import prompts
    fake = _FakeAsyncCompletions(fail_always=True)
    monkeypatch.setattr(prompts, "_get_stance_client", lambda: _fake_stance_client(fake))
    with pytest.raises(prompts.BackendError) as exc:
        asyncio.run(prompts.stance_annotation("0-[hi]", "census", ["pro"], {}))
    assert exc.value.backend == "stance"


# ---------- check_backends ----------

def test_check_backends_all_up(monkeypatch):
    import prompts

    def fake_get(url, headers=None, timeout=None):
        return httpx.Response(200, request=httpx.Request("GET", url))

    monkeypatch.setattr(prompts.httpx, "get", fake_get)
    assert prompts.check_backends() == {"summarizer": True, "stance": True}


def test_check_backends_down(monkeypatch):
    import prompts

    def fake_get(url, headers=None, timeout=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(prompts.httpx, "get", fake_get)
    assert prompts.check_backends() == {"summarizer": False, "stance": False}
