import importlib
from types import SimpleNamespace

import httpx
import pytest
from groq import APITimeoutError, RateLimitError
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import app


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def as_retriever(self):
        return RunnableLambda(lambda _query: list(self.documents))


def _install_broken_chain(monkeypatch, exc):
    class BrokenChain:
        def invoke(self, *args, **kwargs):
            raise exc

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(app, "_get_chat_chain", lambda: BrokenChain())
    return BrokenChain()


def _rate_limit_exception(retry_after="120"):
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    headers = {} if retry_after is None else {"retry-after": retry_after}
    response = httpx.Response(
        429,
        request=request,
        headers=headers,
        json={"error": {"message": "rate limit"}},
    )
    return RateLimitError("rate limit", response=response, body=None)


def _timeout_exception():
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return APITimeoutError(request=request)


def test_build_documents_keeps_page_and_paragraph_metadata():
    pages = [{"page": 3, "content": ("Policy sentence. " * 120).strip()}]

    documents = app.build_documents(pages)

    assert len(documents) > 1
    assert [doc.metadata["page"] for doc in documents] == [3] * len(documents)
    assert [doc.metadata["paragraph"] for doc in documents] == list(
        range(1, len(documents) + 1)
    )
    assert all(doc.page_content.strip() for doc in documents)


def test_build_documents_skips_pages_without_text():
    pages = [
        {"page": 1, "content": None},
        {"page": 2, "content": "   "},
        {"page": 3, "content": "Only this page has text."},
    ]

    documents = app.build_documents(pages)

    assert len(documents) == 1
    assert documents[0].metadata["page"] == 3


def test_index_fingerprint_changes_with_content():
    first = [Document(page_content="alpha", metadata={"page": 1, "paragraph": 1})]
    changed = [Document(page_content="beta", metadata={"page": 1, "paragraph": 1})]

    assert app.index_fingerprint(first) == app.index_fingerprint(first)
    assert app.index_fingerprint(first) != app.index_fingerprint(changed)


def test_extract_text_returns_empty_list_for_missing_file(tmp_path):
    assert app.extract_text(tmp_path / "missing.pdf") == []


def test_answer_question_explains_missing_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def should_not_build():
        raise AssertionError("the index must not be built without a key")

    monkeypatch.setattr(app, "_get_chat_chain", should_not_build)

    answer = app.answer_question("What is the policy?")

    assert "GROQ_API_KEY" in answer
    assert "not configured" in answer


def test_answer_question_reports_generation_failure(monkeypatch):
    _install_broken_chain(monkeypatch, RuntimeError("unexpected provider failure"))

    answer = app.answer_question("What is the policy?")

    assert answer == app.GENERATION_ERROR_MESSAGE
    assert "RuntimeError" not in answer
    assert "unexpected provider failure" not in answer


def test_answer_question_reports_index_failure(monkeypatch):
    def broken_chain():
        raise RuntimeError("embedding model download failed")

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(app, "_get_chat_chain", broken_chain)

    answer = app.answer_question("What is the policy?")

    assert answer == app.INDEX_ERROR_MESSAGE
    assert "embedding model download failed" not in answer


def test_answer_question_reports_rate_limit_with_wait_time(monkeypatch):
    _install_broken_chain(monkeypatch, _rate_limit_exception("120"))

    answer = app.answer_question("What is the policy?")

    assert "Groq rate limit reached" in answer
    assert "about 2 minutes (120 seconds)" in answer
    assert "test-key" not in answer
    assert "Traceback" not in answer


def test_answer_question_reports_rate_limit_without_wait_time(monkeypatch):
    _install_broken_chain(monkeypatch, _rate_limit_exception(None))

    answer = app.answer_question("What is the policy?")

    assert "Groq rate limit reached" in answer
    assert "did not report how long to wait" in answer


def test_answer_question_reports_timeout(monkeypatch):
    _install_broken_chain(monkeypatch, _timeout_exception())

    answer = app.answer_question("What is the policy?")

    assert "did not respond in time" in answer
    assert f"{app.LLM_TIMEOUT_SECONDS:.0f} seconds" in answer
    assert "Traceback" not in answer


def test_llm_is_configured_for_one_attempt_with_timeout(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    llm = app._get_llm()

    assert llm.model_name == app.LLM_MODEL
    assert llm.max_retries == 0
    assert llm.request_timeout == app.LLM_TIMEOUT_SECONDS


@pytest.mark.parametrize("effort", [None, "low", ""])
def test_llm_sends_configured_reasoning_effort_or_omits_it(monkeypatch, effort):
    import dotenv

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    # Preserve module configuration after reloading it with controlled env values.
    monkeypatch.setattr(app, "LLM_MODEL", app.LLM_MODEL)
    monkeypatch.setattr(app, "GROQ_REASONING_EFFORT", app.GROQ_REASONING_EFFORT)
    if effort is None:
        monkeypatch.delenv("GROQ_REASONING_EFFORT", raising=False)
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        expected_model = "qwen/qwen3.8-27b"
    else:
        monkeypatch.setenv("GROQ_REASONING_EFFORT", effort)
        monkeypatch.setenv("GROQ_MODEL", "alternate-model")
        expected_model = "alternate-model"
    importlib.reload(app)
    llm = app._get_llm()
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Cited answer"},
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr(llm, "client", SimpleNamespace(create=fake_create))
    assert llm.invoke("What is the policy?").content == "Cited answer"

    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == expected_model
    assert call["temperature"] == 0.2
    assert llm.model_kwargs == {}
    if effort == "":
        assert "reasoning_effort" not in call
    else:
        assert call["reasoning_effort"] == (effort or "none")


@pytest.mark.parametrize(
    "retry_after, expected",
    [
        ("120", 120),
        ("1.5", 1),
        (None, None),
        ("not-a-number", None),
    ],
)
def test_retry_after_seconds_parses_known_shapes(retry_after, expected):
    assert app._retry_after_seconds(_rate_limit_exception(retry_after)) == expected


def test_startup_badge_reflects_key_presence(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert "not set" in app._startup_badge()

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert "is set" in app._startup_badge()
