import time

import pytest
from google.api_core.exceptions import DeadlineExceeded, ResourceExhausted
from google.protobuf import duration_pb2
from google.rpc import error_details_pb2
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import app


class FakeClient:
    """Stands in for the google generative service client. No network."""

    def __init__(self, exc):
        self.exc = exc
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        raise self.exc


class FakeVectorStore:
    def __init__(self, documents):
        self.documents = documents

    def as_retriever(self):
        return RunnableLambda(lambda _query: list(self.documents))


def _install_fake_gemini(monkeypatch, exc):
    """Wire the real chain to a real bounded model with a fake RPC client."""
    model = app._BoundedChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0.2,
        google_api_key="test-key",
        timeout=app.LLM_TIMEOUT_SECONDS,
    )
    client = FakeClient(exc)
    model.client = client

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(app, "_get_llm", lambda: model)
    monkeypatch.setattr(
        app,
        "_load_or_build_vectorstore",
        lambda: FakeVectorStore([Document(page_content="Policy text.")]),
    )
    monkeypatch.setattr(app, "_chat_chain", None)
    return client


def _quota_exception(message="429 quota_value: 20", retry_seconds=79160):
    details = []
    if retry_seconds is not None:
        details = [
            error_details_pb2.RetryInfo(
                retry_delay=duration_pb2.Duration(seconds=retry_seconds)
            )
        ]
    return ResourceExhausted(message, details=details)


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
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def should_not_build():
        raise AssertionError("the index must not be built without a key")

    monkeypatch.setattr(app, "_get_chat_chain", should_not_build)

    answer = app.answer_question("What is the policy?")

    assert "GEMINI_API_KEY" in answer
    assert "not configured" in answer


def test_answer_question_reports_generation_failure(monkeypatch):
    class BrokenChain:
        def invoke(self, *args, **kwargs):
            raise RuntimeError("429 quota exceeded")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(app, "_get_chat_chain", lambda: BrokenChain())

    answer = app.answer_question("What is the policy?")

    assert answer == app.GENERATION_ERROR_MESSAGE
    assert "RuntimeError" not in answer
    assert "429 quota exceeded" not in answer


def test_answer_question_reports_index_failure(monkeypatch):
    def broken_chain():
        raise RuntimeError("embedding model download failed")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(app, "_get_chat_chain", broken_chain)

    answer = app.answer_question("What is the policy?")

    assert answer == app.INDEX_ERROR_MESSAGE
    assert "embedding model download failed" not in answer


def test_answer_question_reports_quota_with_reset_time(monkeypatch):
    client = _install_fake_gemini(monkeypatch, _quota_exception())

    started = time.monotonic()
    answer = app.answer_question("What is the policy?")
    elapsed = time.monotonic() - started

    assert "free-tier daily quota is exhausted" in answer
    assert "22 hours" in answer
    assert "79160 seconds" in answer
    assert app.GENERATION_ERROR_MESSAGE not in answer
    assert "test-key" not in answer
    assert "Traceback" not in answer
    assert len(client.calls) == 1
    assert elapsed < 5


def test_answer_question_reports_quota_without_retry_delay(monkeypatch):
    _install_fake_gemini(monkeypatch, _quota_exception(retry_seconds=None))

    answer = app.answer_question("What is the policy?")

    assert "free-tier daily quota is exhausted" in answer
    assert "did not report when the quota resets" in answer


def test_answer_question_reports_timeout_quickly(monkeypatch):
    client = _install_fake_gemini(monkeypatch, DeadlineExceeded("deadline exceeded"))

    started = time.monotonic()
    answer = app.answer_question("What is the policy?")
    elapsed = time.monotonic() - started

    assert "did not respond in time" in answer
    assert f"{app.LLM_TIMEOUT_SECONDS:.0f} seconds" in answer
    assert "Traceback" not in answer
    assert len(client.calls) == 1
    assert elapsed < 5


def test_bounded_model_makes_one_attempt_with_timeout(monkeypatch):
    client = _install_fake_gemini(monkeypatch, _quota_exception())

    app.answer_question("What is the policy?")

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["retry"] is None
    assert call["timeout"] == app.LLM_TIMEOUT_SECONDS


@pytest.mark.parametrize(
    "message, expected",
    [
        ("429 retry_delay { seconds: 79160 }", 79160),
        ('{"retryDelay": "120s"}', 120),
        ("429 quota exceeded", None),
    ],
)
def test_retry_delay_seconds_parses_known_shapes(message, expected):
    assert app._retry_delay_seconds(ResourceExhausted(message)) == expected


def test_startup_badge_reflects_key_presence(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert "not set" in app._startup_badge()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert "is set" in app._startup_badge()
