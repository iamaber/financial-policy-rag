from langchain_core.documents import Document

import app


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

    assert "Gemini could not generate an answer" in answer
    assert "quota" in answer


def test_answer_question_reports_index_failure(monkeypatch):
    def broken_chain():
        raise RuntimeError("embedding model download failed")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(app, "_get_chat_chain", broken_chain)

    answer = app.answer_question("What is the policy?")

    assert "document index could not be prepared" in answer
    assert "embedding model download failed" in answer


def test_startup_badge_reflects_key_presence(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert "not set" in app._startup_badge()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    assert "is set" in app._startup_badge()
