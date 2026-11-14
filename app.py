"""Gradio RAG demo for the Financial Policy document.

Document search is entirely local (sentence-transformers + Chroma), so the app
starts without any remote embedding quota. Google Gemini is only used to write
the final answer, behind the optional GEMINI_API_KEY.
"""

import hashlib
import json
import logging
import os
import shutil
import threading
from pathlib import Path

import gradio as gr
import PyPDF2
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
PDF_PATH = APP_DIR / "data" / "For Task - Policy file.pdf"
CHROMA_DIR = APP_DIR / "data" / "chroma_db"
COLLECTION_NAME = "financial_policy"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 300

MISSING_KEY_MESSAGE = (
    "**Answer generation is not configured.**\n\n"
    "This demo needs a Google Gemini API key to write answers. Add "
    "`GEMINI_API_KEY` as a Space secret (Settings → Variables and secrets) or "
    "set it in a local `.env` file, then restart the app.\n\n"
    "Document search uses a local embedding model and does not need an API key, "
    "so the UI and the index still work without one."
)

INDEX_ERROR_MESSAGE = (
    "**The document index could not be prepared.**\n\n"
    "The local embedding model or the PDF could not be loaded. The first "
    "request downloads a small sentence-transformers model, so check the Space "
    "logs if this keeps happening."
)

GENERATION_ERROR_MESSAGE = (
    "**Gemini could not generate an answer.**\n\n"
    "Check that `GEMINI_API_KEY` is valid and that the Google AI free-tier "
    "quota is available, then try again."
)


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain embedding adapter that loads a local model on first use."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s", self.model_name)
                    self._model = SentenceTransformer(
                        self.model_name, device="cpu", trust_remote_code=False
                    )
        return self._model

    def embed_documents(self, texts):
        vectors = self._get_model().encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def extract_text(pdf_path):
    pages = []
    path = Path(pdf_path)
    if not path.is_file():
        logger.error("Policy PDF not found at %s", path)
        return pages
    try:
        with path.open("rb") as handle:
            reader = PyPDF2.PdfReader(handle)
            for number, page in enumerate(reader.pages, 1):
                pages.append({"page": number, "content": page.extract_text() or ""})
    except Exception:
        logger.exception("Error reading PDF file %s", path)
        return []
    return pages


def build_documents(raw_pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?"],
    )
    documents = []
    for page in raw_pages:
        content = (page.get("content") or "").strip()
        if not content:
            continue
        for paragraph, chunk in enumerate(splitter.split_text(content), 1):
            if chunk.strip():
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={"page": page["page"], "paragraph": paragraph},
                    )
                )
    return documents


def index_fingerprint(documents):
    digest = hashlib.sha256()
    digest.update(EMBEDDING_MODEL.encode("utf-8"))
    digest.update(f"{CHUNK_SIZE}:{CHUNK_OVERLAP}".encode("utf-8"))
    for document in documents:
        digest.update(document.page_content.encode("utf-8"))
        digest.update(repr(sorted(document.metadata.items())).encode("utf-8"))
    return digest.hexdigest()


def _index_is_current(meta_path, fingerprint):
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        metadata.get("fingerprint") == fingerprint
        and (CHROMA_DIR / "chroma.sqlite3").is_file()
    )


_embeddings = None
_embeddings_lock = threading.Lock()
_vectorstore = None
_vectorstore_lock = threading.Lock()
_chat_chain = None
_chain_lock = threading.Lock()
_session_store = {}


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        with _embeddings_lock:
            if _embeddings is None:
                _embeddings = SentenceTransformerEmbeddings(EMBEDDING_MODEL)
    return _embeddings


def _load_or_build_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    with _vectorstore_lock:
        if _vectorstore is not None:
            return _vectorstore

        documents = build_documents(extract_text(PDF_PATH))
        if not documents:
            raise RuntimeError(
                f"No text could be extracted from {PDF_PATH.name!r}. "
                "Check that the PDF is present in data/."
            )

        fingerprint = index_fingerprint(documents)
        meta_path = CHROMA_DIR / "index_meta.json"
        if _index_is_current(meta_path, fingerprint):
            logger.info("Loading cached document index from %s", CHROMA_DIR)
            store = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=get_embeddings(),
                collection_name=COLLECTION_NAME,
            )
        else:
            logger.info("Building document index at %s", CHROMA_DIR)
            if CHROMA_DIR.exists():
                shutil.rmtree(CHROMA_DIR)
            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            store = Chroma.from_documents(
                documents,
                get_embeddings(),
                persist_directory=str(CHROMA_DIR),
                collection_name=COLLECTION_NAME,
            )
            meta_path.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "embedding_model": EMBEDDING_MODEL,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        _vectorstore = store
        return _vectorstore


def _get_llm():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.2,
        google_api_key=api_key,
    )


def _get_session_history(session_id):
    return _session_store.setdefault(session_id, InMemoryChatMessageHistory())


def _get_chat_chain():
    global _chat_chain
    if _chat_chain is not None:
        return _chat_chain

    with _chain_lock:
        if _chat_chain is not None:
            return _chat_chain

        llm = _get_llm()
        vectorstore = _load_or_build_vectorstore()
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant for a financial policy document. "
                    "Answer questions and include the exact page & paragraph "
                    "citations.\n\nContext:\n{context}",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        qa_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(vectorstore.as_retriever(), qa_chain)
        _chat_chain = RunnableWithMessageHistory(
            rag_chain,
            _get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
    return _chat_chain


def answer_question(message, session_id="default"):
    if not message or not message.strip():
        return "Please enter a question about the financial policy document."

    if not os.getenv("GEMINI_API_KEY"):
        return MISSING_KEY_MESSAGE

    try:
        chain = _get_chat_chain()
    except Exception as exc:
        logger.exception("Failed to prepare the document index or chat chain")
        return f"{INDEX_ERROR_MESSAGE}\n\nDetails: `{type(exc).__name__}: {exc}`"

    try:
        response = chain.invoke(
            {"input": message},
            {"configurable": {"session_id": session_id}},
        )
    except Exception as exc:
        logger.exception("Gemini request failed")
        return f"{GENERATION_ERROR_MESSAGE}\n\nDetails: `{type(exc).__name__}: {exc}`"

    return response["answer"]


def chat_with_bot(message, history, request: gr.Request = None):
    session_id = getattr(request, "session_hash", None) or "default"
    return answer_question(message, session_id=session_id)


def _startup_badge():
    if os.getenv("GEMINI_API_KEY"):
        return (
            "✅ `GEMINI_API_KEY` is set. Answers are generated by Gemini; "
            "document search runs locally."
        )
    return (
        "⚠️ `GEMINI_API_KEY` is not set. The app loads and the local document "
        "index works, but answers stay disabled until you add the key as a "
        "Space secret or in a local `.env` file."
    )


demo = gr.ChatInterface(
    fn=chat_with_bot,
    type="messages",
    title="Financial Policy Chatbot",
    description=(
        "Ask any question about the **Financial Policy Document**.\n"
        "Responses include exact page & paragraph citations.\n\n"
        + _startup_badge()
    ),
    theme="glass",
    examples=[
        "What are the Principles of Responsible Financial Management?",
        "List the short-term and long-term financial objectives",
        "When must 90% of superannuation liabilities be fully funded?",
    ],
)

demo.queue(default_concurrency_limit=2)

if __name__ == "__main__":
    demo.launch()
