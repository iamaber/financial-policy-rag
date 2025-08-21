from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import PyPDF2
import os
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = "./data/For Task - Policy file.pdf"
CHROMA_DIR = "./data/chroma_db"


def extract_text(pdf_path):
    text = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, 1):
                text.append({"page": i, "content": page.extract_text()})
    except Exception as e:
        print(f"Error reading PDF file: {e}")
        return []
    return text


# Extract text from PDF
raw_pages = extract_text(DATA_DIR)

# Initialize text splitter with semantic chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=300, separators=["\n\n", "\n", ".", "!", "?"]
)

# Create documents with metadata
docs = []
for p in raw_pages:
    chunks = splitter.split_text(p["content"])
    for para_idx, c in enumerate(chunks, 1):
        docs.append(
            Document(
                page_content=c, metadata={"page": p["page"], "paragraph": para_idx}
            )
        )

# Initialize embeddings
try:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")
    )
except Exception as e:
    print(f"Error initializing embeddings: {e}")
    exit(1)

# Create vector store
try:
    vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_DIR)
except Exception as e:
    print(f"Error creating vector store: {e}")
    exit(1)

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit(1)

# Create prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for a financial policy document. "
            "Answer questions and include the exact page & paragraph citations.\n\nContext:\n{context}",
        ),
        ("human", "{input}"),
    ]
)

# Create RAG chain
qa_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
rag_chain = create_retrieval_chain(retriever, qa_chain)

# In-memory session store
store = {}

# Create chatbot with history
chat_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda sid: store.setdefault(sid, ChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


def chat_with_bot(message: str, history) -> str:
    try:
        response = chat_with_history.invoke(
            {"input": message}, {"configurable": {"session_id": "default"}}
        )
        return response["answer"]
    except Exception as e:
        return f"Error generating response: {e}"


# Gradio UI
demo = gr.ChatInterface(
    fn=chat_with_bot,
    title="Financial Policy Chatbot",
    description=(
        "Ask any question about the **Financial Policy Document**.\n"
        "Responses include exact page & paragraph citations."
    ),
    theme="glass",
    examples=[
        "What are the Principles of Responsible Financial Management?",
        "List the short-term and long-term financial objectives",
        "When must 90% of superannuation liabilities be fully funded?",
    ],
)

# Launch the application
if __name__ == "__main__":
    os.makedirs(os.path.dirname(CHROMA_DIR), exist_ok=True)
    demo.launch()
