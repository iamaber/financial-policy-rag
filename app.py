from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY")
)
vectorstore = Chroma(persist_directory="data/chroma_db", embedding_function=embeddings)


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant for a financial policy document. "
            "Answer concisely and include the exact page & paragraph citations.\n\nContext:\n{context}",
        ),
        ("human", "{input}"),
    ]
)

qa_chain = create_stuff_documents_chain(llm, prompt)
retriever = vectorstore.as_retriever()
rag_chain = create_retrieval_chain(retriever, qa_chain)

# In-memory session store
store = {}

chat_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda sid: store.setdefault(sid, ChatMessageHistory()),
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)


def chat_with_bot(message: str, history) -> str:
    """Single-turn helper so we can plug it into Gradio."""
    response = chat_with_history.invoke(
        {"input": message}, {"configurable": {"session_id": "default"}}
    )
    return response["answer"]


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


# Launch
if __name__ == "__main__":
    demo.launch()
