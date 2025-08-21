# Financial Policy RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about financial policy documents. The system uses Google's Gemini AI model with vector embeddings to provide accurate, contextual responses with exact page and paragraph citations.

## Live Demo

Try the chatbot live on Hugging Face Spaces: **[https://huggingface.co/spaces/iamaber/financial-policy-rag](https://huggingface.co/spaces/iamaber/financial-policy-rag)**

## Features

- **Document Processing**: Extracts and chunks PDF content using PyPDF2
- **Vector Search**: Uses ChromaDB with Google Gemini embeddings for semantic search
- **Conversational AI**: Powered by Google Gemini 2.0 Flash model
- **Memory**: Maintains conversation history per session
- **Citations**: Provides exact page and paragraph references
- **Web UI**: Interactive Gradio interface
- **Jupyter Notebook**: Development and experimentation environment

## Project Structure

```
.
├── app.py                    # Main Gradio web application
├── config/
│   └── settings.py          # Configuration and environment variables
├── data/
│   ├── For Task - Policy file.pdf  # Source document
│   └── chroma_db/           # Vector database storage
├── notebook/
│   └── chatbot.ipynb        # Development notebook
├── .env                     # Environment variables (create from .env.example)
├── .env.example             # Environment variables template
├── pyproject.toml           # Project dependencies
└── README.md                # This file
```

## Prerequisites

### Install uv (Python Package Manager)

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: Install via pip
pip install uv
```

For more installation options, visit: https://docs.astral.sh/uv/getting-started/installation/

## Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd financial-policy-rag
   ```

2. **Create and activate virtual environment**
   ```bash
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   uv sync
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your Google AI API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```


5. **Place your PDF document**
   
   Put your financial policy PDF file at: `data/For Task - Policy file.pdf`

## Usage

### Web Application

Run the Gradio web interface:

```bash
uv run python app.py
```

The web interface will be available at `http://127.0.0.1:7860`

### Jupyter Notebook

For development and experimentation:

```bash
uv run --with jupyter jupyter lab
```

## How It Works

1. **Document Processing**: PDF content is extracted using PyPDF2 and split into semantic chunks
2. **Embedding**: Text chunks are converted to vector embeddings using Google's embedding model
3. **Storage**: Embeddings are stored in ChromaDB for fast similarity search
4. **Retrieval**: User queries are embedded and matched against stored document chunks
5. **Generation**: Retrieved context is sent to Gemini 2.0 Flash to generate accurate answers
6. **Citations**: Responses include exact page and paragraph references from the source document

## Technical Details

- **LLM**: Google Gemini 2.0 Flash (temperature=0.2 for consistency)
- **Embeddings**: Google `models/embedding-001`
- **Vector Store**: ChromaDB with persistent storage
- **Text Splitting**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
- **Framework**: LangChain for RAG pipeline
- **UI**: Gradio for web interface

## Configuration

Key settings in [`config/settings.py`](config/settings.py):

- `GEMINI_API_KEY`: Your Google AI API key
- Embedding model: `models/embedding-001`
- LLM model: `gemini-2.0-flash`
- Vector DB: `data/chroma_db/`

## Dependencies

Main dependencies (see [`pyproject.toml`](pyproject.toml) for complete list):

- `langchain` - RAG framework
- `langchain-google-genai` - Google AI integration
- `langchain-chroma` - ChromaDB integration
- `gradio` - Web UI
- `PyPDF2` - PDF processing
- `python-dotenv` - Environment variables

## Development

The [`notebook/chatbot.ipynb`](notebook/chatbot.ipynb) contains the complete development workflow:

1. PDF text extraction
2. Document chunking and preprocessing
3. Vector store creation
4. RAG chain setup
5. Interactive testing
