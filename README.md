# Upwork Technical Support AI Bot (RAG) & Physics Sandbox

A premium, reality-grounded AI RAG (Retrieval-Augmented Generation) assistant developed as part of the evaluation for the **Associate AI Developer** role at **ProAnalyst**.

This system implements a high-fidelity document search and question-answering pipeline using Upwork's technical reference materials. It is designed to run locally, query a persistent vector store, analyze query latencies, enforce strict hallucination limits, and toggle into an experimental physics sandbox environment.

---

## 🛠️ Tech Stack & Key Libraries

*   **Core Language**: Python 3.11+
*   **Vector Database**: ChromaDB
*   **Embeddings Function**: Local `sentence-transformers/all-MiniLM-L6-v2`
*   **LLM Integration**: DeepInfra hosted OpenAI-compatible endpoint (running `Meta-Llama-3.1-8B-Instruct-Turbo`)
*   **RAG Orchestration**: LangChain
*   **Web User Interface**: Streamlit (featuring dark-mode glassmorphism styling)
*   **Environment Configuration**: python-dotenv

---

## 📂 Project Structure

```
upwork_rag_bot/
│
├── data/                             # Raw source documents (PDFs, docs)
│   ├── API Documentation Partial.pdf  # Source API reference PDF
│   └── Associate AI Developer...docx # Spec assignment instructions
│
├── chroma_db/                        # Local persistent vector database (Git-ignored)
├── upwork_api_reference.txt          # Sanitized plain text extracted from PDF
│
├── .env                              # Active environment variables (Git-ignored)
├── .env.example                      # Configuration template
├── .gitignore                        # Standard configuration ignoring secrets & builds
├── requirements.txt                  # Pinned library dependencies
│
├── ingest.py                         # Step A1: Ingestion & sanity checker
├── chunking.py                       # Step A2: Document splitter & overlap highlighter
├── vector_store.py                   # Step A3: Embeddings generation & persistence
├── retrieval.py                      # Step B1: Similarity search evaluator (k=3)
├── llm_chain.py                      # Step B2: LLM generation & Hallucination Guard
├── experimental_modes.py             # Feature: Speculative Physics Sandbox Mode
├── app.py                            # Step B3: Streamlit UI coordination
└── Technical_Summary.md              # Technical assessment details & results
```

---

## ⚙️ Installation & Setup

### 1. Clone & Initialize Environment
Clone the project, set up your Python virtual environment, and activate it:
```bash
python -m venv .venv
# On Windows (cmd):
.venv\Scripts\activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
Install all pinned libraries:
```bash
pip install -r requirements.txt
```

### 3. Configure API Credentials
Copy the environment template and enter your API keys:
```bash
cp .env.example .env
```
Inside `.env`, configure your credentials:
```env
DEEPINFRA_API_KEY=your_actual_key_here
DEEPINFRA_API_BASE=https://api.deepinfra.com/v1/openai
DEEPINFRA_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
DOCUMENT_PATH=upwork_api_reference.txt
```

---

## 🚀 Execution Guide

You can run the RAG pipeline end-to-end or inspect intermediate steps:

### Step 1: Data Ingestion & Sanity Check
Loads the local document, computes counts, and checks console character encoding:
```bash
python ingest.py
```

### Step 2: Document Chunking Analysis
Verifies the recursive character split parameters (size: 500, overlap: 50) and visualizes boundary overlaps:
```bash
python chunking.py
```

### Step 3: Embeddings & Vector Storage
Generates vector indices using the local model and creates the local database:
```bash
python vector_store.py
```

### Step 4: Semantic Retrieval Test
Performs similarity query matching ($k=3$) against the three evaluation questions:
```bash
python retrieval.py
```

### Step 5: Full RAG Pipeline Evaluation
Sends queries along with context to the LLM and executes the Hallucination Guard constraints:
```bash
python llm_chain.py
```

### Step 6: Launch Web Interface
Launches the premium Streamlit dark-mode application locally:
```bash
streamlit run app.py
```

---

## 💡 System Features

### 1. Strict Hallucination Guard
If a query's solution does not exist inside the provided document (e.g., questions regarding rate limits or other APIs), the prompt design forces the LLM to output exactly:
> `"I'm sorry, but the provided documentation does not contain that information."`

### 2. Dual Latency Tracking
Streams timing clocks before and after network request delivery, displaying both the **LLM API Latency** and the **Total RAG Pipeline Latency** directly in the UI.

### 3. Experimental Physics Sandbox Mode
A sidebar toggle shifts the page layout to the **Antigravity Physics Reference Mode**. This mode evaluates speculative physics queries (e.g. anti-gravity, spacetime manipulation) using a reality-grounded behavioral prompt that acknowledges curiosity but gently pivots to General Relativity constraints.
