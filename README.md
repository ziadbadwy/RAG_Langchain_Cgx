# RAG with Qwen3 + LangChain

A Retrieval-Augmented Generation (RAG) chatbot that answers questions from any PDF document using **Qwen3:8b** via Ollama and a **multilingual FAISS** vector store. Built as a teaching session for DEPI students.

## Try It on Google Colab

<a target="_blank" href="https://colab.research.google.com/github/ziadbadwy/RAG_Langchain_Cgx/blob/main/RAG.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

---

## How It Works

```
PDF Document
     │
     ▼
 Text Splitter  ──►  Chunks (1024 tokens, 64 overlap)
     │
     ▼
 Embeddings     ──►  intfloat/multilingual-e5-large-instruct
     │
     ▼
 FAISS Index    ──►  Saved locally (faiss_index/)
     │
  Question
     │
     ▼
 Retrieval      ──►  Top-k chunks by similarity score
     │
     ▼
 Prompt Builder ──►  System + History + Context + Question
     │
     ▼
 Qwen3:8b       ──►  Answer (via Ollama)
```

---

## Project Structure

```
RAG_Langchain_Cgx/
├── Helper.py        # Core RAG pipeline class
├── app.py           # Gradio web interface
├── RAG.ipynb        # Step-by-step Colab notebook
├── questions.csv    # Evaluation questions (Transformer paper)
├── requirements.txt
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ziadbadwy/RAG_Langchain_Cgx
cd RAG_Langchain_Cgx
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install and start Ollama

Download from [ollama.com](https://ollama.com), then:

```bash
ollama serve            # start the server
ollama pull qwen3:8b    # download the model (~5GB)
```

### 4. Run the app

```bash
python app.py
```

A Gradio link will appear in the terminal. Open it in your browser to start chatting.

---

## Helper Class

`Helper.py` contains the full RAG pipeline. Here is a quick reference:

| Method | What it does |
|---|---|
| `__init__(pdf_path, verbose)` | Loads the PDF and sets up conversation memory |
| `splitter()` | Splits the PDF into overlapping chunks |
| `vector_store()` | Creates (or loads) the FAISS index |
| `retrival_with_score()` | Finds the most relevant chunks for a question |
| `build_prompt()` | Assembles system prompt + history + context + question |
| `llm_response(input)` | Full pipeline: retrieve → prompt → generate → remember |
| `eval_pipeline(questions_path)` | Scores the pipeline using a CSV of test questions |

### Basic usage

```python
from Helper import Helper

hp = Helper("your_document.pdf", verbose=True)
answer = hp.llm_response("What is the main idea of this document?")
print(answer)
```

### Verbose mode

Set `verbose=True` to see exactly what happens at each step — great for understanding the RAG pipeline:

```
============================================================
  USER QUESTION
============================================================
What is multi-head attention?

============================================================
  RETRIEVED CHUNKS
============================================================
[Chunk 1] Page: 4 | Score: 0.3821
Multi-head attention allows the model to jointly attend to information...

============================================================
  CONVERSATION HISTORY
============================================================
(no history yet)

============================================================
  FULL PROMPT SENT TO MODEL
============================================================
### System:
You are an advanced AI assistant...

============================================================
  MODEL RESPONSE
============================================================
Multi-head attention is a mechanism that runs several attention...
```

---

## Evaluation

A CSV of test questions is included to measure how well the pipeline answers questions about the **"Attention is All You Need"** paper:

```python
score = hp.eval_pipeline('questions.csv')
print(f'Score: {score:.1f}%')
```

The score is the percentage of answers whose embedding is similar enough to the question embedding (cosine similarity > 0.5), which gives a rough measure of relevance.

---

## Requirements

- Python 3.9+
- 8 GB RAM minimum (16 GB recommended)
- [Ollama](https://ollama.com) installed and running locally
