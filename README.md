# 📚 Multi-Source RAG Chat

A beginner-friendly **Retrieval-Augmented Generation (RAG)** chat app built with **LlamaIndex**, **Qdrant** and **Gradio**. Ask questions about your PDFs, web pages, YouTube videos and text files, and get answers with citations such as `[report.pdf, page 3]`.

The code is intentionally small, readable and heavily commented, so you can learn how each RAG step works.

---

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [How RAG Works (Step by Step)](#how-rag-works-step-by-step)
- [Settings Explained](#settings-explained)
- [Why Local Embeddings?](#why-local-embeddings)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Many sources, one chat** – PDF, TXT, website URL and YouTube transcript.
- **Citations on every answer** – file name, page number or URL.
- **Follow-up questions** – the chat history is used to rewrite "What is its price?" into a full standalone question (shown in the UI).
- **Source filter** – "Search only in this source" using Qdrant payload filters.
- **Transparent retrieval** – see the retrieved chunks and their similarity scores.
- **Optional reranking** – local cross-encoder (`BAAI/bge-reranker-base`) so you can compare answers with and without it.
- **Grounded answers** – the model answers *only* from the context, otherwise it says *"I could not find this in the sources"*.
- **No duplicates** – uploading the same source twice replaces the old chunks.
- **Friendly errors** – bad URLs, missing transcripts and empty PDFs show clear messages.
- **Evaluation script** – 10 test questions, scored with the reranker on and off.

## Tech Stack

| Part | Choice |
|---|---|
| Language | Python 3.11 / 3.12 |
| Framework | LlamaIndex |
| Vector DB | Qdrant (local, path-based, no Docker) |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, free) |
| Reranker | `BAAI/bge-reranker-base` (local, optional) |
| LLM | Groq (`openai/gpt-oss-120b`) or Anthropic Claude Sonnet, switchable in `.env` |
| UI | Gradio |

## Project Structure

```
multi-source-rag-chat/
├── app.py                # Gradio UI
├── config.py             # Chunk size, overlap, top-k, model names
├── ingest.py             # Loaders, chunking, storing in Qdrant, list/delete sources
├── rag.py                # Retriever, reranker, prompts, chat engine
├── eval.py               # Evaluation script
├── eval_questions.json   # 10 test questions with expected source/answer
├── data/                 # Sample text files
├── qdrant_data/          # Vector database (auto-created, git-ignored)
├── .env.example          # Template for your API keys
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

- **Python 3.11 or 3.12** (Python 3.13+ may fail to install some dependencies on Windows).
- A free API key from [Groq](https://console.groq.com) **or** an [Anthropic](https://console.anthropic.com) key.

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/pavithrabhat9/multi-source-rag-chat.git
cd multi-source-rag-chat

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux

# 3. Install dependencies
python -m pip install -r requirements.txt
```

### Configuration

Copy the example file and add your key:

```bash
copy .env.example .env            # Windows  (macOS/Linux: cp .env.example .env)
```

```env
LLM_PROVIDER=groq                 # "groq" or "anthropic"
GROQ_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here
```

> API keys are never hardcoded. `.env` is git-ignored, so your keys stay private.

### Run

```bash
python app.py
```

Open the printed address (usually http://127.0.0.1:7860).

The first run downloads the embedding model (~130 MB). The reranker (~1 GB) downloads the first time you tick **Rerank**.

## Usage

1. **Add sources** – upload PDF/TXT files, paste a website URL, or paste a YouTube link, then click the index button.
2. **Ask a question** – e.g. index the files in `data/` and ask *"How much does the Lumora Lamp X2 cost?"*
3. **Ask a follow-up** – e.g. *"How long does its battery last?"*. The rewritten standalone question appears under the chat.
4. **Inspect retrieval** – open **Retrieved chunks** to see what the model saw, with scores.
5. **Tune it** – change top-k, pick a single source in the dropdown, or tick **Rerank** and compare answers.
6. **Manage sources** – the table lists indexed sources; choose one and click **Delete** to remove it.

## How RAG Works (Step by Step)

```
 ┌──────┐   ┌───────┐   ┌───────┐   ┌───────┐
 │ Load │──▶│ Split │──▶│ Embed │──▶│ Store │      (ingest.py – once per source)
 └──────┘   └───────┘   └───────┘   └───────┘
                                         │
 Question ─▶ Rewrite ─▶ Embed ─▶ Retrieve top-k ─▶ (Rerank) ─▶ Generate ─▶ Answer + citations
             (follow-up)         (Qdrant search)   (optional)    (LLM)            (rag.py)
```

1. **Load** – read text from a PDF (per page), TXT, web page or YouTube transcript. Each piece gets metadata: `source_name`, `source_type`, `page`, `url`.
2. **Split** – long text is cut into chunks with `SentenceSplitter`, because the LLM can't read everything and small chunks are easier to match.
3. **Embed** – each chunk becomes a vector (a list of numbers) that captures its meaning. Similar meaning means nearby vectors.
4. **Store** – vectors, text and metadata are saved in Qdrant (the `qdrant_data/` folder, no server needed).
5. **Rewrite** – a follow-up such as *"What is its price?"* becomes *"What is the price of the Lumora Lamp X2?"* using the chat history.
6. **Retrieve** – the question is embedded and Qdrant returns the `top-k` most similar chunks (optionally from one source only).
7. **Rerank (optional)** – a cross-encoder re-scores the candidates more carefully and keeps the best ones.
8. **Generate** – the LLM receives the chunks and a strict prompt: *answer only from the context, otherwise say "I could not find this in the sources"*.
9. **Cite** – citations are built from the metadata of the chunks that were used.

## Settings Explained

All settings live in [`config.py`](config.py).

| Setting | Default | What it does |
|---|---|---|
| `CHUNK_SIZE` | 512 | Max tokens per chunk. Bigger = more context but less precise search; smaller = precise but may lose context. |
| `CHUNK_OVERLAP` | 80 | Tokens shared between neighbouring chunks so an idea on a boundary isn't cut in half. |
| `TOP_K` | 4 | Chunks sent to the LLM. Too low: the answer may be missing. Too high: noise and cost. Adjustable in the UI. |
| Reranking | off | Vector search is fast but rough. A cross-encoder reads *question + chunk together* for better relevance, but is slower. We fetch `3 × top-k` candidates, then keep the best `top-k`. |

## Why Local Embeddings?

`BAAI/bge-small-en-v1.5` runs on your own machine: **free, private, no API key**, and good quality for English.

**To swap it:**
- Another local model: change `EMBED_MODEL_NAME` in `config.py` (e.g. `BAAI/bge-base-en-v1.5`).
- A hosted model (e.g. OpenAI): replace `HuggingFaceEmbedding` in `get_embed_model()` in `ingest.py`.

> ⚠️ After changing the embedding model, **delete `qdrant_data/`** and re-index. Vectors from different models can't be mixed.

## Evaluation

```bash
python eval.py
```

Close the app first: Qdrant local mode allows only one process at a time.

For each of the 10 questions in `eval_questions.json`, the script prints whether the **correct source was retrieved** and whether the **answer matches** (a simple keyword check, not an AI judge). It then prints a final score such as `8/10`, once with the reranker off and once with it on.

## Troubleshooting

| Problem | Fix |
|---|---|
| `pip` is not recognized | Use `python -m pip ...` instead. |
| Dependencies fail to install | Use Python 3.11 or 3.12 in a fresh virtual environment. |
| "API key is missing" | Check `.env` and restart the app after editing it. |
| Groq "model does not exist" | Set `GROQ_MODEL` in `.env` to a model your key can use (list them at `https://api.groq.com/openai/v1/models`). |
| Scanned PDF is rejected | It contains only images and no text. Use an OCR tool first. |
| Database locked error | Only one process may use `qdrant_data/`. Close the app before running `eval.py`. |

## License

This project is made for learning LlamaIndex and RAG concepts, for personal learning and experimentation.
