# 📚 Multi-Source RAG Chat

A beginner-friendly **Retrieval-Augmented Generation (RAG)** chat app. Ask questions about your PDFs, web pages, YouTube videos and text files, and get answers with citations such as `[report.pdf, page 3]`.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Framework-8B5CF6)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C)
![Groq](https://img.shields.io/badge/Groq-LLM-F55036)
![Claude](https://img.shields.io/badge/Anthropic-Claude-D97757)
![Gradio](https://img.shields.io/badge/Gradio-UI-FF7C00)

## Features

- **Many sources, one chat** – PDF, TXT, website URL and YouTube transcript.
- **Citations on every answer** – file name, page number or URL.
- **Follow-up questions** – chat history rewrites "What is its price?" into a full question (shown in the UI).
- **Source filter** – "Search only in this source" using Qdrant payload filters.
- **Transparent retrieval** – see the retrieved chunks and their similarity scores.
- **Optional reranking** – local cross-encoder (`BAAI/bge-reranker-base`) to compare answers with and without it.
- **Grounded answers** – answers come only from your sources, otherwise: *"I could not find this in the sources"*.
- **Local embeddings** – `BAAI/bge-small-en-v1.5`, free and private.
- **Evaluation script** – 10 test questions, scored with the reranker on and off.

## Getting Started

**Requirements:** Python 3.11 or 3.12, and a free [Groq](https://console.groq.com) or [Anthropic](https://console.anthropic.com) API key.

```bash
# 1. Clone
git clone https://github.com/pavithrabhat9/multi-source-rag-chat.git
cd multi-source-rag-chat

# 2. Virtual environment
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)

# 3. Install
python -m pip install -r requirements.txt

# 4. Add your API key
copy .env.example .env            # Windows  (macOS/Linux: cp .env.example .env)
```

Edit `.env`:

```env
LLM_PROVIDER=groq                 # "groq" or "anthropic"
GROQ_API_KEY=your_key_here
```

```bash
# 5. Run, then open http://127.0.0.1:7860
python app.py
```

The first run downloads the embedding model (~130 MB). The reranker (~1 GB) downloads the first time you tick **Rerank**.

## Usage

1. Add sources: upload PDF/TXT files, paste a URL or a YouTube link.
2. Ask a question, e.g. index `data/` and ask *"How much does the Lumora Lamp X2 cost?"*
3. Ask a follow-up, e.g. *"How long does its battery last?"*
4. Open **Retrieved chunks** to see what the model saw. Adjust top-k, pick a single source, or tick **Rerank**.

## How RAG Works

```
 ┌──────┐   ┌───────┐   ┌───────┐   ┌───────┐
 │ Load │──▶│ Split │──▶│ Embed │──▶│ Store │      (ingest.py – once per source)
 └──────┘   └───────┘   └───────┘   └───────┘
                                         │
 Question ─▶ Rewrite ─▶ Embed ─▶ Retrieve top-k ─▶ (Rerank) ─▶ Generate ─▶ Answer + citations
             (follow-up)         (Qdrant search)   (optional)    (LLM)            (rag.py)
```

1. **Load** – read text from PDF (per page), TXT, web page or YouTube. Each piece gets metadata: `source_name`, `source_type`, `page`, `url`.
2. **Split** – text is cut into chunks (`SentenceSplitter`) so small, focused pieces can be matched.
3. **Embed** – each chunk becomes a vector that captures its meaning.
4. **Store** – vectors, text and metadata are saved in Qdrant (`qdrant_data/`, no server needed).
5. **Rewrite** – follow-up questions become standalone questions using chat history.
6. **Retrieve** – Qdrant returns the `top-k` most similar chunks.
7. **Rerank (optional)** – a cross-encoder re-scores candidates more carefully.
8. **Generate** – the LLM answers using only the retrieved chunks.
9. **Cite** – citations are built from the chunk metadata.

## Settings

Change these in [`config.py`](config.py):

- **Chunk size (512)** – bigger chunks keep more context but search less precisely.
- **Overlap (80)** – shared text between chunks so ideas aren't cut in half.
- **Top-k (4)** – how many chunks go to the LLM. Too low may miss the answer; too high adds noise.
- **Reranking** – a slower but more accurate relevance check. We fetch `3 × top-k` candidates and keep the best `top-k`.

To swap the embedding model, change `EMBED_MODEL_NAME` in `config.py`. After changing it, delete `qdrant_data/` and re-index, because vectors from different models can't be mixed.

## Evaluation

Close the app first (Qdrant local mode allows one process at a time), then run:

```bash
python eval.py
```

It checks whether the correct source was retrieved and whether the answer matches for each question in `eval_questions.json`, and prints a score like `8/10` with the reranker off and on.

## License

This project is made for learning LlamaIndex and RAG concepts, for personal learning and experimentation.
