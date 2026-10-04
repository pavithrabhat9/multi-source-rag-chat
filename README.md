# 📚 Multi-Source RAG Chat

A small, readable app for **learning RAG** (Retrieval-Augmented Generation). Chat with PDFs, web pages, YouTube transcripts and text files. Every answer shows citations like `[report.pdf, page 3]`, and follow-up questions work.

## How to run

```bash
# 1. (recommended) Python 3.11 or 3.12 and a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)

# 2. install
pip install -r requirements.txt

# 3. add your API key
copy .env.example .env          # then edit .env: set LLM_PROVIDER and the matching key

# 4. run
python app.py                   # open the printed http://127.0.0.1:7860
```

First run downloads the embedding model (~130 MB) once. The reranker (~1 GB) downloads the first time you tick "Rerank".

Try it: index the files in `data/`, then ask *"How much does the Lumora Lamp X2 cost?"* followed by *"How long does its battery last?"*.

Run the evaluation (close the app first): `python eval.py`

## What is RAG? The flow, step by step

```
 ┌──────┐   ┌───────┐   ┌───────┐   ┌───────┐
 │ Load │──▶│ Split │──▶│ Embed │──▶│ Store │      (ingest.py, done once per source)
 └──────┘   └───────┘   └───────┘   └───────┘
                                         │
 Question ─▶ Rewrite ─▶ Embed ─▶ Retrieve top-k ─▶ (Rerank) ─▶ Generate ─▶ Answer + citations
             (follow-up)         (Qdrant search)   (optional)    (LLM)            (rag.py)
```

1. **Load** – read text from a PDF (per page), TXT, web page or YouTube transcript. Each piece gets *metadata*: `source_name`, `source_type`, `page`, `url`.
2. **Split** – long text is cut into chunks (`SentenceSplitter`), because the LLM can't read everything and small chunks are easier to match.
3. **Embed** – each chunk becomes a vector (a list of numbers) capturing its meaning. Similar meaning = nearby vectors.
4. **Store** – vectors + text + metadata are saved in Qdrant (a folder, `qdrant_data/`, no server needed).
5. **Rewrite** – a follow-up like *"What is its price?"* is rewritten into *"What is the price of the Lumora Lamp X2?"* using chat history. The UI shows this rewritten question.
6. **Retrieve** – the question is embedded and Qdrant returns the `top-k` most similar chunks (optionally only from one source, using a payload filter).
7. **Rerank (optional)** – a cross-encoder re-scores the candidates more carefully and keeps the best.
8. **Generate** – the LLM gets the chunks and a strict prompt: *answer only from the context, otherwise say "I could not find this in the sources"*.
9. **Cite** – citations are built from the metadata of the chunks that were used.

## Files

| File | Purpose |
|---|---|
| `app.py` | Gradio UI |
| `config.py` | Chunk size, overlap, top-k, model names |
| `ingest.py` | Loaders, chunking, storing in Qdrant, list/delete sources |
| `rag.py` | Retriever, reranker, prompts, chat engine (`CondensePlusContext`) |
| `eval.py`, `eval_questions.json` | Quality check, run with reranker off/on |
| `data/` | Sample text files |

## Settings explained (all in `config.py`)

- **Chunk size (512 tokens)** – bigger chunks keep more context but make search less precise and cost more tokens; smaller chunks are precise but may lose context.
- **Overlap (80 tokens)** – neighbouring chunks share some text so a sentence on a boundary isn't lost.
- **Top-k (4)** – how many chunks go to the LLM. Too low: the answer may be missing. Too high: noise, cost, and the model gets distracted.
- **Reranking** – vector search is fast but rough. A cross-encoder reads *question + chunk together* and scores relevance much better, but is slower. We fetch `3 × top-k` candidates, then keep the best `top-k`. Toggle it in the UI and compare the answers and the scores in "Retrieved chunks".

Note: after a re-upload of the same file, the old chunks are replaced (and chunk IDs are deterministic), so there are no duplicates.

## Why local embeddings?

`BAAI/bge-small-en-v1.5` runs on your own machine: **free, private, no API key**, and good quality for English. To swap it, change `EMBED_MODEL_NAME` in `config.py` (any sentence-transformers model works, e.g. `BAAI/bge-base-en-v1.5`). To use a hosted model (e.g. OpenAI), replace `HuggingFaceEmbedding` in `get_embed_model()` in `ingest.py`.

> ⚠️ If you change the embedding model, **delete `qdrant_data/`** and re-index: vectors from different models (and sizes) can't be mixed.

## Notes

- Switch LLM with `LLM_PROVIDER=anthropic` or `groq` in `.env`.
- Qdrant local mode allows one process at a time – close the app before running `eval.py`.
- Eval "answer matches" is a simple keyword check, not an AI judge – easy to understand, but strict.
- Scanned PDFs (images) have no text and are rejected with a friendly message.
