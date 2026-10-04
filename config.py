"""Central settings. Change values here to experiment - no other file needs editing."""
import os

from dotenv import load_dotenv

# Read API keys from a local .env file (never hardcode keys in source code).
load_dotenv()

# ---------- Chunking ----------
CHUNK_SIZE = 512      # max tokens per chunk. Bigger = more context per chunk, less precise search
CHUNK_OVERLAP = 80    # tokens shared between neighbouring chunks so ideas aren't cut in half

# ---------- Retrieval ----------
TOP_K = 4             # default number of chunks sent to the LLM (adjustable in the UI)
RERANK_FETCH_MULTIPLIER = 3  # with reranking we first fetch TOP_K * 3 chunks, then keep the best TOP_K

# ---------- Models ----------
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # local, free, 384-dim vectors
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"  # local cross-encoder

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()  # "anthropic" or "groq"
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")  # Llama 3.3 70B is no longer offered on Groq

# ---------- Storage ----------
QDRANT_PATH = "./qdrant_data"      # local folder; created automatically, persists between runs
COLLECTION_NAME = "multi_source_rag"

# ---------- Answers ----------
NOT_FOUND_MESSAGE = "I could not find this in the sources"
