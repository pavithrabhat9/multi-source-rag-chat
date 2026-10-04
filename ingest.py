"""Ingestion: load -> split -> embed -> store in Qdrant.

Every public ingest_* function returns a friendly status string,
and raises IngestError with a friendly message when something goes wrong.
"""
import os
import re
import uuid
from functools import lru_cache
from urllib.parse import parse_qs, urlparse

import requests
from bs4 import BeautifulSoup
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pypdf import PdfReader
from qdrant_client import QdrantClient, models

import config


class IngestError(Exception):
    """Raised with a message that is safe to show directly to the user."""


# ---------------------------------------------------------------------------
# Shared resources (created once, reused everywhere)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embed_model() -> HuggingFaceEmbedding:
    """Local embedding model. Downloaded on first use, then cached on disk."""
    model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME)
    Settings.embed_model = model
    return model


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    """Qdrant in local (path) mode. Only ONE process may open this folder at a time,
    so we keep a single client for the whole app."""
    return QdrantClient(path=config.QDRANT_PATH)


def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore(client=get_client(), collection_name=config.COLLECTION_NAME)


def _collection_exists() -> bool:
    return get_client().collection_exists(config.COLLECTION_NAME)


def _source_filter(source_name: str) -> models.Filter:
    # LlamaIndex stores metadata keys as top-level Qdrant payload fields.
    return models.Filter(
        must=[models.FieldCondition(key="source_name", match=models.MatchValue(value=source_name))]
    )


# ---------------------------------------------------------------------------
# Loaders: each returns (source_name, list[Document])
# ---------------------------------------------------------------------------
def _load_pdf(path: str):
    try:
        reader = PdfReader(path)
        pages = [(i + 1, (p.extract_text() or "").strip()) for i, p in enumerate(reader.pages)]
    except Exception as e:
        raise IngestError(f"Could not read the PDF '{os.path.basename(path)}': {e}")
    name = os.path.basename(path)
    # One Document per page so every chunk knows its page number (needed for citations).
    docs = [
        Document(text=text, metadata={"source_name": name, "source_type": "pdf", "page": num})
        for num, text in pages
        if text
    ]
    if not docs:
        raise IngestError(f"'{name}' has no extractable text (it may be a scanned image PDF).")
    return name, docs


def _load_txt(path: str):
    name = os.path.basename(path)
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()
    except OSError as e:
        raise IngestError(f"Could not read '{name}': {e}")
    if not text:
        raise IngestError(f"'{name}' is empty.")
    return name, [Document(text=text, metadata={"source_name": name, "source_type": "txt"})]


def _load_url(url: str):
    url = url.strip()
    if not urlparse(url).scheme:
        url = "https://" + url
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0 (RAG learning app)"})
        resp.raise_for_status()
    except requests.RequestException as e:
        raise IngestError(f"Could not load that URL. Check the address and try again. ({e.__class__.__name__})")

    soup = BeautifulSoup(resp.text, "html.parser")
    # Remove non-content tags so menus/scripts don't pollute the index.
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = re.sub(r"\n\s*\n+", "\n\n", soup.get_text("\n")).strip()
    if len(text) < 50:
        raise IngestError("That page has almost no readable text (it may need JavaScript to load).")

    parsed = urlparse(url)
    name = (parsed.netloc + parsed.path).rstrip("/")  # URL doubles as a unique, readable name
    return name, [Document(text=text, metadata={"source_name": name, "source_type": "web", "url": url})]


def _youtube_video_id(url: str) -> str | None:
    p = urlparse(url.strip())
    if p.netloc.endswith("youtu.be"):
        return p.path.lstrip("/") or None
    if "youtube.com" in p.netloc:
        if p.path == "/watch":
            return parse_qs(p.query).get("v", [None])[0]
        m = re.match(r"^/(embed|shorts|live)/([\w-]+)", p.path)
        if m:
            return m.group(2)
    return None


def _load_youtube(url: str):
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = _youtube_video_id(url)
    if not video_id:
        raise IngestError("That does not look like a YouTube video link.")
    try:
        try:  # newer versions of youtube-transcript-api (>=1.0)
            snippets = YouTubeTranscriptApi().fetch(video_id)
            text = " ".join(s.text for s in snippets)
        except AttributeError:  # older versions
            text = " ".join(s["text"] for s in YouTubeTranscriptApi.get_transcript(video_id))
    except Exception as e:
        raise IngestError(
            f"Could not get a transcript for this video (captions may be disabled). ({e.__class__.__name__})"
        )
    text = text.strip()
    if not text:
        raise IngestError("The transcript for this video is empty.")
    name = f"YouTube {video_id}"
    full_url = f"https://www.youtube.com/watch?v={video_id}"
    return name, [Document(text=text, metadata={"source_name": name, "source_type": "youtube", "url": full_url})]


# ---------------------------------------------------------------------------
# Split + embed + store
# ---------------------------------------------------------------------------
def _index_documents(source_name: str, docs: list[Document]) -> str:
    # Keep embeddings focused on the text only (metadata would add noise to the vectors).
    for d in docs:
        d.excluded_embed_metadata_keys = list(d.metadata.keys())

    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(docs)
    if not nodes:
        raise IngestError("Nothing to index after splitting the text.")

    # Duplicate protection, two layers:
    # 1) delete any old chunks of this source (so re-uploading an edited file replaces it)
    # 2) give each chunk a deterministic ID, so identical chunks can never be stored twice
    delete_source(source_name)
    for i, node in enumerate(nodes):
        node.id_ = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_name}|{i}|{node.get_content()}"))

    storage = StorageContext.from_defaults(vector_store=get_vector_store())
    VectorStoreIndex(nodes, storage_context=storage, embed_model=get_embed_model())
    return f"✅ Indexed '{source_name}' ({len(nodes)} chunks)."


def ingest_file(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        name, docs = _load_pdf(path)
    elif ext == ".txt":
        name, docs = _load_txt(path)
    else:
        raise IngestError(f"Unsupported file type '{ext}'. Please upload a PDF or TXT file.")
    return _index_documents(name, docs)


def ingest_url(url: str) -> str:
    if not url or not url.strip():
        raise IngestError("Please paste a URL first.")
    return _index_documents(*_load_url(url))


def ingest_youtube(url: str) -> str:
    if not url or not url.strip():
        raise IngestError("Please paste a YouTube link first.")
    return _index_documents(*_load_youtube(url))


# ---------------------------------------------------------------------------
# Managing sources
# ---------------------------------------------------------------------------
def list_sources() -> list[dict]:
    """Return one row per source with its chunk count, by scanning the stored payloads."""
    if not _collection_exists():
        return []
    sources: dict[str, dict] = {}
    offset = None
    while True:
        points, offset = get_client().scroll(
            config.COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=["source_name", "source_type", "url"],
            with_vectors=False,
        )
        for p in points:
            pl = p.payload or {}
            name = pl.get("source_name", "unknown")
            row = sources.setdefault(
                name, {"source_name": name, "source_type": pl.get("source_type", ""), "url": pl.get("url", ""), "chunks": 0}
            )
            row["chunks"] += 1
        if offset is None:
            break
    return sorted(sources.values(), key=lambda r: r["source_name"])


def delete_source(source_name: str) -> None:
    """Remove every chunk that belongs to a source."""
    if not _collection_exists():
        return
    get_client().delete(
        config.COLLECTION_NAME,
        points_selector=models.FilterSelector(filter=_source_filter(source_name)),
    )
