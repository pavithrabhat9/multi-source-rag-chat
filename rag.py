"""RAG core: LLM setup, retriever (+ optional reranker), prompts and the chat engine."""
import os
from functools import lru_cache

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters

import config
from ingest import get_embed_model, get_vector_store, list_sources

# ---------------------------------------------------------------------------
# Prompts. The strict "ONLY from context" rule is what makes this a grounded RAG bot.
# ---------------------------------------------------------------------------
CONTEXT_PROMPT = (
    "You are a careful assistant that answers questions using ONLY the context below.\n"
    "Rules:\n"
    "- Use only the information in the context. Do not use outside knowledge.\n"
    f"- If the answer is not in the context, reply exactly: \"{config.NOT_FOUND_MESSAGE}\".\n"
    "- Keep answers clear and concise.\n\n"
    "Context:\n{context_str}\n"
)

# Turns a follow-up like "What is its price?" into a standalone question.
CONDENSE_PROMPT = (
    "Given the conversation and a follow-up message, rewrite the follow-up as ONE standalone "
    "question that can be understood without the conversation. Replace pronouns with the names "
    "they refer to. If it is already standalone, return it unchanged. Output only the question.\n\n"
    "Conversation:\n{chat_history}\n\n"
    "Follow-up message: {question}\n\n"
    "Standalone question:"
)


class RecordingChatEngine(CondensePlusContextChatEngine):
    """Same engine, but remembers the rewritten question so the UI can display it."""

    last_condensed_question: str = ""

    def _condense_question(self, chat_history, latest_message):
        question = super()._condense_question(chat_history, latest_message)
        self.last_condensed_question = question
        return question


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ConfigError(Exception):
    """Friendly configuration problem (e.g. missing API key)."""


@lru_cache(maxsize=1)
def get_llm():
    provider = config.LLM_PROVIDER
    if provider == "anthropic":
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigError("ANTHROPIC_API_KEY is missing. Add it to your .env file.")
        from llama_index.llms.anthropic import Anthropic

        llm = Anthropic(model=config.ANTHROPIC_MODEL, api_key=key, temperature=0.0, max_tokens=1024)
    elif provider == "groq":
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise ConfigError("GROQ_API_KEY is missing. Add it to your .env file.")
        from llama_index.llms.groq import Groq

        llm = Groq(model=config.GROQ_MODEL, api_key=key, temperature=0.0)
    else:
        raise ConfigError(f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic' or 'groq'.")
    Settings.llm = llm
    return llm


@lru_cache(maxsize=1)
def _load_reranker():
    # Cross-encoder: reads (question, chunk) together, so it judges relevance better than
    # the embedding similarity - but it is slower, which is why we only run it on a short list.
    return SentenceTransformerRerank(model=config.RERANK_MODEL_NAME, top_n=config.TOP_K)


def get_reranker(top_n: int):
    reranker = _load_reranker()  # model is loaded once...
    reranker.top_n = top_n       # ...but how many chunks to keep can change per question
    return reranker


def new_memory() -> ChatMemoryBuffer:
    return ChatMemoryBuffer.from_defaults(token_limit=3000)


# ---------------------------------------------------------------------------
# Building the engine and asking questions
# ---------------------------------------------------------------------------
def build_chat_engine(top_k: int, source: str | None, use_reranker: bool, memory) -> RecordingChatEngine:
    index = VectorStoreIndex.from_vector_store(get_vector_store(), embed_model=get_embed_model())

    # Qdrant payload filter: "Search only in this source".
    filters = None
    if source:
        filters = MetadataFilters(filters=[MetadataFilter(key="source_name", value=source, operator=FilterOperator.EQ)])

    # With reranking: fetch more candidates, then let the cross-encoder keep the best top_k.
    fetch_k = top_k * config.RERANK_FETCH_MULTIPLIER if use_reranker else top_k
    retriever = index.as_retriever(similarity_top_k=fetch_k, filters=filters)
    postprocessors = [get_reranker(top_k)] if use_reranker else []

    return RecordingChatEngine.from_defaults(
        retriever=retriever,
        llm=get_llm(),
        memory=memory,
        node_postprocessors=postprocessors,
        context_prompt=CONTEXT_PROMPT,
        condense_prompt=CONDENSE_PROMPT,
    )


def format_citation(meta: dict) -> str:
    """[report.pdf, page 3] / [notes.txt] / [example.com/page]"""
    name = meta.get("source_name", "unknown")
    page = meta.get("page")
    return f"[{name}, page {page}]" if page else f"[{name}]"


def ask(question: str, top_k: int = config.TOP_K, source: str | None = None,
        use_reranker: bool = False, memory=None) -> dict:
    """Answer one question. Pass the same `memory` across calls to get follow-up behaviour.
    Returns: answer, citations, condensed_question, nodes (retrieved chunks with scores)."""
    if not list_sources():
        raise ConfigError("No sources indexed yet. Add a file, URL or YouTube link first.")

    engine = build_chat_engine(int(top_k), source, use_reranker, memory if memory is not None else new_memory())
    response = engine.chat(question)

    answer = str(response).strip()
    nodes = response.source_nodes
    citations: list[str] = []
    if config.NOT_FOUND_MESSAGE.lower() not in answer.lower():
        for n in nodes:  # unique, keep order
            c = format_citation(n.node.metadata)
            if c not in citations:
                citations.append(c)

    return {
        "answer": answer,
        "citations": citations,
        "condensed_question": engine.last_condensed_question or question,
        "nodes": nodes,
    }
