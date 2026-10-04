"""Gradio UI. Run with:  python app.py"""
import gradio as gr

import config
import ingest
import rag

ALL_SOURCES = "All sources"
SESSION_MEMORY = rag.new_memory()  # one shared chat history (fine for a single-user learning app)


# ---------------------------------------------------------------------------
# Source management
# ---------------------------------------------------------------------------
def _refresh():
    """Re-read the source list and update the table + both dropdowns."""
    rows = ingest.list_sources()
    names = [r["source_name"] for r in rows]
    table = [[r["source_name"], r["source_type"], r["chunks"], r["url"]] for r in rows]
    return (
        table,
        gr.update(choices=[ALL_SOURCES] + names, value=ALL_SOURCES),
        gr.update(choices=names, value=None),
    )


def _run_ingest(fn, *args) -> str:
    """Call an ingest function and turn any error into a friendly message."""
    try:
        return fn(*args)
    except ingest.IngestError as e:
        return f"⚠️ {e}"
    except Exception as e:  # last-resort safety net
        return f"⚠️ Unexpected error: {e}"


def add_files(files):
    if not files:
        return ("Please choose at least one PDF or TXT file.", *_refresh())
    msgs = [_run_ingest(ingest.ingest_file, f if isinstance(f, str) else f.name) for f in files]
    return ("\n".join(msgs), *_refresh())


def add_url(url):
    return (_run_ingest(ingest.ingest_url, url), *_refresh())


def add_youtube(url):
    return (_run_ingest(ingest.ingest_youtube, url), *_refresh())


def remove_source(name):
    if not name:
        return ("Pick a source to delete first.", *_refresh())
    ingest.delete_source(name)
    return (f"🗑️ Deleted '{name}'.", *_refresh())


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def chat(message, history, top_k, source, use_reranker):
    history = history or []
    if not message or not message.strip():
        return history, "", "", ""
    history = history + [{"role": "user", "content": message}]
    try:
        result = rag.ask(
            message,
            top_k=top_k,
            source=None if source in (None, ALL_SOURCES) else source,
            use_reranker=use_reranker,
            memory=SESSION_MEMORY,
        )
    except rag.ConfigError as e:
        history.append({"role": "assistant", "content": f"⚠️ {e}"})
        return history, "", "", ""
    except Exception as e:
        history.append({"role": "assistant", "content": f"⚠️ Something went wrong: {e}"})
        return history, "", "", ""

    answer = result["answer"]
    if result["citations"]:
        answer += "\n\n**Sources:** " + " ".join(result["citations"])
    history.append({"role": "assistant", "content": answer})

    # Show exactly what the LLM saw, with similarity (or reranker) scores.
    chunks_md = "\n\n---\n\n".join(
        f"**#{i} · {rag.format_citation(n.node.metadata)} · score {n.score:.3f}**\n\n{n.node.get_content()[:1200]}"
        for i, n in enumerate(result["nodes"], 1)
    ) or "_No chunks retrieved._"
    return history, "", result["condensed_question"], chunks_md


def clear_chat():
    SESSION_MEMORY.reset()
    return [], "", "", ""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Multi-Source RAG Chat") as demo:
    gr.Markdown("# 📚 Multi-Source RAG Chat\nAsk questions about your PDFs, web pages, YouTube videos and text files. Answers come with citations.")

    with gr.Row():
        # ---- Left: sources ----
        with gr.Column(scale=2):
            gr.Markdown("### 1. Add sources")
            with gr.Tab("Files"):
                files = gr.File(label="PDF / TXT", file_count="multiple", file_types=[".pdf", ".txt"])
                btn_files = gr.Button("Index files", variant="primary")
            with gr.Tab("Website"):
                url_in = gr.Textbox(label="Page URL", placeholder="https://example.com/article")
                btn_url = gr.Button("Index page", variant="primary")
            with gr.Tab("YouTube"):
                yt_in = gr.Textbox(label="Video link", placeholder="https://www.youtube.com/watch?v=...")
                btn_yt = gr.Button("Index transcript", variant="primary")
            status = gr.Textbox(label="Status", interactive=False, lines=2)

            gr.Markdown("### Indexed sources")
            table = gr.Dataframe(headers=["Source", "Type", "Chunks", "URL"], interactive=False)
            with gr.Row():
                delete_dd = gr.Dropdown(label="Delete a source", choices=[])
                btn_delete = gr.Button("Delete", variant="stop")

        # ---- Right: chat ----
        with gr.Column(scale=3):
            gr.Markdown("### 2. Ask questions")
            chatbot = gr.Chatbot(height=380, type="messages")
            msg = gr.Textbox(label="Your question", placeholder="Ask something about your sources…")
            with gr.Row():
                btn_send = gr.Button("Send", variant="primary")
                btn_clear = gr.Button("Clear chat")
            with gr.Row():
                top_k = gr.Slider(1, 10, value=config.TOP_K, step=1, label="Top-k chunks")
                filter_dd = gr.Dropdown(label="Search only in this source", choices=[ALL_SOURCES], value=ALL_SOURCES)
                rerank = gr.Checkbox(label="Rerank (cross-encoder)", value=False)
            condensed = gr.Textbox(label="Rewritten standalone question (what was actually searched)", interactive=False)
            with gr.Accordion("Retrieved chunks (what the model saw)", open=False):
                chunks = gr.Markdown()

    refresh_outputs = [table, filter_dd, delete_dd]
    btn_files.click(add_files, files, [status, *refresh_outputs])
    btn_url.click(add_url, url_in, [status, *refresh_outputs])
    btn_yt.click(add_youtube, yt_in, [status, *refresh_outputs])
    btn_delete.click(remove_source, delete_dd, [status, *refresh_outputs])
    demo.load(_refresh, None, refresh_outputs)

    chat_inputs = [msg, chatbot, top_k, filter_dd, rerank]
    chat_outputs = [chatbot, msg, condensed, chunks]
    msg.submit(chat, chat_inputs, chat_outputs)
    btn_send.click(chat, chat_inputs, chat_outputs)
    btn_clear.click(clear_chat, None, [chatbot, msg, condensed, chunks])

if __name__ == "__main__":
    demo.launch()
