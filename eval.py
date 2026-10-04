"""Tiny evaluation script. Run with:  python eval.py

Close the Gradio app first - Qdrant local mode allows only one process at a time.
Checks, for every question, with the reranker OFF and ON:
  1. Was the expected source retrieved?   (retrieval quality)
  2. Does the answer contain the expected keywords?   (answer quality)
"""
import glob
import json
import sys

import config
import ingest
import rag

# Windows consoles often use cp1252 and crash on emoji in our status messages.
sys.stdout.reconfigure(encoding="utf-8")


def run(questions: list[dict], use_reranker: bool) -> int:
    label = "ON" if use_reranker else "OFF"
    print(f"\n=== Reranker {label} ===")
    score = 0
    for i, q in enumerate(questions, 1):
        # Fresh memory per question so earlier questions can't influence the result.
        result = rag.ask(q["question"], top_k=config.TOP_K, use_reranker=use_reranker, memory=rag.new_memory())

        expected = q.get("expected_source")
        retrieved = {n.node.metadata.get("source_name") for n in result["nodes"]}
        # Questions with no answer in the data have no expected source -> nothing to check.
        source_ok = True if expected is None else expected in retrieved

        answer = result["answer"].lower()
        answer_ok = all(k.lower() in answer for k in q["expected_keywords"])

        score += int(source_ok and answer_ok)
        print(f"{i:>2}. {q['question']}\n    source retrieved: {'yes' if source_ok else 'no'} | "
              f"answer matches: {'yes' if answer_ok else 'no'}")
    print(f"Score ({label}): {score}/{len(questions)}")
    return score


def main():
    # Make sure the sample files are indexed (safe to repeat - duplicates are replaced).
    for path in sorted(glob.glob("data/*.txt")):
        print(ingest.ingest_file(path))

    with open("eval_questions.json", encoding="utf-8") as f:
        questions = json.load(f)

    try:
        off = run(questions, use_reranker=False)
        on = run(questions, use_reranker=True)
    except rag.ConfigError as e:
        print(f"⚠️ {e}")
        return
    print(f"\nSummary: without reranker {off}/{len(questions)} | with reranker {on}/{len(questions)}")


if __name__ == "__main__":
    main()
