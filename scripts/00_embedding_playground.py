"""Embedding playground: see embeddings with your own eyes.

Run it:            uv run python scripts/00_embedding_playground.py
First run:         downloads the BGE-small model (~130MB) from Hugging Face.

What this shows
---------------
An embedding model turns a sentence into a list of numbers (a vector) such
that sentences with similar MEANING end up with similar vectors — even when
they share no words. "Similar" is measured with cosine similarity: the cosine
of the angle between two vectors (1.0 = same direction, 0 = unrelated).

Everything this pipeline does with "semantic search" reduces to:
embed the query, embed the chunks, find the chunks whose vectors point
in nearly the same direction as the query's.

"""

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# Deliberately chosen sentences — some paraphrase pairs, an exact-identifier
# sentence, and one that has nothing to do with anything:
SENTENCES = [
    "How do I return JSON from an API endpoint?",           # 0  ─┐ paraphrases:
    "Returning a JSON response from a route handler",       # 1  ─┘ no shared keywords needed
    "What is the default port for a FastAPI dev server?",   # 2  ─┐ same topic area,
    "Uvicorn serves the app on port 8000 unless changed",   # 3  ─┘ different phrasing
    "Set response_model_exclude_none=True on the decorator",# 4  exact identifier — remember this one
    "Removing null fields from the endpoint's output",      # 5  what #4 MEANS, in plain words
    "My cat sleeps on the keyboard all afternoon",          # 6  unrelated on purpose
    "Chunk overlap keeps sentences intact across boundaries",# 7  RAG vocab
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """cos(theta) = (a . b) / (|a| * |b|) — this one line is the heart of dense retrieval."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    console.print("[bold]Loading embedding model (first run downloads ~130MB)...[/bold]")
    from sentence_transformers import SentenceTransformer  # imported late: slow import

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    vectors = model.encode(SENTENCES, normalize_embeddings=True)
    console.print(
        f"\nEach sentence became a vector of [bold]{vectors.shape[1]}[/bold] numbers. "
        f"The first 6 numbers of sentence 0:\n  {np.round(vectors[0][:6], 4).tolist()} ...\n"
    )

    # Pairwise cosine similarity matrix
    n = len(SENTENCES)
    sims = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sims[i, j] = cosine_similarity(vectors[i], vectors[j])

    table = Table(title="Cosine similarity (1.00 = identical meaning direction)")
    table.add_column("")
    for j in range(n):
        table.add_column(f"s{j}", justify="right")
    for i in range(n):
        table.add_row(f"s{i}", *[f"{sims[i, j]:.2f}" for j in range(n)])
    console.print(table)

    for i, s in enumerate(SENTENCES):
        console.print(f"  [dim]s{i}[/dim] {s}")

    # Most / least similar distinct pairs
    pairs = [(sims[i, j], i, j) for i in range(n) for j in range(i + 1, n)]
    pairs.sort(reverse=True)
    console.print("\n[bold]Top 3 most similar pairs:[/bold]")
    for score, i, j in pairs[:3]:
        console.print(f"  {score:.3f}  s{i} <-> s{j}")
    score, i, j = pairs[-1]
    console.print(f"[bold]Least similar pair:[/bold]\n  {score:.3f}  s{i} <-> s{j}")

    # Console-safe ASCII only below (Windows terminals may not render em-dashes)
    console.print(
        "\n[bold]What to notice (think before reading on):[/bold]\n"
        "  1. s0/s1 score high with almost no shared words -- that's semantics, not keywords.\n"
        "  2. s6 (the cat) scores low against everything -- 'unrelated' is measurable.\n"
        "  3. Look at s4 vs s5: the identifier sentence and its plain-English meaning.\n"
        "     Decent -- but now imagine QUERYING for the exact string\n"
        "     'response_model_exclude_none' in a 5,000-chunk corpus. Embeddings blur\n"
        "     exact identifiers into their general neighborhood. THIS GAP is why the\n"
        "     project has BM25 keyword search alongside embeddings.\n"
        "  4. Note the similarity floor: even unrelated pairs sit well above 0.\n"
        "     Absolute values mean little; RANKINGS are what retrieval uses.\n"
        "\n[bold]Try:[/bold] add 2-3 of your own sentences to SENTENCES and predict the\n"
        "scores before re-running. Being wrong here is the fastest way to build intuition."
    )


if __name__ == "__main__":
    main()
