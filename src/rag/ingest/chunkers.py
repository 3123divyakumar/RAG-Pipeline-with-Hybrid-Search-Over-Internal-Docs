"""Chunking strategies — split documents into retrieval-sized pieces.

Why chunking exists at all
--------------------------
One embedding must represent ONE idea. Embed a whole 20-page document and every
topic in it gets averaged into mush ("embedding dilution"); embed single
sentences and you lose the context that makes them answerable. Chunking is the
tradeoff dial between those two failure modes.

Three strategies live here so the eval suite can MEASURE which one wins instead of
guessing:

  fixed      dumb sliding window over characters. The baseline every paper
             compares against. Cheap, ignores structure, will happily cut a
             code block in half.
  recursive  structure-aware. Split on the biggest structural boundary first
             (headings), fall back to smaller ones (blank lines, sentences),
             then merge small pieces back up toward the target size. This is
             what LangChain's RecursiveCharacterTextSplitter does — we build
             our own so we own every line of it.
  semantic   split where the TOPIC shifts: embed each sentence and cut where
             consecutive-sentence similarity dips hardest. Highest quality in
             theory, costs an embedding pass over every sentence in the corpus.

Every chunk records which strategy made it, and each strategy gets its own
index (see index/vector.py), so the three can be compared side by side.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rag.config import Settings
    from rag.embeddings import Embedder
    from rag.ingest.loaders import Document


@dataclass
class Chunk:
    """The unit of retrieval. Everything downstream — indexes, retrievers,
    citations, the dashboard's chunk inspector — passes these around."""

    chunk_id: str  # f"{doc_id}::{strategy}::{index}" — globally unique and stable
    doc_id: str  # which document this came from (relative path, e.g. "fastapi/tutorial/body.md")
    text: str  # the actual content that gets embedded and shown to the LLM
    index: int  # 0-based position within the document (chunk 3 comes after chunk 2)
    strategy: str  # "fixed" | "recursive" | "semantic" — which chunker made it
    section: str | None  # nearest markdown heading above this chunk, if known (citation context)
    char_count: int  # len(text), precomputed because reports/dashboards ask for it constantly
    metadata: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

# A sentence ends at ". ", "! ", "? " or a newline. Crude but dependency-free;
# fine for docs prose. (Abbreviations like "e.g." will over-split — acceptable
# noise, because the pieces get merged back up toward the target size anyway.)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n")

# Markdown heading line, e.g. "## Request Body". Captures the heading text so
# chunks can record the section they live under.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _make_chunk(doc: Document, strategy: str, index: int, text: str, section: str | None) -> Chunk:
    """Single construction point so chunk_id formatting can never drift
    between strategies."""
    return Chunk(
        chunk_id=f"{doc.doc_id}::{strategy}::{index}",
        doc_id=doc.doc_id,
        text=text,
        index=index,
        strategy=strategy,
        section=section,
        char_count=len(text),
        metadata={"title": doc.title or "", "format": doc.format},
    )


def _section_for_offset(doc_text: str, offset: int) -> str | None:
    """Return the text of the nearest markdown heading at or above `offset`.

    Used by the fixed chunker (which is structure-blind by design) so its
    chunks still carry a `section` label for citations. Linear scan is fine:
    it runs once per chunk at ingest time, never at query time.
    """
    section = None
    for m in _HEADING_RE.finditer(doc_text):
        if m.start() > offset:
            break
        section = m.group(2).strip()
    return section


# --------------------------------------------------------------------------
# Strategy 1: fixed-size sliding window
# --------------------------------------------------------------------------


def chunk_fixed(doc: Document, *, size: int, overlap: int) -> list[Chunk]:
    """Baseline: slide a `size`-character window over the text, stepping
    `size - overlap` characters each time.

    Why overlap exists: a sentence cut in half at a window boundary is
    unfindable in both halves. Overlap guarantees anything shorter than
    `overlap` characters appears WHOLE in at least one chunk.

    We nudge each cut to the nearest whitespace (within 80 chars) so we never
    split mid-word — pure cosmetics for the LLM, no effect on coverage.
    """
    if overlap >= size:
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    text = doc.text
    if len(text) <= size:  # tiny doc -> exactly one chunk, no windowing math
        return [_make_chunk(doc, "fixed", 0, text, _section_for_offset(text, 0))]

    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Nudge the end forward to the next whitespace so words stay whole
        # (skip when we're already at the end of the document).
        if end < len(text):
            ws = text.find(" ", end)
            nl = text.find("\n", end)
            candidates = [c for c in (ws, nl) if c != -1 and c - end <= 80]
            if candidates:
                end = min(candidates)
        piece = text[start:end].strip()
        if piece:  # windows over pure-whitespace stretches produce nothing
            chunks.append(
                _make_chunk(doc, "fixed", len(chunks), piece, _section_for_offset(text, start))
            )
        if end == len(text):
            break
        start += step
    return chunks


# --------------------------------------------------------------------------
# Strategy 2: recursive, structure-aware
# --------------------------------------------------------------------------

# Separators tried in order, biggest structural boundary first. The logic:
# prefer to cut where the AUTHOR already drew a line (a heading), fall back to
# paragraph breaks, then sentences, and only hard-cut characters as a last
# resort. All loaders emit markdown-style headings (loaders.py rewrites HTML
# <h1>-<h6> into "#"-headings) precisely so this list works on every format.
_SEPARATORS = [
    "\n# ",
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n\n",  # paragraph break
    "\n",  # line break
    ". ",  # sentence-ish
    " ",  # word
]


def _split_recursive(text: str, size: int, separators: list[str]) -> list[str]:
    """Recursively split `text` until every piece is <= size.

    Uses the FIRST separator in the list that actually appears in the text;
    pieces still too big are re-split with the REMAINING (finer) separators.
    If no separator is left, hard-cut every `size` characters — the guarantee
    that this function always terminates with pieces <= size.
    """
    if len(text) <= size:
        return [text]

    for i, sep in enumerate(separators):
        if sep not in text:
            continue
        pieces = text.split(sep)
        # Splitting eats the separator; glue it back onto the piece it opened
        # so heading markers ("## ...") survive into the chunks. For "\n## ",
        # the "\n" belongs to the previous piece and "## " to the next one.
        rejoined: list[str] = []
        for j, piece in enumerate(pieces):
            if j > 0:
                piece = sep.lstrip("\n") + piece
            if piece.strip():
                rejoined.append(piece)
        out: list[str] = []
        for piece in rejoined:
            if len(piece) <= size:
                out.append(piece)
            else:  # this piece needs a finer knife
                out.extend(_split_recursive(piece, size, separators[i + 1 :]))
        return out

    # No separator matched at all (e.g. one enormous unbroken token): hard cut.
    return [text[i : i + size] for i in range(0, len(text), size)]


def chunk_recursive(doc: Document, *, size: int, overlap: int) -> list[Chunk]:
    """Structure-aware chunking: split on the largest boundary, then merge
    small neighbors back up toward `size`.

    Two passes:
      1. SPLIT: `_split_recursive` produces pieces that are each <= size but
         possibly tiny (a lone heading line is its own piece).
      2. MERGE: greedily pack consecutive pieces into a buffer until adding
         the next one would exceed `size`, then flush. This keeps a heading
         glued to the paragraph below it instead of floating alone.

    Overlap here is *textual context carry-over*: each new chunk starts with
    the tail (`overlap` chars, snapped to a sentence start) of the previous
    one, so a fact straddling a flush boundary is still readable in one chunk.

    `section` tracking: while walking the pieces we remember the last heading
    seen, so every chunk knows which part of the document it belongs to —
    that's what makes citations like "Request Body — body.md" possible.
    """
    pieces = _split_recursive(doc.text, size, _SEPARATORS)

    chunks: list[Chunk] = []
    buffer = ""  # pieces merged so far, waiting to be flushed as one chunk
    buffer_section: str | None = None  # heading in effect when the buffer STARTED
    current_section: str | None = None  # heading most recently seen while walking

    def flush() -> None:
        nonlocal buffer
        text = buffer.strip()
        if text:
            chunks.append(_make_chunk(doc, "recursive", len(chunks), text, buffer_section))
        buffer = ""

    for piece in pieces:
        # Track the current heading (checked BEFORE merging so the section
        # label is right even when the heading opens a brand-new buffer).
        m = _HEADING_RE.match(piece.strip())
        if m:
            current_section = m.group(2).strip()

        if not buffer:
            buffer = piece
            buffer_section = current_section
            continue

        if len(buffer) + len(piece) + 1 <= size:
            buffer += "\n" + piece  # still fits — keep packing
        else:
            flush()
            # Start the next chunk with the tail of the previous one (overlap).
            tail = chunks[-1].text[-overlap:] if overlap and chunks else ""
            # Snap the tail to a sentence boundary so chunks don't open mid-word.
            first_break = _SENTENCE_RE.search(tail)
            if first_break:
                tail = tail[first_break.end() :]
            buffer = (tail + "\n" + piece).strip() if tail.strip() else piece
            buffer_section = current_section
    flush()
    return chunks


# --------------------------------------------------------------------------
# Strategy 3: semantic (embedding-driven breakpoints)
# --------------------------------------------------------------------------


def chunk_semantic(
    doc: Document,
    *,
    embedder: Embedder,
    breakpoint_percentile: int = 90,
    max_size: int = 2000,
    min_sentences: int = 3,
) -> list[Chunk]:
    """Split where the TOPIC shifts, as measured by embeddings.

    How it works, step by step:
      1. Split the document into sentences.
      2. Embed every sentence (this is the expensive part — one embedding
         pass over the whole corpus at ingest time).
      3. Cosine similarity between each consecutive sentence pair. Because
         embed_texts() returns normalized vectors, cosine = plain dot product.
      4. A "breakpoint" is any gap whose similarity is BELOW the Nth
         percentile of all gaps in this document — i.e. we cut at the ~10%
         most abrupt topic shifts (with the default percentile of 90).
      5. Sentences between breakpoints become one chunk.

    Guardrails (the raw algorithm misbehaves without them):
      - `min_sentences`: never emit a chunk smaller than this — one weird
        sentence shouldn't become its own retrieval unit.
      - `max_size`: a long stretch with no topic shift still gets cut, or a
        single chunk could swallow half the document.
    """
    sentences = [s.strip() for s in _SENTENCE_RE.split(doc.text) if s.strip()]
    if len(sentences) <= min_sentences:
        text = doc.text.strip()
        return [_make_chunk(doc, "semantic", 0, text, _first_section(doc))] if text else []

    vectors = embedder.embed_texts(sentences)  # (n_sentences, dim), unit-normalized

    # Similarity of each consecutive pair: dot product row i with row i+1.
    sims = np.sum(vectors[:-1] * vectors[1:], axis=1)  # shape (n-1,)

    # Cut where similarity dips below the percentile threshold. Note the
    # threshold is PER-DOCUMENT: a doc with uniform topics gets few cuts, a
    # doc that jumps around gets many. That adaptivity is the point.
    threshold = np.percentile(sims, 100 - breakpoint_percentile)
    breakpoints = {i for i in range(len(sims)) if sims[i] < threshold}

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    for i, sentence in enumerate(sentences):
        current.append(sentence)
        current_len += len(sentence) + 1
        gap_is_break = i in breakpoints and len(current) >= min_sentences
        too_big = current_len >= max_size
        if gap_is_break or too_big:
            text = " ".join(current).strip()
            chunks.append(
                _make_chunk(doc, "semantic", len(chunks), text, _section_for_sentence(doc, text))
            )
            current, current_len = [], 0
    if current:  # whatever's left after the last breakpoint
        text = " ".join(current).strip()
        if text:
            chunks.append(
                _make_chunk(doc, "semantic", len(chunks), text, _section_for_sentence(doc, text))
            )
    return chunks


def _first_section(doc: Document) -> str | None:
    m = _HEADING_RE.search(doc.text)
    return m.group(2).strip() if m else None


def _section_for_sentence(doc: Document, chunk_text: str) -> str | None:
    """Best-effort section label for a semantic chunk: find where the chunk's
    opening text sits in the original document and look up the heading above
    it. Sentence splitting rewrote whitespace, so fall back to the first 60
    chars as a search needle; if it moved too much, we just return None —
    a missing section label degrades a citation's *display*, nothing else."""
    needle = chunk_text[:60]
    pos = doc.text.find(needle)
    return _section_for_offset(doc.text, pos) if pos != -1 else None


# --------------------------------------------------------------------------
# Dispatcher — the only entry point the rest of the pipeline uses
# --------------------------------------------------------------------------

STRATEGIES = ("fixed", "recursive", "semantic")


def chunk(
    doc: Document,
    strategy: str,
    settings: Settings,
    embedder: Embedder | None = None,
) -> list[Chunk]:
    """Chunk one document with the named strategy.

    The embedder is only required for "semantic" (it embeds sentences);
    demanding it up front for the other strategies would force every caller
    to load a torch model it never uses.
    """
    if strategy == "fixed":
        return chunk_fixed(doc, size=settings.chunk_size, overlap=settings.chunk_overlap)
    if strategy == "recursive":
        return chunk_recursive(doc, size=settings.chunk_size, overlap=settings.chunk_overlap)
    if strategy == "semantic":
        if embedder is None:
            raise ValueError("semantic chunking needs an embedder (it embeds every sentence)")
        return chunk_semantic(doc, embedder=embedder)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
