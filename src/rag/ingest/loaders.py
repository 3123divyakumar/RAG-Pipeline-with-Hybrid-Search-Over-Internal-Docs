"""Document loaders — normalize every input format into one Document model.

The rest of the pipeline never wants to know whether text came from markdown,
HTML, or a PDF. Loaders erase that difference and attach the metadata that
citations will need later (source file, title, page numbers).

Rules that matter here:
- Read text as UTF-8 explicitly. On Windows, open() without an encoding uses
  cp1252 and quietly mangles characters — a classic silent bug.
- Keep markdown heading lines (# ...): the recursive chunker splits on them.
- A file that fails to load is LOGGED and skipped, never silently dropped and
  never allowed to kill the whole ingest run.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Document:
    doc_id: str  # stable ID, e.g. relative path: "fastapi/tutorial/body.md"
    text: str  # clean plaintext (markdown syntax kept — chunkers use it)
    source_path: str
    format: str  # "md" | "html" | "pdf" | "txt"
    title: str | None  # first heading / <title> / filename
    metadata: dict = field(default_factory=dict)


def _normalize_whitespace(text: str) -> str:
    """Collapse the whitespace mess PDF/HTML extraction leaves behind."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _first_heading(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()
            # FastAPI docs suffix headings with anchors: "Request Body { #request-body }"
            return re.sub(r"\s*\{\s*#[^}]*\}\s*$", "", heading)
    return None


def _load_md(path: Path) -> tuple[str, str | None, dict]:
    text = path.read_text(encoding="utf-8")
    # Strip a frontmatter block (--- ... ---) if the file starts with one.
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + len("\n---") :]
    text = text.strip()
    return text, _first_heading(text), {}


def _load_txt(path: Path) -> tuple[str, str | None, dict]:
    return path.read_text(encoding="utf-8").strip(), None, {}


def _load_html(path: Path) -> tuple[str, str | None, dict]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else None

    # Boilerplate carries no meaning worth indexing — remove it before get_text().
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Rewrite <h1>-<h6> as markdown headings so ALL formats speak the same
    # structure language and the recursive chunker works on any of them.
    for level in range(1, 7):
        for heading in soup.find_all(f"h{level}"):
            heading.replace_with(f"\n\n{'#' * level} {heading.get_text(strip=True)}\n\n")

    return _normalize_whitespace(soup.get_text()), title, {}


def _load_pdf(path: Path) -> tuple[str, str | None, dict]:
    from pypdf import PdfReader

    reader = PdfReader(path)
    parts: list[str] = []
    page_offsets: list[dict] = []  # where each page starts in the joined text -> page citations
    pos = 0
    for page_no, page in enumerate(reader.pages, start=1):
        page_text = _normalize_whitespace(page.extract_text() or "")
        page_offsets.append({"page": page_no, "start": pos})
        parts.append(page_text)
        pos += len(page_text) + len("\n\n")
    text = "\n\n".join(parts)
    return text, None, {"page_count": len(reader.pages), "page_offsets": page_offsets}


_LOADERS = {
    ".md": _load_md,
    ".markdown": _load_md,
    ".html": _load_html,
    ".htm": _load_html,
    ".pdf": _load_pdf,
    ".txt": _load_txt,
}


def load_file(path: Path, corpus_dir: Path | None = None) -> Document:
    """Load one file, dispatching on its suffix. Raises on unsupported/broken files."""
    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        raise ValueError(f"unsupported format: {path}")
    text, title, metadata = loader(path)
    # Forward slashes even on Windows: doc_ids live in indexes and citations,
    # and must not change when the repo moves between OSes.
    doc_id = (
        str(path.relative_to(corpus_dir)).replace("\\", "/") if corpus_dir else path.name
    )
    return Document(
        doc_id=doc_id,
        text=text,
        source_path=str(path),
        format=path.suffix.lower().lstrip("."),
        title=title or path.stem,
        metadata=metadata,
    )


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Walk corpus_dir recursively and load every supported file.

    Survives individual failures: a broken file is logged and skipped so one
    bad PDF can't kill a 244-file ingest run.
    """
    docs: list[Document] = []
    skipped = 0
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _LOADERS:
            logger.warning("skipping unsupported file: %s", path)
            skipped += 1
            continue
        try:
            doc = load_file(path, corpus_dir=corpus_dir)
        except Exception:
            logger.exception("failed to load %s", path)
            skipped += 1
            continue
        if not doc.text:
            logger.warning("empty after extraction, skipping: %s", path)
            skipped += 1
            continue
        docs.append(doc)
    logger.info("loaded %d documents (%d skipped)", len(docs), skipped)
    return docs
