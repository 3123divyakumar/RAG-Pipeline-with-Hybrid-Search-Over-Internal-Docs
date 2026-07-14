"""Download the documentation corpus into data/corpus/.

Run it:  uv run python scripts/fetch_corpus.py

What it fetches (all public, all free):
  - FastAPI docs   (markdown, from the fastapi/fastapi GitHub repo)
  - Pydantic docs  (markdown, from the pydantic/pydantic GitHub repo)
  - One PDF  ("Attention Is All You Need" — so your PDF loader has real work)
  - One HTML page (PEP 8 — same, for the HTML loader)

This corpus stands in for "a company's internal documentation": real technical
writing, full of exact identifiers and config keys — exactly the content where
hybrid retrieval visibly beats dense-only. data/ is gitignored; re-run this
script any time to restore it.
"""

import io
import sys
import tarfile
from pathlib import Path

import httpx

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"

# (name, github repo, branch, subdirectory containing the markdown docs)
REPOS = [
    ("fastapi", "fastapi/fastapi", "master", "docs/en/docs"),
    ("pydantic", "pydantic/pydantic", "main", "docs"),
]

# (relative output path, url) — single-file extras for the PDF/HTML loaders
EXTRAS = [
    ("papers/attention-is-all-you-need.pdf", "https://arxiv.org/pdf/1706.03762"),
    ("web/pep8.html", "https://peps.python.org/pep-0008/"),
]


def fetch_repo_docs(client: httpx.Client, name: str, repo: str, branch: str, subdir: str) -> int:
    """Download a repo tarball and extract only the markdown files under `subdir`."""
    url = f"https://codeload.github.com/{repo}/tar.gz/refs/heads/{branch}"
    print(f"[{name}] downloading {url} ...")
    resp = client.get(url)
    resp.raise_for_status()

    out_root = DATA_DIR / name
    count = 0
    with tarfile.open(fileobj=io.BytesIO(resp.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # tar paths look like "<repo>-<branch>/docs/en/docs/tutorial/body.md"
            parts = member.name.split("/", 1)
            if len(parts) < 2:
                continue
            rel = parts[1]
            if not rel.startswith(subdir + "/") or not rel.endswith(".md"):
                continue
            rel_out = rel[len(subdir) + 1 :]
            target = out_root / rel_out
            if not target.resolve().is_relative_to(out_root.resolve()):
                continue  # path traversal guard
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            if extracted is None:
                continue
            target.write_bytes(extracted.read())
            count += 1
    print(f"[{name}] wrote {count} markdown files -> {out_root}")
    return count


def fetch_extras(client: httpx.Client) -> int:
    count = 0
    for rel, url in EXTRAS:
        target = DATA_DIR / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[extra] downloading {url} ...")
        resp = client.get(url)
        resp.raise_for_status()
        target.write_bytes(resp.content)
        print(f"[extra] wrote {target} ({len(resp.content):,} bytes)")
        count += 1
    return count


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for name, repo, branch, subdir in REPOS:
            try:
                total += fetch_repo_docs(client, name, repo, branch, subdir)
            except httpx.HTTPError as e:
                print(f"[{name}] FAILED: {e} — check the branch name or your network", file=sys.stderr)
        try:
            total += fetch_extras(client)
        except httpx.HTTPError as e:
            print(f"[extras] FAILED: {e}", file=sys.stderr)

    print(f"\nDone. {total} files in {DATA_DIR}")
    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
