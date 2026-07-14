# Decision Journal

Every tunable and architectural choice in this repo gets an entry: what was decided,
what the alternatives were, and what evidence picked the winner. Entries are written at decision time, so some rest on reasoning rather
than measurement — those say so.

---

## Embedding model: BGE-small-en-v1.5 (local)          (2026-07-08)

**Decision:** `BAAI/bge-small-en-v1.5` via sentence-transformers, running on CPU.
**Alternatives considered:**
- `all-MiniLM-L6-v2` — smaller/faster, but measurably weaker on retrieval benchmarks (MTEB), and it lacks BGE's asymmetric query-prefix scheme that helps question→passage matching.
- OpenAI `text-embedding-3-small` — stronger and cheap (~$0.02/1M tokens), but this build is deliberately $0 and offline-capable; the `Embedder` interface keeps the swap to a one-class change if evals ever justify it.
**Evidence:** MTEB retrieval scores + the constraint that the deployed container must run the same model on CPU within ~1GB RAM.
**Revisit when:** retrieval hit@5 is the weakest metric — the first upgrade to try is a bigger embedder (`bge-base`), before touching chunking again.

## LLM strategy: OpenAI-compatible client, Ollama local / Groq deployed   (2026-07-08)

**Decision:** one `LLMClient` built on the `openai` package with configurable `base_url` — `qwen2.5:7b` on Ollama for development, `llama-3.3-70b-versatile` on Groq's free tier for the deployed demo.
**Alternatives considered:**
- Ollama in the cloud container — an 8B model needs more RAM than Railway/Render hobby tiers offer; a $0 deploy is impossible this way.
- Provider-specific SDKs per vendor — more code, no benefit while every chosen provider speaks the OpenAI protocol. (Anthropic's API is a different SDK — noted as a stretch goal, behind the same interface.)
**Evidence:** qwen2.5:7b was already pulled locally (4.7GB); Groq free tier serves a 70B model with generous limits — the deployed demo is *stronger* than local dev, for free.
**Revisit when:** Groq free limits throttle the demo, or eval faithfulness with the local 7B is too weak to develop against (then: better local model or a paid API for eval runs only).

---

## Chunk size 800 chars, overlap 150                    (2026-07-14)

**Decision:** `chunk_size=800`, `chunk_overlap=150` (characters) for fixed and recursive.
**Alternatives considered:** 400 (too many single-sentence chunks — context-free answers), 1500+ (embedding dilution: one vector averaging several ideas). Token-based sizing — more principled but adds a tokenizer dependency at ingest; characters are ~4:1 to tokens and good enough for docs prose.
**Evidence:** 800 chars ≈ 200 tokens ≈ one tutorial subsection — matches "one embedding, one idea" for this corpus. Overlap 150 ≈ 1–2 sentences so a boundary-straddling fact survives whole in one chunk.
**Revisit when:** the evals show retrieval hit@5 lagging while answers cite truncated context — try 1200/200 first.

## Dedup threshold 0.95 cosine                          (2026-07-14)

**Decision:** drop chunks with cosine ≥ 0.95 to an earlier chunk (keep-first).
**Alternatives considered:** 0.90 (inspection risk: merely-related paragraphs start dying), exact-hash dedup only (misses reworded boilerplate), MinHash (needed at millions of chunks, overkill at ~4k).
**Evidence:** at 0.95 the ingest run drops 44/190/121 chunks (fixed/recursive/semantic) out of ~3.2–4k — spot-checking the report shows repeated doc boilerplate, not real content. Keep-first keeps chunk_ids stable across runs (citations + eval caches depend on that).
**Revisit when:** eval answers show the top-5 filled with near-copies (threshold too high) or golden docs missing from the index (too low). At 1M chunks: ANN-based dedup.

## RRF k=60, weights 0.7 dense / 0.3 sparse             (2026-07-14)

**Decision:** weighted reciprocal rank fusion, `rrf_k=60`, dense_weight 0.7.
**Alternatives considered:** score normalization + weighted sum (needs per-corpus calibration, breaks when the corpus changes — the reason RRF exists); k=0 (top-of-list tyranny: rank 1 scores 2× rank 2); equal weights (BM25's exact-match wins matter most on identifier queries, which are a minority of real questions — dense deserves the majority vote).
**Evidence:** k=60 is the value from Cormack & Clarke 2009 and has survived replication since. Weights are a starting hypothesis to be tested by `run_eval.py --mode dense` vs `--mode hybrid`.
**Revisit when:** the eval numbers land — if hybrid barely beats dense-only, try 0.6/0.4 before concluding fusion doesn't pay.

## Rerank 20 → 5 with ms-marco-MiniLM-L-6-v2            (2026-07-14)

**Decision:** cross-encoder reranks the top 20 fused candidates; top 5 reach the LLM.
**Alternatives considered:** no reranking (bi-encoder ordering is noticeably noisier at the top); rerank 50 (2.5× the latency for candidates that rarely crack the top 5); larger cross-encoders (better but slower on CPU — this one is the standard speed/quality point).
**Evidence:** two-stage retrieve-wide/rerank-narrow is the production-standard shape; 20 candidates ≈ 1–2s of CPU cross-encoding. Bonus: reranker logits are roughly calibrated, which is what makes the confidence gate possible at all.
**Revisit when:** deployed container RAM forces `RERANK_ENABLED=false` (documented degradation), or eval shows the right doc retrieved at rank 8–20 but dropped — then rerank more, keep more, or both.

## Confidence gate at 0.35; composite 0.4/0.4/0.2       (2026-07-14)

**Decision:** skip generation when sigmoid-squashed top-3 reranker confidence < 0.35; composite confidence = 0.4·retrieval + 0.4·citation_coverage + 0.2·completeness.
**Alternatives considered:** always generate and rely on the prompt's refusal rule alone (models under instruction pressure answer anyway — belt needs braces); equal thirds for the composite (a complete wrong answer is worse than an incomplete right one, so completeness gets half-weight).
**Evidence:** 0.35 ≈ mean reranker logit of about -0.6 across the top 3 — i.e. the reranker actively thinks the chunks are irrelevant. Cheap to gate: refusing costs zero LLM calls.
**Revisit when:** the evals measure the gate against the 4 unanswerable golden questions — false-refusals on answerable questions mean lower the threshold; hallucinated answers on unanswerables mean raise it. Also check composite-vs-correctness correlation and retune the weights.

## Judge = same local model, separately configurable    (2026-07-14)

**Decision:** citation verification and eval judging run on `JUDGE_MODEL` (locally the same qwen2.5:7b that writes answers; on Groq a different model, llama-3.1-8b-instant).
**Alternatives considered:** always a second local model (another 4GB+ download and RAM bill for dev); skipping verification locally (the whole differentiator gone from the dev loop).
**Evidence:** self-preference bias is real but mitigated: strict binary rubric, evidence-only framing (judge sees one chunk + one claim, never its own full answer), schema-constrained JSON verdicts, and different judge in deployment.
**Revisit when:** the manual audit (grading 20 judge verdicts by hand) shows <80% agreement with the judge — then a stronger judge model becomes non-negotiable.

## Eval runs cache raw results before scoring           (2026-07-14)

**Decision:** every AskResult is serialized to `data/eval_runs/<run_id>/qNNN.json` before any metric touches it; `--rescore <run_id>` re-grades without re-generating.
**Alternatives considered:** score-as-you-go with no cache (every judge-prompt tweak costs a full generation hour); a database (a folder of JSON files is greppable, diffable, and zero-dependency at this scale).
**Evidence:** metrics/judge prompts change far more often than the pipeline's answers do; re-scoring is minutes, re-generating is an hour on CPU.
**Revisit when:** eval sets grow past a few hundred questions — then sqlite.

## Indexes baked into the Docker image                  (2026-07-14)

**Decision:** `docker build` copies the locally-built `.chroma/` and BM25 pickles into the image; the container boots ready to answer.
**Alternatives considered:** volume + seed-on-boot script (first boot spends ~15 min embedding on the host's CPU and can time out Railway's healthcheck); ingest endpoint only (empty demo until someone uploads).
**Evidence:** the corpus is fixed and small (~50MB indexed); an image that IS the demo is more reproducible than image + external state, and rebuilding the image is the natural "re-ingest" for a demo.
**Revisit when:** the corpus grows or updates frequently — then a volume and background re-indexing job.

<!-- New entries go above this line. -->

