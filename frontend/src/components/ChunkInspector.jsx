// ChunkInspector — the exact context blocks the LLM saw, in prompt order.
//
// "What did the model actually read?" is the first question in every RAG
// debugging session; this panel answers it in one glance. Block numbers here
// are the same [n]s the answer cites (the prompt's build_context ordering),
// so clicking [2] in the answer highlights block 2 here.

import { useEffect, useRef, useState } from "react";

function ChunkBlock({ chunk, n, highlighted }) {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  // When this block gets selected via an answer chip, open it and scroll it
  // into view — the effect runs whenever `highlighted` flips.
  useEffect(() => {
    if (highlighted) {
      setOpen(true);
      ref.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }, [highlighted]);

  return (
    <div ref={ref} className={"chunk-block" + (highlighted ? " highlighted" : "")}>
      <button className="chunk-header" onClick={() => setOpen(!open)}>
        <span className="chunk-n">[{n}]</span>
        <span className="mono">{chunk.doc_id}</span>
        {chunk.section && <span className="chunk-section">§ {chunk.section}</span>}
        <span className="chunk-score" title={`source: ${chunk.source}`}>
          {chunk.source} score {chunk.score.toFixed(2)}
        </span>
        <span className="chunk-toggle">{open ? "▾" : "▸"}</span>
      </button>
      {open && <pre className="chunk-text">{chunk.text}</pre>}
    </div>
  );
}

export default function ChunkInspector({ chunks, selectedMarker, mode }) {
  return (
    <div className="card">
      <div className="card-title">
        context blocks — what the LLM saw ({mode} retrieval, {chunks.length} chunks)
      </div>
      {chunks.map((chunk, i) => (
        <ChunkBlock
          key={chunk.chunk_id}
          chunk={chunk}
          n={i + 1} /* 1-based to match the [n] markers */
          highlighted={selectedMarker === i + 1}
        />
      ))}
    </div>
  );
}
