// CitationsTable — every (claim, cited chunk) pair and its judge verdict.
//
// This table is the project's differentiator made visible: most RAG demos
// stop at "the model printed [1]". Here each row says whether an independent
// judge call agreed that block [n] actually supports that exact sentence.
// A red row on screen is a hallucinated citation caught in the act.

export default function CitationsTable({ citations }) {
  return (
    <div className="card">
      <div className="card-title">
        citations · {citations.filter((c) => c.verified).length}/{citations.length} verified
      </div>
      <table className="citations-table">
        <thead>
          <tr>
            <th>[n]</th>
            <th>verdict</th>
            <th>claim</th>
            <th>cited chunk</th>
          </tr>
        </thead>
        <tbody>
          {citations.map((c, i) => (
            <tr key={i} className={c.verified ? "row-ok" : "row-bad"}>
              <td>[{c.marker}]</td>
              <td>
                {c.verified === true ? "✓ supported" : c.verified === false ? "✗ unsupported" : "—"}
              </td>
              <td className="claim-cell">{c.claim}</td>
              {/* Empty chunk_id = the model cited a block number that doesn't
                  exist — the most blatant hallucination this UI can display. */}
              <td className="mono">{c.chunk_id || "(nonexistent block!)"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
