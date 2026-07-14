// AnswerCard — renders the answer with [n] markers turned into live chips.
//
// The interesting part is the parsing: the answer arrives as plain text with
// "[2]"-style markers embedded. We split it on the marker pattern and rebuild
// it as text nodes + <button> chips, colored by that citation's verification
// verdict (green = judge confirmed the chunk supports the claim, red = it
// didn't). Clicking a chip tells the ChunkInspector to reveal that block —
// the [n] -> block mapping is the same one the backend prompt established.

const MARKER_SPLIT = /(\[\d+\])/g; // capture group => markers stay in the split output

export default function AnswerCard({ result, selectedMarker, onMarkerClick }) {
  // marker number -> verified? Verification is per-citation; a marker cited
  // twice gets the AND of its verdicts (one bad use taints the marker chip).
  const verdictByMarker = {};
  for (const c of result.citations) {
    verdictByMarker[c.marker] =
      c.marker in verdictByMarker ? verdictByMarker[c.marker] && c.verified : c.verified;
  }

  function renderWithChips(text) {
    return text.split(MARKER_SPLIT).map((part, i) => {
      const m = part.match(/^\[(\d+)\]$/);
      if (!m) return <span key={i}>{part}</span>;
      const n = Number(m[1]);
      const verified = verdictByMarker[n];
      const cls =
        verified === true ? "chip verified" : verified === false ? "chip failed" : "chip";
      return (
        <button
          key={i}
          className={cls + (selectedMarker === n ? " selected" : "")}
          title={
            verified === true
              ? "verified: the cited chunk supports this claim"
              : verified === false
                ? "FAILED verification: the cited chunk does not support this claim"
                : "unverified"
          }
          onClick={() => onMarkerClick(n)}
        >
          {n}
        </button>
      );
    });
  }

  return (
    <div className={"card answer-card" + (result.answered ? "" : " idk")}>
      <div className="card-title">
        {result.answered ? "answer" : "no answer — honest refusal"}
        <span className="badge">{result.mode}</span>
        <span className="badge">{result.strategy}</span>
      </div>
      <div className="answer-text">{renderWithChips(result.answer)}</div>
    </div>
  );
}
