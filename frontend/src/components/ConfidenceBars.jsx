// ConfidenceBars — the composite confidence and its three ingredients.
//
// Shown as bars instead of one number because the PARTS carry the diagnosis:
// high retrieval + low coverage = the model freelanced past its sources;
// low retrieval = we never found good material to begin with. One blended
// number would bury exactly the story worth telling. Timings ride along here
// because "what took how long" is the other at-a-glance health readout.

const PARTS = [
  { key: "retrieval", label: "retrieval", hint: "how relevant were the top chunks (reranker scores)" },
  { key: "citation_coverage", label: "citation coverage", hint: "verified citations / total citations" },
  { key: "completeness", label: "completeness", hint: "did the answer address every part of the question" },
];

function barColor(v) {
  return v >= 0.7 ? "var(--good)" : v >= 0.4 ? "var(--warn)" : "var(--bad)";
}

export default function ConfidenceBars({ confidence, timings }) {
  return (
    <div className="card confidence-card">
      <div className="card-title">
        confidence
        <span className="composite" style={{ color: barColor(confidence.composite) }}>
          {confidence.composite.toFixed(2)}
        </span>
      </div>

      {PARTS.map(({ key, label, hint }) => (
        <div className="bar-row" key={key} title={hint}>
          <span className="bar-label">{label}</span>
          <div className="bar-track">
            <div
              className="bar-fill"
              style={{
                width: `${confidence[key] * 100}%`,
                background: barColor(confidence[key]),
              }}
            />
          </div>
          <span className="bar-value">{confidence[key].toFixed(2)}</span>
        </div>
      ))}

      <div className="timings">
        {Object.entries(timings).map(([stage, ms]) => (
          <span key={stage} className="timing">
            {stage} {Math.round(ms)}ms
          </span>
        ))}
      </div>
    </div>
  );
}
