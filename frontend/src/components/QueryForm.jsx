// QueryForm — question input + the two experiment knobs.
//
// The mode toggle (hybrid vs dense) is the dashboard's best demo move:
// ask an identifier-style question ("response_model_exclude_unset") in both
// modes and watch dense-only retrieval fumble what hybrid nails.

import { useState } from "react";

const EXAMPLES = [
  "How do I declare a request body in FastAPI?",
  "What does response_model_exclude_unset do?",
  "How do I load settings from a .env file with pydantic-settings?",
  "What is multi-head attention?",
];

export default function QueryForm({ onSubmit, loading }) {
  // Form fields are "controlled": React state is the single source of truth,
  // the DOM just displays it. That's what makes the example buttons able to
  // fill the box programmatically.
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState("hybrid");
  const [strategy, setStrategy] = useState("recursive");

  function submit(e) {
    e.preventDefault(); // stop the browser's full-page form POST
    if (question.trim().length >= 3 && !loading) onSubmit(question.trim(), mode, strategy);
  }

  return (
    <form className="query-form" onSubmit={submit}>
      <textarea
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask anything about the indexed docs (FastAPI, Pydantic, the Transformer paper, PEP 8)…"
        rows={3}
        onKeyDown={(e) => {
          // Enter submits; Shift+Enter makes a newline (chat-app convention).
          if (e.key === "Enter" && !e.shiftKey) submit(e);
        }}
      />

      <div className="controls">
        <div className="control-group">
          <label>retrieval</label>
          {/* segmented toggle — exactly two options, so two buttons beat a dropdown */}
          <div className="segmented">
            {["hybrid", "dense"].map((m) => (
              <button
                key={m}
                type="button"
                className={mode === m ? "active" : ""}
                onClick={() => setMode(m)}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        <div className="control-group">
          <label>chunking</label>
          <select value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            <option value="recursive">recursive (default)</option>
            <option value="fixed">fixed window</option>
            <option value="semantic">semantic</option>
          </select>
        </div>

        <button type="submit" className="ask-button" disabled={loading || question.trim().length < 3}>
          {loading ? "thinking…" : "ask"}
        </button>
      </div>

      <div className="examples">
        {EXAMPLES.map((ex) => (
          <button key={ex} type="button" className="example" onClick={() => setQuestion(ex)}>
            {ex}
          </button>
        ))}
      </div>
    </form>
  );
}
