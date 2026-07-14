// App — the whole dashboard state lives here.
//
// Component tree (one file each, in components/):
//   App
//   ├── QueryForm        question box + hybrid/dense toggle + strategy picker
//   ├── AnswerCard       the answer, with [n] markers rendered as clickable chips
//   ├── ConfidenceBars   the composite score and its three parts
//   ├── CitationsTable   every citation with its verification verdict
//   └── ChunkInspector   the exact chunks the LLM saw, expandable
//
// Data flows one way: App owns the AskResponse and hands pieces down as
// props. The only upward communication is callbacks (onSubmit, onSelect).
// That's the entire React mental model this project needs.

import { useState } from "react";
import { askQuestion } from "./api.js";
import QueryForm from "./components/QueryForm.jsx";
import AnswerCard from "./components/AnswerCard.jsx";
import ConfidenceBars from "./components/ConfidenceBars.jsx";
import CitationsTable from "./components/CitationsTable.jsx";
import ChunkInspector from "./components/ChunkInspector.jsx";

export default function App() {
  const [result, setResult] = useState(null); // last AskResponse, or null
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  // Which context block [n] the user clicked in the answer — the inspector
  // highlights and scrolls to it. Shared state, so it lives up here.
  const [selectedMarker, setSelectedMarker] = useState(null);

  async function handleAsk(question, mode, strategy) {
    setLoading(true);
    setError(null);
    setSelectedMarker(null);
    try {
      setResult(await askQuestion(question, mode, strategy));
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setLoading(false); // runs on success AND failure — the button un-sticks
    }
  }

  return (
    <div className="app">
      <header>
        <h1>RAG dashboard</h1>
        <p className="subtitle">
          hybrid retrieval · RRF fusion · cross-encoder rerank · verified citations
        </p>
      </header>

      <QueryForm onSubmit={handleAsk} loading={loading} />

      {error && <div className="error-box">{error}</div>}

      {loading && (
        <div className="loading">
          retrieving, generating and verifying… (local 7B model — can take ~a minute)
        </div>
      )}

      {result && !loading && (
        <>
          <div className="result-grid">
            <AnswerCard
              result={result}
              selectedMarker={selectedMarker}
              onMarkerClick={setSelectedMarker}
            />
            <ConfidenceBars confidence={result.confidence} timings={result.timings_ms} />
          </div>
          {result.citations.length > 0 && <CitationsTable citations={result.citations} />}
          <ChunkInspector
            chunks={result.chunks}
            selectedMarker={selectedMarker}
            mode={result.mode}
          />
        </>
      )}
    </div>
  );
}
