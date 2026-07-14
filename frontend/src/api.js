// The one place the frontend talks to the backend.
//
// Paths are relative ("/v1/ask", never "http://localhost:8000/v1/ask"):
// in dev the Vite proxy forwards them to :8000, in production FastAPI serves
// this app itself so the same origin answers directly. Hardcoding a host
// here is the classic way to build a dashboard that only works on one laptop.

export async function askQuestion(question, mode, strategy) {
  const response = await fetch("/v1/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, mode, strategy }),
  });
  if (!response.ok) {
    // FastAPI puts human-readable errors in {"detail": ...} — surface them
    // instead of a generic "request failed".
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || `request failed (${response.status})`);
  }
  return response.json(); // shape: AskResponse in src/rag/api/schemas.py
}

export async function fetchDocuments(strategy) {
  const response = await fetch(`/v1/documents?strategy=${strategy}`);
  if (!response.ok) throw new Error(`documents request failed (${response.status})`);
  return response.json();
}
