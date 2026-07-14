"""API surface tests — the cheap contract checks.

These use FastAPI's TestClient (no server process). They deliberately avoid
the /v1/ask happy path: it loads two ML models and needs a live LLM, which is
integration-test territory (scripts/ask.py, the eval suite) — unit tests here
verify the edge behavior: validation, clear errors, health.
"""

from fastapi.testclient import TestClient

from rag.api.main import app


def test_health():
    with TestClient(app) as client:  # `with` runs the lifespan handler
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_empty_question_rejected_at_the_edge():
    """min_length on AskRequest.question must 422 before any model loads."""
    with TestClient(app) as client:
        response = client.post("/v1/ask", json={"question": ""})
        assert response.status_code == 422


def test_unknown_strategy_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/v1/ask", json={"question": "valid question?", "strategy": "banana"}
        )
        assert response.status_code == 422  # Literal[...] in the schema does this


def test_documents_unknown_strategy():
    with TestClient(app) as client:
        response = client.get("/v1/documents", params={"strategy": "banana"})
        assert response.status_code == 422


def test_openapi_lists_the_three_endpoints():
    with TestClient(app) as client:
        paths = client.get("/openapi.json").json()["paths"]
        assert "/v1/ask" in paths
        assert "/v1/documents" in paths
        assert "/v1/ingest" in paths
        assert "/health" in paths
