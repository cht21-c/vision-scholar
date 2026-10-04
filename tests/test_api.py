import json
import time

from fastapi.testclient import TestClient

from backend.main import create_app


def test_http_upload_search_sessions_export_and_sse(context, pdf_bytes):
    config, _, _, _ = context
    with TestClient(create_app(config)) as client:
        assert client.get("/api/health").json()["ok"]
        response = client.post("/api/papers/upload",
                               files={"file": ("../../evil.pdf", pdf_bytes, "application/pdf")})
        assert response.status_code == 200
        assert response.json()["id"] == "fixture"
        assert len(client.get("/api/papers").json()) == 1
        invalid = client.post("/api/papers/upload", files={"file": ("fake.pdf", b"oops")})
        assert invalid.status_code == 400
        assert client.get("/api/papers/fixture/pages/99").status_code == 400
        assert client.get("/api/papers/missing/pages/1").status_code == 404
        assert client.get("/api/papers/fixture/pdf").content.startswith(b"%PDF-")
        assert client.get("/api/search", params={"q": "residual shortcut"}).json()["results"]
        assert client.get("/api/code", params={"path": "../backend/config.py"}).status_code == 400
        session = client.post("/api/sessions", json={}).json()
        submitted = client.post("/api/runs", json={
            "session_id": session["id"], "prompt": "查看 patchify 函数源码", "mode": "code",
        })
        assert submitted.status_code == 202
        run_id = submitted.json()["id"]
        conflict = client.post("/api/runs", json={
            "session_id": session["id"], "prompt": "second task",
        })
        assert conflict.status_code == 409
        for _ in range(200):
            run = client.get(f"/api/runs/{run_id}").json()
            if run["status"] not in {"running", "queued"}:
                break
            time.sleep(0.02)
        assert run["status"] == "completed", run
        events = client.get(f"/api/runs/{run_id}/trace").json()["events"]
        cursor = events[-2]["seq"]
        replay = client.get(f"/api/runs/{run_id}/events", headers={"Last-Event-ID": str(cursor)})
        parsed = [json.loads(line[6:]) for line in replay.text.splitlines() if line.startswith("data: ")]
        assert [e["seq"] for e in parsed] == [events[-1]["seq"]]
        assert parsed[0]["type"] == "terminal"
        note = client.post("/api/notes", json={
            "session_id": session["id"], "title": "回溯记录", "content": run["answer"],
        })
        assert note.status_code == 201
        exported = client.get(f"/api/sessions/{session['id']}/export")
        assert "patchify" in exported.text
        assert "回溯记录" in exported.text
        assert len(client.get("/api/bootstrap").json()["notes"]) == 1
        # A browser page from another origin cannot mutate this local workspace.
        assert client.post("/api/sessions", json={},
                           headers={"Origin": "https://untrusted.test"}).status_code == 403


def test_http_validation_and_real_experiment(context):
    config, _, _, _ = context
    with TestClient(create_app(config)) as client:
        assert client.post("/api/experiments", json={"kind": "shell"}).status_code == 422
        assert client.post("/api/papers/import", json={"arxiv_id": "../bad"}).status_code == 400
        response = client.post("/api/experiments", json={"kind": "attention", "seed": 11})
        assert response.status_code == 200
        result = response.json()
        assert result["result"]["token_count"] == 16
        assert client.get(f"/api/experiments/{result['id']}").json() == result
        assert client.post("/api/logs/analyze", json={"content": "[]"}).status_code == 400
        assert client.post("/api/logs/upload", files={"file": ("bad.csv", b"\xff\xff")}).status_code == 400
