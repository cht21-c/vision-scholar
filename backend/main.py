from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.agent import AgentService, safe_error
from backend.config import ROOT, Settings
from backend.experiments import ExperimentInput, analyze_log, execute_experiment
from backend.papers import CATALOG, MAX_PDF_BYTES, PaperLibrary, search_arxiv
from backend.retrieval import HybridIndex
from backend.store import ConflictError, Store, dump
from backend.studies import StudyConfig, StudyService
from backend.tools import inspect_code


class SessionInput(BaseModel):
    title: str = Field(default="新的研究", min_length=1, max_length=160)


class RunInput(BaseModel):
    session_id: str
    prompt: str = Field(min_length=1, max_length=16000)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    mode: Literal["auto", "paper", "code", "experiment"] = "auto"
    harness: Literal["legacy", "openharness", "upgraded", "research"] = "research"
    retriever: Literal["lsa", "neural"] = "lsa"


class ImportInput(BaseModel):
    arxiv_id: str = Field(min_length=1, max_length=32)


class NoteInput(BaseModel):
    title: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=20000)
    session_id: str | None = None


class LogInput(BaseModel):
    content: str = Field(min_length=1, max_length=2_000_000)


def read_reports(directory: Path) -> list[dict]:
    reports = []
    for path in sorted(directory.glob("*.json"), reverse=True)[:20]:
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                reports.append({"file": path.name, **data})
        except (ValueError, OSError):
            continue
    return reports


def create_app(settings: Settings | None = None, client_factory=None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.prepare()
    store = Store(settings.data_dir / "scholar.db")
    library = PaperLibrary(store, settings.data_dir / "papers")
    index = HybridIndex(store)
    agent = AgentService(settings, store, library, index, client_factory)
    studies = StudyService(store, settings.data_dir / "studies")
    experiment_slot = asyncio.Semaphore(1)

    @asynccontextmanager
    async def lifespan(app):
        studies.recover_interrupted()
        await agent.start()
        yield
        await agent.close()

    app = FastAPI(title="Vision Scholar", version="0.1.0", lifespan=lifespan)
    app.state.store, app.state.agent, app.state.library = store, agent, library
    allowed_origins = {f"http://{host}:{port}" for host in ("localhost", "127.0.0.1")
                       for port in (5173, 8765)}
    app.add_middleware(CORSMiddleware, allow_origins=list(allowed_origins),
                       allow_methods=["GET", "POST"], allow_headers=["Content-Type", "Last-Event-ID"])

    @app.middleware("http")
    async def local_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method not in {"GET", "HEAD", "OPTIONS"} and origin and origin not in allowed_origins:
            return JSONResponse({"detail": "不允许来自该网页的写请求"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(ConflictError)
    async def conflict_handler(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(KeyError)
    async def key_handler(request, exc):
        return JSONResponse({"detail": str(exc).strip("'")}, status_code=404)

    @app.exception_handler(ValueError)
    async def value_handler(request, exc):
        return JSONResponse({"detail": safe_error(exc, settings)}, status_code=400)

    @app.exception_handler(httpx.HTTPError)
    async def network_handler(request, exc):
        return JSONResponse({"detail": "论文服务连接失败，请稍后重试。"}, status_code=502)

    @app.get("/api/health")
    async def health():
        return {"ok": True, "model": settings.public()}

    @app.get("/api/bootstrap")
    async def bootstrap():
        papers = store.papers()
        return {
            "model": settings.public(), "papers": papers, "sessions": store.sessions(),
            "notes": store.notes(), "experiments": store.experiments(), "studies": studies.list(),
            "reports": read_reports(settings.data_dir / "reports"),
            "stats": {"papers": len(papers), "pages": sum(p["page_count"] for p in papers),
                      "chunks": sum(p["chunk_count"] for p in papers)},
        }

    @app.post("/api/sessions", status_code=201)
    async def create_session(data: SessionInput):
        return store.create_session(data.title)

    @app.get("/api/sessions/{session_id}")
    async def session(session_id: str):
        data = store.session(session_id)
        data.pop("history")
        return {**data, "runs": store.runs(session_id)}

    @app.get("/api/sessions/{session_id}/export")
    async def export_session(session_id: str):
        data = store.session(session_id)
        lines = [f"# {data['title']}", ""]
        for run in store.runs(session_id):
            lines += [f"## 问题\n\n{run['prompt']}", "", run["answer"] or f"任务状态：{run['status']}",
                      "", f"模式：{run['usage'].get('provider', '—')} · Run ID：{run['id']}", ""]
            for citation in run["citations"]:
                lines.append(f"- {citation['id']}: {citation.get('title', '')}")
        for note in store.notes():
            if note["session_id"] == session_id:
                lines += ["", f"## 笔记：{note['title']}", note["content"]]
        return Response("\n".join(lines), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": 'attachment; filename="research.md"'})

    @app.post("/api/runs", status_code=202)
    async def create_run(data: RunInput):
        if not data.prompt.strip():
            raise ValueError("问题不能为空")
        if not settings.public()["available"] and not client_factory:
            raise HTTPException(503, "尚未配置模型；论文检索和实验室仍可独立使用。")
        run = store.create_run(data.session_id, data.prompt.strip(), data.paper_ids, data.mode,
                               data.harness, data.retriever)
        store.event(run["id"], "queued", {"status": "queued"})
        agent.launch(run["id"])
        return run

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str):
        return store.run(run_id)

    @app.post("/api/runs/{run_id}/cancel")
    async def cancel_run(run_id: str):
        return await agent.cancel(run_id)

    @app.get("/api/runs/{run_id}/trace")
    async def trace(run_id: str, after: int = 0):
        store.run(run_id)
        return {"events": store.events(run_id, max(after, 0))}

    @app.get("/api/runs/{run_id}/events")
    async def stream(run_id: str, request: Request, after: int = 0):
        store.run(run_id)
        last_id = request.headers.get("last-event-id", "0")
        try:
            cursor = max(0, after, int(last_id))
        except ValueError as exc:
            raise ValueError("无效的事件序号") from exc

        async def generate():
            nonlocal cursor
            ticks = 0
            while not await request.is_disconnected():
                events = store.events(run_id, cursor)
                for event in events:
                    cursor = event["seq"]
                    yield f"id: {cursor}\ndata: {dump(event)}\n\n"
                if len(events) == 500:
                    continue
                if store.run(run_id)["status"] not in {"running", "queued"}:
                    break
                ticks += 1
                if ticks % 60 == 0:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.25)
        return StreamingResponse(generate(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/papers")
    async def papers():
        return store.papers()

    @app.get("/api/catalog")
    async def catalog():
        return CATALOG

    @app.get("/api/search")
    async def search(q: str, paper_id: str | None = None, limit: int = 6):
        if len(q) > 1200:
            raise ValueError("搜索词过长")
        if paper_id:
            store.paper(paper_id)
        return {"results": await asyncio.to_thread(
            index.search, q, [paper_id] if paper_id else None, limit,
        )}

    @app.get("/api/arxiv")
    async def arxiv(q: str):
        return {"results": await search_arxiv(q)}

    @app.post("/api/papers/seed")
    async def seed():
        papers = await library.seed()
        index.invalidate()
        return {"papers": papers}

    @app.post("/api/papers/import")
    async def import_paper(data: ImportInput):
        result = await library.import_arxiv(data.arxiv_id)
        index.invalidate()
        return result

    @app.post("/api/papers/upload")
    async def upload(file: UploadFile = File(...)):
        data = await file.read(MAX_PDF_BYTES + 1)
        await file.close()
        result = await asyncio.to_thread(library.ingest, data)
        index.invalidate()
        return result

    @app.get("/api/papers/{paper_id}/pages/{page}")
    async def page(paper_id: str, page: int):
        return library.page(paper_id, page)

    @app.get("/api/papers/{paper_id}/pdf")
    async def pdf(paper_id: str):
        paper = store.paper(paper_id)
        return FileResponse(settings.data_dir / "papers" / paper["file_name"],
                            media_type="application/pdf",
                            headers={"Content-Disposition": "inline"})

    @app.get("/api/code")
    async def code(path: str = "vision_ops.py", symbol: str = ""):
        return inspect_code(path, symbol)

    @app.post("/api/experiments")
    async def experiment(data: ExperimentInput):
        async with experiment_slot:
            return await asyncio.to_thread(execute_experiment, data, store,
                                           settings.data_dir / "experiments")

    @app.post("/api/studies", status_code=201)
    async def plan_study(data: StudyConfig):
        return await asyncio.to_thread(studies.plan, data)

    @app.get("/api/studies/{plan_id}")
    async def study_detail(plan_id: str):
        return await asyncio.to_thread(studies.read, plan_id)

    @app.post("/api/studies/{plan_id}/run")
    async def run_study(plan_id: str):
        return await studies.run(plan_id)

    @app.get("/api/studies/{plan_id}/evidence/{filename}")
    async def study_evidence(plan_id: str, filename: str):
        path = studies.evidence_path(plan_id, filename)
        return FileResponse(path, filename=filename)

    @app.get("/api/experiments/{experiment_id}")
    async def experiment_detail(experiment_id: str):
        result = next((e for e in store.experiments() if e["id"] == experiment_id), None)
        if not result:
            raise KeyError("实验不存在")
        return result

    @app.post("/api/logs/analyze")
    async def log_analysis(data: LogInput):
        return await asyncio.to_thread(analyze_log, data.content)

    @app.post("/api/logs/upload")
    async def log_upload(file: UploadFile = File(...)):
        data = await file.read(2_000_001)
        await file.close()
        if len(data) > 2_000_000:
            raise ValueError("日志超过 2 MB")
        try:
            content = data.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ValueError("日志需为 UTF-8 JSON 或 CSV") from exc
        return await asyncio.to_thread(analyze_log, content)

    @app.post("/api/notes", status_code=201)
    async def note(data: NoteInput):
        return store.add_note(data.title, data.content, data.session_id)

    @app.get("/api/reports")
    async def reports():
        return read_reports(settings.data_dir / "reports")

    @app.get("/api/reports/{filename}")
    async def report(filename: str):
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise ValueError("非法报告文件名")
        target = settings.data_dir / "reports" / filename
        if not target.is_file():
            raise KeyError("报告不存在")
        return FileResponse(target, media_type="application/json", filename=filename)

    dist = ROOT / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
    return app


app = create_app()
