"""Schema-driven, bounded research tools for the real OpenHarness registry."""
from __future__ import annotations

import ast
import asyncio
import inspect
import re
from collections.abc import Callable
from pathlib import Path

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolRegistry, ToolResult
from pydantic import BaseModel, ConfigDict, Field

from backend.artifacts import ArtifactInput, ArtifactStore
from backend.config import ROOT
from backend.experiments import ExperimentInput, analyze_log, execute_experiment
from backend.papers import PaperLibrary, search_arxiv
from backend.retrieval import HybridIndex
from backend.store import Store, dump
from backend.studies import StudyConfig, StudyID, StudyService


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchInput(Arguments):
    query: str = Field(min_length=1, max_length=1200)
    paper_ids: list[str] = Field(default_factory=list, max_length=20)
    limit: int = Field(default=6, ge=1, le=12)


class ReadInput(Arguments):
    paper_id: str = Field(min_length=1, max_length=64)
    page: int = Field(ge=1, le=100)


class ArxivSearchInput(Arguments):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=5, ge=1, le=10)


class ArxivImportInput(Arguments):
    arxiv_id: str


class CodeInput(Arguments):
    path: str = Field(default="vision_ops.py", max_length=200)
    symbol: str = Field(default="", max_length=120)


class LogInput(Arguments):
    content: str = Field(default="", max_length=2_000_000)
    experiment_id: str = Field(default="", max_length=32)


class NoteInput(Arguments):
    title: str = Field(min_length=1, max_length=160)
    content: str = Field(min_length=1, max_length=12000)

class StudyReference(Arguments):
    plan_id: str = Field(min_length=1, max_length=64,
                         description="Prefer the exact short plan_ref (e.g. study-1) returned "
                                     "by plan_study this run; full 64-char plan_id also accepted.")


class DomainTool(BaseTool):
    def __init__(self, name: str, description: str, input_model: type[BaseModel],
                 handler: Callable, read_only: bool = True):
        self.name, self.description, self.input_model = name, description, input_model
        self.handler, self.read_only = handler, read_only

    def is_read_only(self, arguments: BaseModel) -> bool:
        return self.read_only

    async def execute(self, arguments: BaseModel, context: ToolExecutionContext) -> ToolResult:
        try:
            if inspect.iscoroutinefunction(self.handler):
                result = await self.handler(arguments)
            else:
                result = await asyncio.to_thread(self.handler, arguments)
            return ToolResult(output=dump(result))
        except (ValueError, KeyError, OSError) as exc:
            return ToolResult(output=dump({"error": str(exc)}), is_error=True)


def inspect_code(path: str, symbol: str = "", root: Path | None = None) -> dict:
    root = (root or ROOT / "examples").resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root) or target.suffix != ".py" or not target.is_file():
        raise ValueError("仅能读取 examples/ 下的 Python 文件")
    if target.stat().st_size > 100000:
        raise ValueError("源码超过 100 KB")
    source = target.read_text(encoding="utf-8")
    tree = ast.parse(source)
    symbols = [
        {"name": node.name, "kind": type(node).__name__,
         "start": node.lineno, "end": node.end_lineno}
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    lines = source.splitlines()
    start, end = 1, len(lines)
    if symbol:
        match = next((s for s in symbols if s["name"] == symbol), None)
        if not match:
            raise ValueError(f"符号不存在；可用符号：{', '.join(s['name'] for s in symbols)}")
        start, end = match["start"], match["end"]
    relative = target.relative_to(root).as_posix()
    evidence_id = f"code:examples/{relative}:L{start}-L{end}"
    return {
        "id": evidence_id, "citation": f"[{evidence_id}]", "kind": "code",
        "path": f"examples/{relative}", "start": start, "end": end, "symbols": symbols,
        "text": "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1)),
        "title": f"{relative} · L{start}–{end}",
    }


class ResearchTools:
    def __init__(self, store: Store, library: PaperLibrary, index: HybridIndex,
                 data_dir: Path, run: dict, *, idempotent_notes: bool = False):
        self.store, self.library, self.index = store, library, index
        self.data_dir, self.run = data_dir, run
        self.evidence: dict[str, dict] = {}
        self.experiment_ids: list[str] = []
        self.idempotent_notes = idempotent_notes
        self.artifacts = ArtifactStore(data_dir / "artifacts" / run["session_id"])
        self.studies = StudyService(store, data_dir / "studies")
        self.study_refs: dict[str, str] = {}

    def remember(self, items: list[dict]) -> None:
        for item in items:
            self.evidence[item["id"]] = item

    def search(self, args: SearchInput) -> dict:
        selected = self.run["paper_ids"]
        ids = args.paper_ids or selected
        if selected and any(i not in selected for i in ids):
            raise ValueError("请只检索当前选定的论文")
        for paper_id in ids:
            self.store.paper(paper_id)
        results = self.index.search(args.query, ids, args.limit)
        self.remember(results)
        return {"method": getattr(self.index, "method", "BM25 + LSA cosine + RRF + lexical reranking"),
                "results": results}

    def read(self, args: ReadInput) -> dict:
        if self.run["paper_ids"] and args.paper_id not in self.run["paper_ids"]:
            raise ValueError("该论文不在当前选择范围")
        result = self.library.page(args.paper_id, args.page)
        chunks = [{**c, "title": result["paper"]["title"], "citation": f"[{c['id']}]"}
                  for c in result["chunks"]]
        self.remember(chunks)
        return {"paper_id": args.paper_id, "page": args.page, "chunks": chunks}

    def code(self, args: CodeInput) -> dict:
        result = inspect_code(args.path, args.symbol)
        self.remember([result])
        return result

    async def experiment(self, args: ExperimentInput) -> dict:
        if (not re.search(r"运行|跑|做.{0,5}实验|实验一下|演示|run|execute|experiment|demo",
                          self.run["prompt"], re.I)
                or re.search(r"不要运行|别运行|不运行|do not run|don't run",
                             self.run["prompt"], re.I)):
            raise ValueError("只有用户明确要求运行实验时才能执行")
        # Bound work before scheduling; at most two small experiments per Agent run.
        if len(self.experiment_ids) >= 2:
            raise ValueError("每轮最多运行两个实验")
        self.experiment_ids.append("pending")
        result = await asyncio.to_thread(
            execute_experiment, args, self.store, self.data_dir / "experiments",
        )
        self.experiment_ids[self.experiment_ids.index("pending")] = result["id"]
        return result

    def log(self, args: LogInput) -> dict:
        if args.experiment_id:
            row = next((e for e in self.store.experiments() if e["id"] == args.experiment_id), None)
            if not row or "training_log" not in row["result"]:
                raise ValueError("实验日志不存在")
            result = row["result"]
            return {
                **analyze_log(dump(result["training_log"])),
                "metrics_origin": "Python computation of this application's stored experiment log",
                "log_source": result.get("training_log_source", "stored experiment"),
                "experiment_id": row["id"], "seed": result.get("seed"),
                "split": result.get("split"), "split_sha256": result.get("split_sha256"),
                "code_sha256": result.get("code_sha256"),
            }
        return analyze_log(args.content)

    def save(self, args: NoteInput) -> dict:
        if (not re.search(r"保存|记住|记录下来|存为|save|remember", self.run["prompt"], re.I)
                or re.search(r"不要保存|别保存|不用保存|不要记住|do not save|don't save",
                             self.run["prompt"], re.I)):
            raise ValueError("只有用户本轮明确要求保存时才能写入研究笔记")
        key = None
        if self.idempotent_notes:
            key = dump([self.run["id"], "save_note", args.title, args.content])
        return self.store.add_note(args.title, args.content, self.run["session_id"],
                                   idempotency_key=key)

    async def import_paper(self, args: ArxivImportInput) -> dict:
        if (not re.search(r"导入|下载|import|download", self.run["prompt"], re.I)
                or re.search(r"不要下载|不要导入|别下载|别导入|do not download|don't import",
                             self.run["prompt"], re.I)):
            raise ValueError("请先让用户明确要求下载或导入论文")
        result = await self.library.import_arxiv(args.arxiv_id)
        self.index.invalidate()
        return result

    def plan_study(self, args: StudyConfig) -> dict:
        row = self.studies.plan(args)
        ref = next((k for k, v in self.study_refs.items() if v == row["id"]), None)
        if ref is None:
            ref = f"study-{len(self.study_refs) + 1}"
            self.study_refs[ref] = row["id"]
        return {"plan_ref": ref, "plan_id": row["id"], "status": row["status"], "plan": row["plan"],
                "next": "Pass the short plan_ref as plan_id to run_study/read_study this run. "
                        "This reference resolves exactly to the full hash, without fuzzy matching."}

    def resolve_study(self, args: StudyReference) -> str:
        value = self.study_refs.get(args.plan_id, args.plan_id)
        return StudyID(plan_id=value).plan_id

    async def run_study(self, args: StudyReference) -> dict:
        prompt = self.run["prompt"]
        if (not re.search(r"运行|跑|执行|run|execute", prompt, re.I)
                or re.search(r"不要运行|不运行|别运行|只创建|do not run|don't run", prompt, re.I)):
            raise ValueError("只有用户明确要求运行研究实验时才能执行")
        row = await self.studies.run(self.resolve_study(args))
        return {"plan_id": row["id"], "status": row["status"], "reused": row["reused"],
                "next": "Use read_study to inspect measured results."}

    def read_study(self, args: StudyReference) -> dict:
        row = self.studies.read(self.resolve_study(args))
        result = row["result"]
        return {"plan_id": row["id"], "status": row["status"], "error": row["error"],
                "config": row["plan"]["config"],
                "result": {k: result[k] for k in (
                    "summary", "paired_comparisons", "fit_policy", "limitations",
                    "metrics_origin", "evidence_files")} if result else None}

    def registry(self, role: str) -> ToolRegistry:
        all_tools = [
            DomainTool("search_papers", "Search local PDFs. Translate Chinese intent into specific "
                       "English CV keywords. Return exact page chunks and citation IDs.",
                       SearchInput, self.search),
            DomainTool("read_paper", "Read a real PDF page; returns citeable chunks, no OCR.",
                       ReadInput, self.read),
            DomainTool("search_arxiv", "Search arXiv metadata; metadata is not full-paper evidence.",
                       ArxivSearchInput, self.arxiv_search),
            DomainTool("import_arxiv", "Download a public arXiv PDF only when user requests import.",
                       ArxivImportInput, self.import_paper, False),
            DomainTool("inspect_code", "Read examples/ Python source and AST symbol line ranges. "
                       "Available file: vision_ops.py; symbols: patchify, "
                       "scaled_dot_product_attention, residual_block.", CodeInput, self.code),
            DomainTool("run_experiment", "Run real CPU digits feature/classifier baseline or NumPy "
                       "attention demo. digits also produces a separate real SGD training log. "
                       "Only run when user requests an experiment.", ExperimentInput,
                       self.experiment, False),
            DomainTool("analyze_log", "Compute best epoch, generalization gap, loss changes from "
                       "JSON/CSV or a stored digits experiment_id. No inferred metrics.",
                       LogInput, self.log),
            DomainTool("save_note", "Save a research decision only if explicitly requested.",
                       NoteInput, self.save, False),
            DomainTool("list_notes", "Read saved research decisions.", Arguments,
                       lambda _: {"notes": self.store.notes()}),
            DomainTool("plan_study", "Create an immutable digits robustness plan comparing pixel "
                       "and PCA SVMs across seeds and clean/noise/occlusion. Does not execute. "
                       "Returns plan_id binding config, code and data fingerprints.",
                       StudyConfig, self.plan_study, False),
            DomainTool("run_study", "Execute an existing plan_id only on explicit user request. "
                       "Reuses completed results. C chosen on validation; test reserved.",
                       StudyReference, self.run_study, False),
            DomainTool("read_study", "Read measured study results and evidence hashes by plan_id. "
                       "Never infer success from plan creation alone.", StudyReference, self.read_study),
            DomainTool("read_artifact", "Read this session's stored tool output by SHA256 ID. "
                       "Use query to locate an exact substring or offset/limit to paginate.",
                       ArtifactInput, self.artifacts.read),
        ]
        allowed = {
            "paper": {"search_papers", "read_paper", "search_arxiv", "import_arxiv"},
            "code": {"inspect_code", "search_papers", "read_paper"},
            "experiment": {"run_experiment", "analyze_log", "inspect_code",
                           "search_papers", "read_paper", "plan_study", "run_study", "read_study"},
        }[role] | {"save_note", "list_notes", "read_artifact"}
        registry = ToolRegistry()
        for tool in all_tools:
            if tool.name in allowed:
                registry.register(tool)
        return registry

    async def arxiv_search(self, args: ArxivSearchInput) -> dict:
        return {"results": await search_arxiv(args.query, args.limit)}
