"""PDF ingestion and a narrowly scoped arXiv connector."""
from __future__ import annotations

import asyncio
import hashlib
import io
import re
import threading
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urljoin, urlsplit

import httpx
from pypdf import PdfReader

from backend.store import Store, now

MAX_PDF_BYTES = 25 * 1024 * 1024
MAX_PAGES = 100
ARXIV_ID = re.compile(r"^\d{4}\.\d{4,5}(v\d+)?$")
CATALOG = [
    {"id": "resnet", "arxiv_id": "1512.03385", "year": 2015,
     "title": "Deep Residual Learning for Image Recognition",
     "authors": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun", "tag": "ResNet"},
    {"id": "vit", "arxiv_id": "2010.11929", "year": 2020,
     "title": "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
     "authors": "Alexey Dosovitskiy et al.", "tag": "ViT"},
    {"id": "clip", "arxiv_id": "2103.00020", "year": 2021,
     "title": "Learning Transferable Visual Models From Natural Language Supervision",
     "authors": "Alec Radford et al.", "tag": "CLIP"},
    {"id": "detr", "arxiv_id": "2005.12872", "year": 2020,
     "title": "End-to-End Object Detection with Transformers",
     "authors": "Nicolas Carion et al.", "tag": "DETR"},
]


def validate_arxiv_id(value: str) -> str:
    value = value.strip()
    if not ARXIV_ID.fullmatch(value):
        raise ValueError("请提供合法 arXiv ID，例如 2010.11929")
    return value


def validate_arxiv_url(url: str) -> None:
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname not in {"arxiv.org", "export.arxiv.org"}
            or parsed.port not in {None, 443} or parsed.username or parsed.password):
        raise ValueError("只允许从 arxiv.org 的 HTTPS 地址读取论文")


async def fetch_arxiv(url: str, max_bytes: int = MAX_PDF_BYTES) -> bytes:
    headers = {"User-Agent": "VisionScholar/0.1 (local academic reading assistant)"}
    async with httpx.AsyncClient(timeout=httpx.Timeout(45, connect=10), headers=headers) as client:
        for _ in range(4):
            validate_arxiv_url(url)
            async with client.stream("GET", url, follow_redirects=False) as response:
                if response.is_redirect:
                    target = response.headers.get("location")
                    if not target:
                        raise ValueError("arXiv 返回了无目标重定向")
                    url = urljoin(url, target)
                    continue
                response.raise_for_status()
                if int(response.headers.get("content-length", "0")) > max_bytes:
                    raise ValueError("论文超过大小限制")
                result = bytearray()
                async for part in response.aiter_bytes():
                    result.extend(part)
                    if len(result) > max_bytes:
                        raise ValueError("论文超过大小限制")
                return bytes(result)
    raise ValueError("arXiv 重定向次数过多")


def parse_pdf(data: bytes) -> tuple[list[str], str]:
    if len(data) > MAX_PDF_BYTES:
        raise ValueError("PDF 不能超过 25 MB")
    if not data.startswith(b"%PDF-"):
        raise ValueError("文件不是有效 PDF")
    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise ValueError("暂不支持加密 PDF")
        if not 1 <= len(reader.pages) <= MAX_PAGES:
            raise ValueError("PDF 需为 1–100 页")
        pages = [
            re.sub(r"[ \t]+", " ", page.extract_text(extraction_mode="plain") or "").strip()
            for page in reader.pages
        ]
        title = str((reader.metadata or {}).get("/Title", "")).strip()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF 解析失败，请检查文件是否损坏") from exc
    if sum(len(text) for text in pages) < 150:
        raise ValueError("PDF 中没有足够可检索文字；扫描件请先做 OCR")
    if sum(len(text) for text in pages) > 2_000_000:
        raise ValueError("PDF 文字量超过本地索引限制")
    return pages, title[:250]


def split_pages(paper_id: str, pages: list[str]) -> list[dict]:
    chunks = []
    section = "正文"
    for page, text in enumerate(pages, 1):
        heading = re.search(
            r"(?:^|\n)\s*((?:\d+(?:\.\d+)*\.?\s+)[A-Z][^\n]{3,80}|Abstract|References)\s*(?:\n|$)",
            text,
        )
        if heading:
            section = heading.group(1).strip()
        # Whitespace spans preserve exact original substrings and stable page provenance.
        spans = list(re.finditer(r"\S+", text))
        for ordinal, start in enumerate(range(0, len(spans), 160), 1):
            end = min(start + 200, len(spans))
            content = text[spans[start].start():spans[end - 1].end()]
            chunks.append({
                "id": f"{paper_id}:p{page}:c{ordinal}", "paper_id": paper_id,
                "page": page, "ordinal": ordinal, "text": content, "section": section,
            })
            if end == len(spans):
                break
    return chunks


class PaperLibrary:
    def __init__(self, store: Store, paper_dir: Path):
        self.store = store
        self.paper_dir = paper_dir
        paper_dir.mkdir(parents=True, exist_ok=True)
        self._import_lock = asyncio.Lock()
        self._ingest_lock = threading.Lock()

    def ingest(self, data: bytes, *, title: str = "", paper_id: str = "",
               authors: str = "", year: int | None = None, source_url: str = "local-upload") -> dict:
        with self._ingest_lock:
            return self._ingest(data, title=title, paper_id=paper_id, authors=authors,
                                year=year, source_url=source_url)

    def _ingest(self, data: bytes, *, title: str, paper_id: str, authors: str,
                year: int | None, source_url: str) -> dict:
        sha = hashlib.sha256(data).hexdigest()
        existing = self.store.one("SELECT * FROM papers WHERE sha256=?", (sha,))
        if existing:
            return existing
        pages, embedded_title = parse_pdf(data)
        paper_id = paper_id or f"paper-{sha[:12]}"
        if not re.fullmatch(r"[a-z0-9.-]{1,64}", paper_id):
            raise ValueError("非法论文标识")
        if self.store.one("SELECT id FROM papers WHERE id=?", (paper_id,)):
            paper_id = f"{paper_id[:48]}-{sha[:8]}"
        chunks = split_pages(paper_id, pages)
        file_name = f"{sha}.pdf"
        target = self.paper_dir / file_name
        if not target.exists():
            temporary = self.paper_dir / f"{sha}.part"
            temporary.write_bytes(data)
            temporary.replace(target)
        paper = {
            "id": paper_id, "title": (title or embedded_title or "未命名论文")[:250],
            "authors": authors[:500], "year": year, "source_url": source_url,
            "sha256": sha, "file_name": file_name, "page_count": len(pages),
            "chunk_count": len(chunks), "created_at": now(),
        }
        return self.store.add_paper(paper, pages, chunks)

    async def import_arxiv(self, arxiv_id: str) -> dict:
        arxiv_id = validate_arxiv_id(arxiv_id)
        async with self._import_lock:
            source = f"https://arxiv.org/pdf/{arxiv_id}"
            existing = self.store.one("SELECT * FROM papers WHERE source_url=?", (source,))
            if existing:
                return existing
            known = next((p for p in CATALOG if p["arxiv_id"] == arxiv_id), None)
            data = await fetch_arxiv(source)
            return await asyncio.to_thread(
                self.ingest, data,
                paper_id=known["id"] if known else f"arxiv-{arxiv_id}",
                title=known["title"] if known else "",
                authors=known["authors"] if known else "",
                year=known["year"] if known else None,
                source_url=source,
            )

    def page(self, paper_id: str, number: int) -> dict:
        paper = self.store.paper(paper_id)
        if not 1 <= number <= paper["page_count"]:
            raise ValueError(f"页码超出范围，论文共 {paper['page_count']} 页")
        page = self.store.one("SELECT text FROM pages WHERE paper_id=? AND number=?",
                              (paper_id, number))
        chunks = self.store.rows(
            "SELECT * FROM chunks WHERE paper_id=? AND page=? ORDER BY ordinal",
            (paper_id, number),
        )
        return {"paper": paper, "page": number, "text": page["text"], "chunks": chunks}

    async def seed(self) -> list[dict]:
        result = []
        for entry in CATALOG:
            existing = self.store.one("SELECT * FROM papers WHERE id=?", (entry["id"],))
            if existing:
                result.append(existing)
                continue
            # Also consumes PDFs downloaded explicitly during initial project setup.
            cached = self.paper_dir / f"{entry['id']}.pdf"
            if cached.exists():
                paper = await asyncio.to_thread(
                    self.ingest, cached.read_bytes(), paper_id=entry["id"], title=entry["title"],
                    authors=entry["authors"], year=entry["year"],
                    source_url=f"https://arxiv.org/pdf/{entry['arxiv_id']}",
                )
            else:
                paper = await self.import_arxiv(entry["arxiv_id"])
            result.append(paper)
        return result


async def search_arxiv(query: str, limit: int = 5) -> list[dict]:
    if not query.strip() or len(query) > 300 or not 1 <= limit <= 10:
        raise ValueError("搜索词不能为空且需小于 300 字符，数量为 1–10")
    # URL encoding is handled by httpx; user input is never treated as a URL.
    params = {"search_query": f"all:{query}", "start": 0, "max_results": limit,
              "sortBy": "relevance"}
    url = str(httpx.URL("https://export.arxiv.org/api/query", params=params))
    payload = await fetch_arxiv(url, max_bytes=2 * 1024 * 1024)
    if b"<!DOCTYPE" in payload.upper() or b"<!ENTITY" in payload.upper():
        raise ValueError("arXiv 返回了不支持的 XML")
    root = ET.fromstring(payload)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    results = []
    for entry in root.findall("a:entry", ns):
        raw_id = entry.findtext("a:id", "", ns).rsplit("/", 1)[-1]
        if not ARXIV_ID.fullmatch(raw_id):
            continue
        results.append({
            "arxiv_id": raw_id,
            "title": " ".join(entry.findtext("a:title", "", ns).split()),
            "summary": " ".join(entry.findtext("a:summary", "", ns).split())[:1400],
            "authors": ", ".join(a.findtext("a:name", "", ns)
                                for a in entry.findall("a:author", ns)),
            "year": entry.findtext("a:published", "", ns)[:4],
            "url": f"https://arxiv.org/abs/{raw_id}",
        })
    return results
