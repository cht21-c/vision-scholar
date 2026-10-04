"""Reproducible engineering cases over the running HTTP service, never a second worker."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = "9b2efd795c6aa09f88b0c257d269a9e518da6ae7"
RETRIEVAL = [
    ("r1", "resnet", "residual learning identity shortcut mapping", [2, 3]),
    ("r2", "resnet", "degradation deeper network higher training error", [1, 2, 3]),
    ("r3", "vit", "flattened patches linear projection patch embeddings", [3]),
    ("r4", "vit", "learnable position embeddings class token", [3]),
    ("r5", "clip", "symmetric cross entropy loss image text pairs", [4, 5]),
    ("r6", "clip", "zero shot classifier text encoder class names", [2, 6, 7]),
    ("r7", "detr", "Hungarian algorithm bipartite matching loss", [5]),
    ("r8", "detr", "object queries learned positional embeddings decoder", [6, 7]),
]
REAL = [
    {"id": "resnet", "paper_ids": ["resnet"],
     "question": "ResNet 为什么把目标写成 F(x)+x？解释恒等快捷连接并引用原文。",
     "papers": ["resnet"], "tools": ["search_papers"],
     "expected": "完成；实际检索 ResNet；引用至少一个本轮读取的合法 ResNet 片段。"},
    {"id": "vit", "paper_ids": ["vit"],
     "question": "ViT 如何把图像变成 token？给出 patch 数量公式，解释 class token 和位置编码，引用原文。",
     "papers": ["vit"], "tools": ["search_papers"],
     "expected": "完成；实际检索 ViT；引用至少一个本轮读取的合法 ViT 片段。"},
    {"id": "clip", "paper_ids": ["clip"],
     "question": "CLIP 的一个 batch 怎样构造正负样本，symmetric cross entropy 优化什么？请查原文并引用。",
     "papers": ["clip"], "tools": ["search_papers"],
     "expected": "完成；实际检索 CLIP；引用至少一个本轮读取的合法 CLIP 片段。"},
    {"id": "detr", "paper_ids": ["detr"],
     "question": "DETR 为什么需要二分图匹配，Hungarian loss 有哪些项？从论文给出证据。",
     "papers": ["detr"], "tools": ["search_papers"],
     "expected": "完成；实际检索 DETR；引用至少一个本轮读取的合法 DETR 片段。"},
    {"id": "compare", "paper_ids": ["vit", "clip"],
     "question": "比较 ViT 和 CLIP 的训练目标与监督信号，分别查找两篇原文并引用，不要只凭常识。",
     "papers": ["vit", "clip"], "tools": ["search_papers"],
     "expected": "完成；同时引用本轮读取的 ViT 和 CLIP 合法片段。"},
    {"id": "unknown", "paper_ids": ["resnet", "vit", "clip", "detr"],
     "question": "只根据这四篇论文，能确认 2026 年 ImageNet 全球最高准确率是多少、由谁取得吗？请检索核对，不能确定就明确说。",
     "tools": ["search_papers"], "refusal": True,
     "expected": "完成；检索后包含明确的无法确认表达；不要求无引用，语义拒答由人工另审。"},
    {"id": "code", "mode": "code",
     "question": "查看 examples/vision_ops.py 的 patchify 函数，解释 reshape 和 transpose，附真实代码行号。",
     "tools": ["inspect_code"], "code": True,
     "expected": "完成；调用 inspect_code；包含本轮读取的合法代码行号引用。"},
    {"id": "digits", "mode": "experiment",
     "question": "请运行 seed=42、PCA维度24 的 digits 实验，比较原始像素与PCA方案，说明独立测试集和局限。",
     "tools": ["run_experiment"], "experiment": True,
     "expected": "完成；真实 run_experiment 返回 digits 数值；拆分1077/360/360；回答含实验ID和两个实测准确率。"},
    {"id": "log", "mode": "experiment",
     "question": "使用 analyze_log 分析实验 {experiment_id} 的真实 SGD 日志，列出最佳准确率 epoch、最佳 loss epoch、末轮泛化差距，说明结论边界。",
     "tools": ["analyze_log"], "log": True,
     "expected": "完成；analyze_log 返回16轮真实日志；计算结果与独立HTTP日志分析一致。"},
    {"id": "memory-save", "session": "memory",
     "question": "请记住并保存研究决策：本项目先用 seed=42、PCA维度24 的 digits 教学基线验证流程，不声称复现 ViT。",
     "tools": ["save_note"],
     "expected": "完成；当前明确授权下调用 save_note，并保存含 seed=42/PCA24 的笔记。"},
    {"id": "memory-recall", "session": "memory",
     "question": "回顾我们刚才保存的研究决策：先做什么、随机种子和PCA维度是多少？请调用 list_notes 核对。",
     "tools": ["list_notes"], "memory": True,
     "expected": "同一会话第二轮完成；调用 list_notes；回答包含42、24和digits。"},
]


def stamp():
    return datetime.now(UTC).isoformat()


def write(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def report_base(kind, name):
    return {"kind": kind, "name": name, "created_at": stamp(), "cases": [],
            "upstream_commit": UPSTREAM,
            "suite_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evaluation_basis": "项目编写的工程回归用例；不是外部人评金标，不衡量总体语义质量。"}


def summarize(report, description):
    cases = report["cases"]
    report["summary"] = {"total": len(cases), "passed": sum(c["passed"] for c in cases),
                         "description": description}
    elapsed = [c["elapsed_seconds"] for c in cases if "elapsed_seconds" in c]
    if elapsed:
        report["timing"] = {"median_seconds": round(statistics.median(elapsed), 3),
                            "max_seconds": max(elapsed)}


def get(client, path, **kwargs):
    response = client.get(path, **kwargs)
    response.raise_for_status()
    return response.json()


def post(client, path, **kwargs):
    response = client.post(path, **kwargs)
    response.raise_for_status()
    return response.json()


def retrieval(client, output):
    report = report_base("retrieval", "检索回归 · 8 个固定问题")
    report["corpus"] = get(client, "/api/papers")
    report["reference_method"] = "根据下载版本PDF原文预先指定方法所在物理页；判断top6是否命中其中一页，不是段落级相关性人评。"
    reciprocal = []
    for cid, paper, query, pages in RETRIEVAL:
        start = time.monotonic()
        hits = get(client, "/api/search", params={"q": query, "paper_id": paper, "limit": 6})["results"]
        rank = next((i for i, h in enumerate(hits, 1) if h["page"] in pages), None)
        reciprocal.append(1 / rank if rank else 0)
        report["cases"].append({
            "id": cid, "question": f"{paper}: {query}", "passed": rank is not None,
            "expected": f"限定论文 {paper} 的 top6 命中 PDF 页 {pages}",
            "expected_pages": pages, "first_relevant_rank": rank,
            "citations": [h["id"] for h in hits], "hits": hits,
            "elapsed_seconds": round(time.monotonic() - start, 3),
            "reason": f"首个命中排名：{rank or '未命中'}",
        })
    summarize(report, "固定8题的页级 Hit@6；限定论文检索，不能外推为中文开放域问答准确率。")
    report["mrr_at_6"] = sum(reciprocal) / len(reciprocal)
    write(output / "retrieval.json", report)
    print(json.dumps({"retrieval": report["summary"], "mrr": report["mrr_at_6"]}, ensure_ascii=False))


def wait_run(client, run_id):
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        run = get(client, f"/api/runs/{run_id}")
        if run["status"] not in {"running", "queued"}:
            return run
        time.sleep(0.5)
    post(client, f"/api/runs/{run_id}/cancel")
    raise TimeoutError("HTTP评测等待超过240秒，已取消该任务")


def real(client, output, resume):
    model = get(client, "/api/health")["model"]
    if not model["available"] or model["mock"]:
        raise SystemExit("real suite 必须连接真实模型；拒绝用 mock 替代。")
    target = output / "real-model.json"
    report = json.loads(target.read_text()) if resume and target.exists() else report_base(
        "real_model", "真实模型 · 11 项任务链路")
    report["model"] = model["model"]
    report["provider"] = model["provider"]
    report["corpus"] = get(client, "/api/papers")
    done = {c["id"] for c in report["cases"]}
    sessions = report.setdefault("sessions", {})
    experiment_id = report.get("experiment_id")
    for case in REAL:
        if case["id"] in done:
            continue
        start = time.monotonic()
        key = case.get("session", case["id"])
        if key not in sessions:
            sessions[key] = post(client, "/api/sessions", json={"title": f"评测 · {key}"})["id"]
        question = case["question"].replace("{experiment_id}", experiment_id or "不存在")
        print(f"[{len(report['cases'])+1}/{len(REAL)}] {case['id']} …", flush=True)
        row = {**case, "question": question, "semantic_review": "pending_manual_review"}
        try:
            run = post(client, "/api/runs", json={
                "session_id": sessions[key], "prompt": question,
                "paper_ids": case.get("paper_ids", []), "mode": case.get("mode", "auto"),
            })
            run = wait_run(client, run["id"])
            events = get(client, f"/api/runs/{run['id']}/trace")["events"]
            completed = [e["data"] for e in events if e["type"] == "tool_end"]
            successful = {d["name"] for d in completed if not d.get("is_error")}
            outputs = {}
            for d in completed:
                if not d.get("is_error"):
                    try:
                        outputs.setdefault(d["name"], []).append(json.loads(d["output"]))
                    except (ValueError, KeyError):
                        pass
            checks = {
                "run_completed": run["status"] == "completed",
                "required_tools": all(t in successful for t in case["tools"]),
                "no_invalid_citations": not run["verification"].get("invalid"),
            }
            answer = run["answer"]
            cited_papers = {c.get("paper_id") for c in run["citations"]}
            if case.get("papers"):
                checks["paper_coverage"] = all(p in cited_papers for p in case["papers"])
            if case.get("code"):
                checks["source_citation"] = any(c["id"].startswith("code:") for c in run["citations"])
            if case.get("refusal"):
                checks["explicit_uncertainty_marker"] = bool(re.search(
                    r"无法确认|不能确认|无法确定|不能确定|无法.*(?:给出|回答|判断)|不足以|没有.*(?:证据|信息)", answer))
            if case.get("experiment"):
                experiment = next((e for e in outputs.get("run_experiment", [])
                                   if e.get("kind") == "digits"), {})
                experiment_id = experiment.get("id")
                report["experiment_id"] = experiment_id
                result = experiment.get("result", {})
                checks["real_split"] = result.get("split") == {
                    "train": 1077, "validation": 360, "test": 360}
                checks["id_in_answer"] = bool(experiment_id and experiment_id in answer)
                checks["accuracies_in_answer"] = bool(result.get("results")) and all(
                    f"{r['test_accuracy']*100:.2f}" in answer for r in result.get("results", []))
                row["experiment"] = experiment
            if case.get("log"):
                log = (outputs.get("analyze_log") or [{}])[-1]
                experiment = get(client, f"/api/experiments/{experiment_id}")
                reference = post(client, "/api/logs/analyze", json={
                    "content": json.dumps(experiment["result"]["training_log"])})
                checks["computed_log_matches"] = log.get("epochs") == 16 and all(
                    log.get(k) == reference[k] for k in (
                        "best_accuracy_epoch", "best_loss_epoch", "last_generalization_gap"))
                row["reference_log_metrics"] = reference
            if case.get("memory"):
                checks["decision_recalled"] = all(t in answer.lower() for t in ("42", "24", "digits"))
            if case["id"] == "memory-save":
                notes = outputs.get("save_note", [])
                checks["decision_saved"] = any(
                    all(t in json.dumps(n).lower() for t in ("42", "24", "digits")) for n in notes)
            row.update({"passed": all(checks.values()), "checks": checks,
                        "answer": answer, "citations": [c["id"] for c in run["citations"]],
                        "run": run, "events": events, "tool_count": len(completed),
                        "reason": "; ".join(k for k, v in checks.items() if not v) or
                        "预设工程判据全部满足；回答语义仍需独立复核。"})
        except (httpx.HTTPError, ValueError, TimeoutError, KeyError) as exc:
            row.update({"passed": False, "reason": f"{type(exc).__name__}: 评测未完成，检查服务日志"})
        row["elapsed_seconds"] = round(time.monotonic() - start, 3)
        report["cases"].append(row)
        summarize(report, "真实模型端到端运行；判据仅为任务终态、工具调用、引用覆盖、计算值/记忆回读。非模型准确率。")
        write(target, report)
        print(f"  {'PASS' if row['passed'] else 'FAIL'} {row['elapsed_seconds']}s · {row['reason']}", flush=True)
    print(json.dumps(report["summary"], ensure_ascii=False))


def engineering(output, junit):
    root = ET.parse(junit).getroot()
    all_cases = []
    for i, node in enumerate(root.iter("testcase"), 1):
        issue = next((node.find(tag) for tag in ("failure", "error", "skipped")
                      if node.find(tag) is not None), None)
        all_cases.append({
            "id": f"test-{i}", "name": f"{node.get('classname')}.{node.get('name')}",
            "passed": issue is None, "elapsed_seconds": float(node.get("time", "0")),
            "expected": "pytest 断言通过，详见 tests/ 对应测试源码。",
            "reason": "" if issue is None else issue.get("message", issue.tag),
        })
    report = report_base("engineering", "工程回归 · pytest")
    report["cases"] = all_cases
    report["source_junit"] = junit.name
    summarize(report, "覆盖工具边界、HTTP、持久化、SSE及真实OpenHarness循环；部分用脚本mock替代模型/网络，不衡量模型能力。")
    write(output / "engineering.json", report)
    mock = report_base("mock", "脚本 Mock · Agent 回归")
    mock["cases"] = [c for c in all_cases if c["name"].startswith("tests.test_agent.")]
    summarize(mock, "工程回归的子集，不可与总数相加；脚本驱动真实OpenHarness，验证工具调用与恢复，不代表真实模型质量。")
    write(output / "mock.json", mock)
    print(json.dumps({"engineering": report["summary"], "mock_subset": mock["summary"]}, ensure_ascii=False))


def browser(output):
    source = json.loads((ROOT / "data" / "playwright-results.json").read_text())
    report = report_base("engineering", "浏览器验收 · 桌面与手机")

    def visit(suite):
        for spec in suite.get("specs", []):
            for test in spec.get("tests", []):
                results = test.get("results", [])
                result = results[-1] if results else {}
                report["cases"].append({
                    "id": f"{spec['id']}-{test['projectName']}",
                    "name": f"{test['projectName']}: {spec['title']}",
                    "passed": test["status"] == "expected" and result.get("status") == "passed",
                    "elapsed_seconds": result.get("duration", 0) / 1000,
                    "expected": "真实HTTP服务与浏览器操作；桌面1440×1000 / 手机390×844，无文档横向溢出。",
                    "reason": json.dumps(result.get("errors"), ensure_ascii=False)
                    if result.get("errors") else "交互断言通过；截图保存在 data/screenshots/。",
                })
        for child in suite.get("suites", []):
            visit(child)

    for suite in source["suites"]:
        visit(suite)
    report["source_stats"] = source["stats"]
    summarize(report, "两条端到端流程在两个尺寸执行；没有mock后端响应。截图另由AI目视检查，非人工设计验收。")
    write(output / "browser.json", report)
    print(json.dumps(report["summary"], ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["retrieval", "real", "engineering", "browser"], required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reports")
    parser.add_argument("--junit", type=Path, default=ROOT / "data" / "engineering-tests.xml")
    parser.add_argument("--resume", action="store_true", help="仅跳过已有条目，失败也保留不自动洗掉")
    args = parser.parse_args()
    if args.suite == "engineering":
        engineering(args.output, args.junit)
        return
    if args.suite == "browser":
        browser(args.output)
        return
    with httpx.Client(base_url=args.base_url, timeout=60) as client:
        if args.suite == "retrieval":
            retrieval(client, args.output)
        else:
            real(client, args.output, args.resume)


if __name__ == "__main__":
    main()
