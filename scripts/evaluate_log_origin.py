"""Targeted real-model regression for stored-log provenance; retain original checks on rescore."""
import argparse
import json
import time

import httpx
from evaluate import ROOT, get, post, report_base, summarize, wait_run, write


def checks_for(row, experiment_id):
    run, events = row["run"], row["events"]
    outputs = [json.loads(e["data"]["output"]) for e in events if e["type"] == "tool_end"
               and e["data"]["name"] == "analyze_log" and not e["data"]["is_error"]]
    result = outputs[-1] if outputs else {}
    answer = run["answer"]
    return {
        "completed": run["status"] == "completed",
        "stored_origin": "stored experiment log" in result.get("metrics_origin", ""),
        "source_fields": result.get("seed") == 42
        and result.get("experiment_id") == experiment_id and bool(result.get("code_sha256"))
        and result.get("split") == {"train": 1077, "validation": 360, "test": 360},
        "answer_source": any(t in answer for t in (
            "本应用", "应用内", "应用自身", "this application's stored experiment log",
        )) and "用户提供" not in answer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--rescore", action="store_true", help="重算已保存回答，保留原判据，不调用模型")
    args = parser.parse_args()
    directory = ROOT / "data" / "reports"
    baseline = json.loads((directory / "real-model.json").read_text())
    experiment_id = baseline["experiment_id"]
    target = directory / "log-provenance.json"
    if args.rescore:
        report = json.loads(target.read_text())
        row = report["cases"][0]
        row.setdefault("original_checks", row["checks"].copy())
        row["rescoring_note"] = (
            "初版判据仅识别中文来源措辞，误报了正确引用工具英文来源的回答。"
            "将工具返回的准确英文来源加入允许表达，对同一份原始回答复算，未重新抽样。"
        )
    else:
        with httpx.Client(base_url=args.base_url, timeout=60) as client:
            model = get(client, "/api/health")["model"]
            if not model["available"] or model["mock"]:
                raise SystemExit("需要真实模型")
            question = (f"请使用 analyze_log 分析实验 {experiment_id} 的真实SGD日志，报告最佳epoch、"
                        "泛化差距，并准确说明日志来源、种子、数据拆分和代码哈希。")
            session = post(client, "/api/sessions", json={"title": "评测 · 日志来源"})
            start = time.monotonic()
            run = post(client, "/api/runs", json={
                "session_id": session["id"], "prompt": question, "mode": "experiment"})
            run = wait_run(client, run["id"])
            row = {"id": "log-origin", "question": question, "run": run,
                   "events": get(client, f"/api/runs/{run['id']}/trace")["events"],
                   "elapsed_seconds": round(time.monotonic() - start, 3), "answer": run["answer"],
                   "expected": "真实模型准确说明存储实验日志的来源及实验ID/seed/split/code hash。"}
            report = report_base("real_model", "日志溯源 · 修复回归")
            report["model"] = model["model"]
            report["cases"] = [row]
    row["checks"] = checks_for(row, experiment_id)
    row["passed"] = all(row["checks"].values())
    row["reason"] = row.get("rescoring_note", "单独验证日志来源修复；保留原始11题结果。")
    summarize(report, "针对性真实模型回归，验证存储实验日志来源；不替换原始11题。")
    write(target, report)
    print(json.dumps({"summary": report["summary"], "checks": row["checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
