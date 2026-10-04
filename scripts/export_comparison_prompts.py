"""Export the actual prompts and complete tool schemas, without API configuration."""
import json

from backend.agent import SYSTEM_PROMPT
from scripts.compare_harnesses import NORMAL, OUTPUT, ROOT
from scripts.context_cases import SYSTEM, context_cases
from scripts.review_comparison import RUBRIC


def fenced(text, language="text"):
    return f"```{language}\n{text.rstrip()}\n```\n"


def main():
    sections = [
        "# 对照实验：完整Prompt、Schema与输入\n\n"
        "本文件由`scripts/export_comparison_prompts.py`从实际代码与保留trace导出。"
        "不含密钥或endpoint配置。原始请求还保存于`data/comparison/trials/`。\n\n"
        "## 1. 三版共有系统Prompt原文\n\n"
        "占位符role来自共享规则路由；catalog来自当前论文库；paper_ids来自用户范围；"
        "max_turns主实验为8。V0/V1使用相同模板，V2仅按下文附加决策原文。\n",
        fenced(SYSTEM_PROMPT),
        "## 2. 正常任务的完整用户输入\n",
    ]
    for case in NORMAL:
        sections.append(f"### {case['id']}\n\n")
        for i, prompt in enumerate(case["prompts"], 1):
            sections.append(f"用户轮{i}：\n\n" + fenced(prompt))
    sections += [
        "## 3. 合成上下文的完整系统Prompt\n", fenced(SYSTEM),
        "## 4. 合成上下文构造原文\n\n"
        "以下是完整生成器，包含固定标识算法、旧值、后续修正、填充内容、消息顺序、"
        "压缩阈值和三个追问。不是从真实用户聊天抽取。\n",
        fenced((ROOT / "scripts" / "context_cases.py").read_text(), "python"),
    ]
    cases = context_cases()
    sections.append("具体阈值与事实值：\n\n" + fenced(json.dumps([
        {k: c[k] for k in ("id", "facts", "superseded_pilot", "compact_threshold",
                          "history_estimated_tokens", "prompts")} for c in cases
    ], ensure_ascii=False, indent=2), "json"))
    protected = json.loads((OUTPUT / "trials" / "context-buried_correction-upgraded-r1.json").read_text())
    system = protected["calls"][0]["request"]["messages"][0]["content"]
    assert system.startswith(SYSTEM)
    sections += [
        "## 5. V2实际附加的决策原文\n\n"
        "下面是埋藏修正r1第一次模型请求中追加在共有系统Prompt之后的完整内容。"
        "它来自用户原句与来源消息SHA256，按时间排列；不是模型生成摘要。\n",
        fenced(system[len(SYSTEM):]),
        "## 6. 上游完整摘要请求实例\n\n"
        "下面逐条列出`context-full_summary-openharness-r1`中首个无tools请求的全部"
        "messages（包含实际历史与摘要指令）。该调用计入总usage，输出不直接展示给用户。\n",
    ]
    full = json.loads((OUTPUT / "trials" / "context-full_summary-openharness-r1.json").read_text())
    compact = next(c for c in full["calls"] if not c["request"].get("tools"))
    sections.append(fenced(json.dumps(compact["request"]["messages"],
                                     ensure_ascii=False, indent=2), "json"))
    sections += [
        "## 7. 匿名AI语义复核完整Prompt\n\n"
        "每题每版首次重复，共30条；隐藏运行版本和工程结果，输入完整回答与工具返回。"
        "相同模型复核不等于独立模型评审，更不是人评金标。完整输入输出在"
        "`data/comparison/semantic-review/normal-*.json`。\n", fenced(RUBRIC),
        "## 8. 完整工具Schema\n\n"
        "下列Schema直接合并自主实验实际发送的tools字段。角色只获得允许子集；"
        "运行循环从schema调用相同handler，因此不能把schema差异算成harness收益。\n",
    ]
    schemas = {}
    for path in sorted((OUTPUT / "trials").glob("normal-*.json")):
        trial = json.loads(path.read_text())
        for call in trial["calls"]:
            for tool in call["request"].get("tools", []):
                schemas[tool["function"]["name"]] = tool
    assert len(schemas) == 9
    sections.append(fenced(json.dumps(list(schemas.values()), ensure_ascii=False, indent=2), "json"))
    path = ROOT / "docs" / "comparison-prompts.md"
    path.write_text("\n".join(sections))
    print(path, path.stat().st_size, "bytes; 9 tool schemas")


if __name__ == "__main__":
    main()
