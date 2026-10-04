# 验证方法、真实结果与边界

本文件保留首版V1的模型与评测记录。最新三版实现、59项工程测试、90次正常任务和36次合成上下文对照见[Harness比较报告](harness-comparison.md)，两轮使用不同模型，不合并计算改进率。

验证日期：2026-10-03。本次在 macOS、本地单 worker、Python 3.12 / Node 24 下运行。论文下载版本与 SHA256 记录在报告的 `corpus` 中；模型是 `modelhub/gpt-5.5-2026-04-24`，通过 Anthropic 兼容接口实际调用，未使用脚本替代。

## 结果总览

| 类别 | 结果 | 证据 | 结果实际说明什么 |
|---|---:|---|---|
| pytest 工程回归 | **38 / 38**，9.15秒 | `data/test-run.log`、`engineering-tests.xml`、`reports/engineering.json` | 工具边界、持久化、异常处理等断言通过 |
| 脚本 Mock Agent 测试 | **6 / 6** | `reports/mock.json`、`tests/test_agent.py` | 真正 OpenHarness 工具循环与恢复可测；属于38项的子集 |
| 固定检索 | **8 / 8 Hit@6**，MRR@6=1.0 | `reports/retrieval.json` | 限定论文检索均命中预设方法所在页 |
| 真实模型任务 | **11 / 11** | `reports/real-model.json` | 完成指定工具链、引用覆盖、指标/记忆回读判据 |
| 日志来源追加回归 | **1 / 1** | `reports/log-provenance.json` | 工具和模型正确表达本应用生成日志的来源 |
| 浏览器端到端 | **4 / 4**，16.2秒 | `playwright-results.json`、`reports/browser.json` | 2条流程 × 桌面/手机尺寸，真实后端，交互断言通过 |

这些数字不是业务准确率，不能相加成一个“综合正确率”。没有外部标注员、人评0–4分、独立人评金标或线上用户效果数据。

## 评估依据从哪里来

**检索预设页：** 阅读实际下载的 PDF 文本后，由本项目编写8个固定问题，并指定方法所在物理页。查询和目标页均保存在 `scripts/evaluate.py::RETRIEVAL`。top6 命中任意一个预设页算一次 hit；计算首个命中的倒数排名作为 MRR。题目限定 paper ID、主要用英文术语，结果不能代表不限定范围的复杂中文问题，也不衡量每个片段是否相关。

**真实模型判据：** 问题与判据在执行前写入 `REAL`。自动检查 run 终态、指定工具是否成功、引用是否在当前 evidence 中、指定论文是否都有引用、实验指标/拆分是否匹配、记忆是否回读。库外问题只检查明确不确定表达，仍需看全文是否真的拒绝编造。

**语义复核：** 本次助手读取了全部11项回答，核对论文核心概念、真实工具结果和实验数字；这是 AI 辅助复核，不是独立人工评审。原始报告的 `semantic_review=pending_manual_review` 保留，避免将 AI 判断冒称人评金标。需要进一步证明产品质量，应增加独立编写、未参与调试的测试题及人工标注。

## 真实模型逐项结果

| 问题 | 工具完成数 | 耗时（秒） | 可回溯结果 / 检查 |
|---|---:|---:|---|
| ResNet 残差与恒等连接 | 1 | 10.299 | ResNet p2/p3，解释 F(x)=H(x)−x |
| ViT patch / class token / 位置编码 | 2 | 16.487 | ViT p3/p4等，N=HW/P²，加class token后N+1 |
| CLIP 正负样本与对称损失 | 1 | 53.711 | CLIP p4/p5，N个正对、N²−N个负对、双向CE |
| DETR 匹配与 Hungarian loss | 3 | 23.578 | DETR p2/p5/p6，区分匹配代价与训练损失 |
| ViT 与 CLIP 的目标/监督对比 | 5 | 23.052 | 两篇均有引用，类别标签与图文配对分开 |
| 根据旧论文询问2026最高准确率 | 7 | 25.072 | 明确不能确认，历史结果未冒充2026榜单 |
| patchify 源码分析 | 1 | 9.892 | 实际 AST/源码 L7–L15，解释维度变换 |
| 运行 digits 特征对比 | 1 | 11.848 | 真实实验ID、正确拆分和两项准确率 |
| 分析存储的 SGD 日志 | 1 | 9.288 | 16轮指标与 Python 参考计算一致 |
| 保存研究决策 | 1 | 4.666 | 明确授权下持久化 note |
| 同会话回读决策 | 1 | 4.146 | list_notes 回读 seed42、PCA24、digits |

总计 **192.039 秒**，单题中位数 **11.848 秒**，最大值 **53.711 秒**。这是单次顺序评测，包含工具、模型和HTTP等待，未剥离网络波动。没有做优化前后对照，也不据此声称生产 SLA。

每题保存完整问题、答案、run、逐条事件、工具参数/输出、检查结果与耗时。额外试跑 `real-pilot.json` 是早期 ResNet 小样，不纳入11题统计。浏览器执行的实际问答和取消任务也不混入11题。

## 发现的问题与修正记录

原11题的日志问题计算值正确，但工具复用了上传日志的默认 `metrics_origin`，使模型把本应用生成的 SGD 日志说成“用户提供”。这说明“任务成功、数字正确”没有覆盖“来源措辞准确”。

修复：`ResearchTools.log` 在按 experiment ID 读取时增加真实 `log_source`、`experiment_id`、seed、split、split/code SHA，并明确 `metrics_origin` 为本应用存储实验日志；用户上传的日志仍标为用户提供。新增 `test_stored_experiment_log_retains_provenance`，最终工程测试由37项变为38项。

追加真实模型回归正确输出了来源、数据拆分和哈希。其初版检查只识别中文来源措辞，把准确引用英文工具来源的回答误报为失败；校正判据后对**同一份答案**复算为通过，原检查保存在 `original_checks`，原因保存在 `rescoring_note`，未重新抽样挑答案。检查源码为 `scripts/evaluate_log_origin.py`。

原始11题回答没有被覆写；“11/11通过预设工程判据”也没有被改写为“11/11语义完全正确”。

## 真实实验

实验 ID：`6bce3266d141433e`。原始文件：`data/experiments/6bce3266d141433e.json`。

| 条件 | 本次值 |
|---|---|
| 数据集 | scikit-learn load_digits，真实1797个8×8手写数字，10类 |
| 随机种子 / PCA维度 | 42 / 24 |
| train / validation / test | 1077 / 360 / 360，分层拆分 |
| 选参 | 各特征方案仅用validation从C=0.5/2/8中选择 |
| 预处理 | scaler/PCA仅fit训练集 |
| 原始像素 + SVM | test accuracy **97.7778%**，macro-F1 **0.977457** |
| PCA + SVM | test accuracy **97.5000%**，macro-F1 **0.974719** |
| CPU计算耗时 | 0.212秒，单次本机观测，不含模型等待 |
| 日志 | 单独的SGDClassifier，16个真实epoch，不是SVC训练日志 |
| SGD最佳accuracy/loss epoch | 16 / 16 |
| SGD末轮accuracy gap | 0.01013618，约1.01个百分点 |

两个方案测试集分别错8个和9个样本；PCA没有提升本次准确率。实验记录数据 hash、拆分 hash、代码 hash、numpy/sklearn 版本。单次拆分差异约0.28个百分点，未进行统计显著性检验；如继续比较多seed，应预先规定汇总方法并避免反复根据测试结果调参。

这验证的是“Agent能真实调用实验、使用工具计算值、准确说明结果”的流程，不是 ViT/ResNet/CLIP/DETR 的训练复现。attention 另为随机矩阵数学演示，不混为模型实验指标。

## 浏览器证据

Playwright 使用 Chromium、真实 HTTP 后端，两种视口：1440×1000、390×844。验收流程：

1. 首页→论文库检索→实际运行digits→切换方案/混淆矩阵→日志诊断→验证报告。
2. 恢复已有真实回答→打开引用/整页→保存笔记/导出→创建新任务→取消→刷新恢复取消终态。

检查文档无横向溢出、关键控件可见、核心结果正确；第一条流程还检查页面无 JavaScript 异常。截图由本次助手目视检查，主要区域未见重叠。无人工设计验收或完整无障碍测试。

这一历史批次截图保存在本地`data/screenshots/`。当前公开的桌面/手机实战截图与导出回答见[演示文档](demo.md)；当前测试规模见[V3实验报告](research-evaluation.md)，不与本页首版数据合并计数。

## 复现

启动后执行 README 中的 pytest / evaluate / Playwright 命令。以下两项补充将浏览器结果转成工作台报告，并验证日志溯源：

```bash
.venv/bin/python scripts/evaluate.py --suite browser
.venv/bin/python scripts/evaluate_log_origin.py
```

全新环境先下载论文、运行真实11题；追加日志回归读取11题产出的 experiment ID。模型调用会产生供应商费用（取决于你的配置）；脚本显式拒绝用 mock 运行真实模型套件。终端逐题显示结果，`--resume` 保留已经记录的成功和失败项。

检索的目标页依当前下载版本而定；arXiv 后续版本可能改变分页，重新下载后需要先核对原文，而不是为了通过率直接改目标页。

## 尚未覆盖

未覆盖大规模文献库、扫描PDF/OCR、所有公式解析、神经检索泛化质量、任意第三方仓库、GPU任务、多人并发平台、长期在线可用性和独立人工语义评审。当前构建有 Markdown/KaTeX 主包约712KB的体积提示，构建成功；进一步拆包属于后续优化。
