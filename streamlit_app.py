"""Run: .venv/bin/streamlit run streamlit_app.py --server.port 8501

Streamlit is a baseline UI over the same FastAPI service; it always selects the
independent LangGraph loop. It does not start a second service/database worker.
"""
import json

import httpx
import streamlit as st

st.set_page_config(page_title="Vision Scholar · LangGraph 基线", page_icon="📚", layout="wide")
st.title("Vision Scholar · LangGraph 基线")
st.caption("按原简历设计重建：Graph State / Checkpoint · Dense + BM25 + Reranker · 证据回溯")
base = "http://127.0.0.1:8765/api"


def api(path, body=None):
    with httpx.Client(timeout=30) as client:
        response = client.get(base + path) if body is None else client.post(base + path, json=body)
        response.raise_for_status()
        return response.json()


try:
    bootstrap = api("/bootstrap")
except httpx.HTTPError:
    st.error("请先启动 FastAPI 服务：.venv/bin/uvicorn backend.main:app --port 8765")
    st.stop()

with st.sidebar:
    st.header("研究设置")
    retrieval = st.selectbox("检索方案", ["neural", "lsa"],
                            format_func=lambda x: "神经混合检索" if x == "neural" else "轻量检索")
    mode = st.selectbox("研究类型", ["auto", "paper", "code", "experiment"])
    paper_ids = st.multiselect("限定论文", [p["id"] for p in bootstrap["papers"]])
    if st.button("新建研究") or st.session_state.get("retrieval") != retrieval:
        st.session_state.pop("session_id", None)
    st.session_state["retrieval"] = retrieval
    st.caption("运行版本固定为 V0 · 独立 LangGraph 循环。神经模型首次加载需要较长时间。")

if "session_id" not in st.session_state:
    st.session_state["session_id"] = api("/sessions", {"title": "LangGraph 基线研究"})["id"]
sid = st.session_state["session_id"]


@st.fragment(run_every=1)
def conversation():
    session = api(f"/sessions/{sid}")
    for run in session["runs"]:
        with st.chat_message("user"):
            st.markdown(run["prompt"])
        with st.chat_message("assistant"):
            st.markdown(run["answer"] or run.get("error") or "正在查找证据、调用工具…")
            if run["citations"]:
                with st.expander("原文证据"):
                    for citation in run["citations"]:
                        st.caption(citation["id"])
                        st.text(citation["text"])
            with st.expander("执行记录与调用统计"):
                st.json(run["usage"])
                for event in api(f"/runs/{run['id']}/trace")["events"]:
                    if event["type"] in {"tool_start", "tool_end", "compact"}:
                        st.code(json.dumps(event["data"], ensure_ascii=False, indent=2))
            if run["status"] in {"queued", "running"}:
                if st.button("停止", key="cancel-" + run["id"]):
                    api(f"/runs/{run['id']}/cancel", {})


conversation()
if question := st.chat_input("读论文、看源码、分析训练日志…"):
    try:
        api("/runs", {"session_id": sid, "prompt": question, "paper_ids": paper_ids,
                      "mode": mode, "harness": "legacy", "retriever": retrieval})
        st.rerun()
    except httpx.HTTPStatusError as exc:
        st.error(exc.response.json().get("detail", "请求失败"))
