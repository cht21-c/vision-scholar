#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  uv_bin="uv"
elif [[ -x .venv/bin/uv ]]; then
  uv_bin=".venv/bin/uv"
else
  echo "请先安装 uv：https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi
command -v npm >/dev/null 2>&1 || { echo "请先安装 Node.js 20.19+ 或 22.12+。"; exit 1; }
"$uv_bin" sync --locked --extra dev
npm --prefix frontend ci
npm --prefix frontend run build
mkdir -p data/reports data/screenshots
echo "准备完成。执行 ./start.sh；随后在论文库点击「导入经典四篇」。"
