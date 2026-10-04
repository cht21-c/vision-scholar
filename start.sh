#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python || ! -f frontend/dist/index.html ]]; then
  echo "依赖或前端构建缺失，请先执行 ./setup.sh。"
  exit 1
fi
echo "Vision Scholar → http://127.0.0.1:8765"
echo "单进程本地运行；Ctrl+C 停止。"
exec .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8765 --workers 1
