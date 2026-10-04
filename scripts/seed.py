"""Download/parse the four paper seeds through an already running service."""
import argparse
import json

import httpx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://127.0.0.1:8765")
args = parser.parse_args()
print("正在下载 ResNet / ViT / CLIP / DETR；已有论文会复用缓存。", flush=True)
response = httpx.post(f"{args.base_url}/api/papers/seed", timeout=240)
response.raise_for_status()
print(json.dumps([{"id": p["id"], "pages": p["page_count"], "chunks": p["chunk_count"]}
                  for p in response.json()["papers"]], ensure_ascii=False, indent=2))
