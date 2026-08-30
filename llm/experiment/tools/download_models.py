#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT_DIR
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_config import MODELS  # noqa: E402


def cli_command() -> list[str]:
    hf = shutil.which("hf")
    if hf:
        return [hf]
    legacy = shutil.which("huggingface-cli")
    if legacy:
        return [legacy]
    raise RuntimeError("Neither `hf` nor `huggingface-cli` is available. Install huggingface_hub[cli] in mllm.")


_SUPPORTS_RESUME_DOWNLOAD: Optional[bool] = None


def supports_resume_download(cmd_prefix: list[str]) -> bool:
    global _SUPPORTS_RESUME_DOWNLOAD
    if _SUPPORTS_RESUME_DOWNLOAD is None:
        try:
            result = subprocess.run(
                cmd_prefix + ["download", "--help"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            _SUPPORTS_RESUME_DOWNLOAD = "--resume-download" in result.stdout
        except Exception:
            _SUPPORTS_RESUME_DOWNLOAD = False
    return _SUPPORTS_RESUME_DOWNLOAD


def get_model_info(repo_id: str, endpoint: str, token: Optional[str]) -> dict[str, Any]:
    try:
        from huggingface_hub import HfApi

        api = HfApi(endpoint=endpoint, token=token)
        info = api.model_info(repo_id=repo_id)
        return {"sha": getattr(info, "sha", None), "private": getattr(info, "private", None), "gated": getattr(info, "gated", None)}
    except Exception as exc:
        return {"info_error": str(exc)}


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def download_one(model_name: str, config: dict[str, Any], endpoint: str, token: Optional[str], dry_run: bool) -> dict[str, Any]:
    repo_id = config["hf_repo"]
    local_dir = Path(config["model_path"]).expanduser()
    local_dir.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "model_name": model_name,
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "endpoint": endpoint,
        "started_at": datetime.now().isoformat(),
        "status": "pending",
    }
    record.update(get_model_info(repo_id, endpoint, token))

    if dry_run:
        record["status"] = "dry_run"
        return record

    cmd_prefix = cli_command()
    cmd = cmd_prefix + [
        "download",
        repo_id,
        "--local-dir",
        str(local_dir),
    ]
    if supports_resume_download(cmd_prefix):
        cmd.append("--resume-download")
    env = os.environ.copy()
    env["HF_ENDPOINT"] = endpoint
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        env.pop(key, None)
    if token:
        env["HF_TOKEN"] = token

    record["command"] = " ".join(cmd)
    print(f"\n=== Downloading {model_name}: {repo_id} -> {local_dir}", flush=True)
    process = subprocess.run(cmd, env=env, text=True)
    record["returncode"] = process.returncode
    record["finished_at"] = datetime.now().isoformat()
    record["file_count"] = count_files(local_dir)
    if process.returncode == 0:
        record["status"] = "ok"
    else:
        record["status"] = "blocked_or_failed" if config.get("gated") else "failed"
    manifest_path = local_dir.parent / "download_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False)
    record["manifest_path"] = str(manifest_path)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Download all experiment model weights through HF-Mirror.")
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--models", nargs="+", help="Optional subset of model names.")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", default=str(ROOT_DIR / "logs" / "download_summary.json"))
    args = parser.parse_args()

    selected = set(args.models or MODELS.keys())
    results = []
    for model_name, config in MODELS.items():
        if model_name not in selected:
            continue
        try:
            results.append(download_one(model_name, config, args.endpoint, args.token, args.dry_run))
        except Exception as exc:
            results.append(
                {
                    "model_name": model_name,
                    "repo_id": config.get("hf_repo"),
                    "local_dir": config.get("model_path"),
                    "endpoint": args.endpoint,
                    "status": "failed",
                    "error": str(exc),
                    "finished_at": datetime.now().isoformat(),
                }
            )
            print(f"ERROR downloading {model_name}: {exc}", file=sys.stderr, flush=True)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump({"endpoint": args.endpoint, "results": results}, file, indent=2, ensure_ascii=False)
    print(f"\nDownload summary written to {summary_path}")


if __name__ == "__main__":
    main()
