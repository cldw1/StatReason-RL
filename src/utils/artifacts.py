"""使用标准库保存实验记录；禁止用虚构指标填充缺失的实验。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    """记录时间统一为 UTC，展示时再转换为当地时间。"""
    return datetime.now(timezone.utc).isoformat()


def write_json(path: str | Path, value: Any) -> None:
    """拒绝 NaN/Inf，避免日志中出现非标准 JSON；临时文件后原子替换。"""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: str | Path) -> str:
    """按原始字节计算指纹。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_run_dir(root: str | Path, stage: str) -> Path:
    """时间加随机后缀保证同秒启动的运行也不会相互覆盖。"""
    if not stage or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in stage):
        raise ValueError("stage 只能使用小写英文、数字、下划线和连字符")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / f"{stage}_{stamp}_{uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path

