"""复用已有缓存，固定模型 commit，显式授权下载后才联网。"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re

from src.utils.artifacts import file_sha256

ALLOWED_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                 "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
                 "chat_template.jinja", "chat_template.json", "*.safetensors", "*.safetensors.index.json"]
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class ModelNotCachedError(FileNotFoundError):
    """没有可离线使用的完整模型快照。"""


@dataclass(frozen=True)
class ModelSnapshot:
    repository: str
    revision: str
    path: Path


def snapshot_complete(path: Path) -> bool:
    if not all((path / name).is_file() for name in ("config.json", "tokenizer_config.json", "tokenizer.json")):
        return False
    single = path / "model.safetensors"
    if single.is_file() and single.stat().st_size > 0:
        return True
    index = path / "model.safetensors.index.json"
    if index.is_file():
        try:
            mapping = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            shards = {Path(name) for name in mapping.values()}
            if not shards or any(name.is_absolute() or len(name.parts) != 1 for name in shards):
                return False
            return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in shards)
        except (KeyError, ValueError, TypeError):
            return False
    return False


def resolve_snapshot(model_config: dict, *, allow_download: bool = False, max_download_gib: float = 2.0) -> ModelSnapshot:
    """本地快照优先；多个无默认引用的版本存在时要求显式选择。"""
    repository = model_config["name_or_path"]
    if repository not in ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen2.5-Math-1.5B-Instruct"):
        raise ValueError("本项目下载入口只接受三个指定的 Qwen 候选仓库")
    model_cache = Path(model_config["hf_home"]) / "hub" / ("models--" + repository.replace("/", "--"))
    revision = model_config.get("revision")
    if revision is not None and not COMMIT_PATTERN.fullmatch(revision):
        raise ValueError("模型 revision 必须为完整 commit，不能使用可变分支作为固定版本")
    if revision:
        candidate = model_cache / "snapshots" / revision
        if snapshot_complete(candidate):
            return ModelSnapshot(repository, revision, candidate)
    else:
        ref = model_cache / "refs/main"
        if ref.is_file():
            cached_revision = ref.read_text(encoding="utf-8").strip()
            if COMMIT_PATTERN.fullmatch(cached_revision):
                candidate = model_cache / "snapshots" / cached_revision
                if snapshot_complete(candidate):
                    return ModelSnapshot(repository, cached_revision, candidate)
        complete = [path for path in (model_cache / "snapshots").glob("*") if COMMIT_PATTERN.fullmatch(path.name) and snapshot_complete(path)]
        if len(complete) == 1:
            return ModelSnapshot(repository, complete[0].name, complete[0])
        if len(complete) > 1:
            raise ValueError("存在多个完整模型 commit，请显式指定 revision")
    if not allow_download:
        raise ModelNotCachedError("未发现完整 Qwen3 快照；请使用 --download-model 获取后再离线运行")
    if max_download_gib <= 0:
        raise ValueError("下载预算必须为正")
    from fnmatch import fnmatch
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(token=False).model_info(repository, revision=revision or "main", files_metadata=True, timeout=15)
    if not info.sha or not COMMIT_PATTERN.fullmatch(info.sha):
        raise ValueError("无法确定模型 commit")
    selected = [item for item in info.siblings if any(fnmatch(item.rfilename, pattern) for pattern in ALLOWED_FILES)]
    if not selected or any(item.size is None for item in selected):
        raise ValueError("上游文件大小信息不完整，不能自动检查下载预算")
    total_bytes = sum(item.size for item in selected)
    if total_bytes > max_download_gib * 1024 ** 3:
        raise ValueError(f"模型文件共 {total_bytes} 字节，超出下载预算；请先确认资源预算")
    path = Path(snapshot_download(repository, revision=info.sha, cache_dir=Path(model_config["hf_home"]) / "hub",
                                  # Windows 上并发首次检测 symlink 支持会读到尚未完成的
                                  # 能力探测缓存，触发 WinError 1314。串行下载使 HF 自身的
                                  # 普通文件回退路径生效，不要求管理员权限或修改系统设置。
                                  allow_patterns=ALLOWED_FILES, token=False,
                                  max_workers=1 if os.name == "nt" else 2, etag_timeout=15))
    if not snapshot_complete(path):
        raise ModelNotCachedError("下载结束后模型快照仍不完整")
    return ModelSnapshot(repository, info.sha, path)


def snapshot_metadata(snapshot: ModelSnapshot) -> dict:
    """记录真实权重文件哈希，避免仅凭模型名称追溯实验。"""
    files = {}
    for path in sorted(snapshot.path.iterdir()):
        if path.is_file() and path.suffix in (".json", ".safetensors", ".jinja", ".txt"):
            files[path.name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    return {"repository": snapshot.repository, "revision": snapshot.revision,
            "snapshot_path": str(snapshot.path), "files": files,
            "total_file_bytes": sum(value["bytes"] for value in files.values())}
