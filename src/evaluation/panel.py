"""仅从验证 split 构建可复现开发题组；不接受测试集路径。"""
from __future__ import annotations

import json
from pathlib import Path
import random

from src.data.schema import content_fingerprint
from src.data.statistics import TASK_TYPES
from src.data.storage import read_jsonl
from src.utils.artifacts import file_sha256


def load_validation_release(directory: Path, expected_source: str) -> tuple[list[dict], dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "validation.jsonl"
    if file_sha256(path) != manifest["file_sha256"]["validation.jsonl"]:
        raise ValueError("验证文件哈希与数据 manifest 不一致")
    records = read_jsonl(path)
    if any(record["split"] != "validation" or record["source"] != expected_source for record in records):
        raise ValueError("开发评测只接受来源正确且标记为 validation 的题目")
    return records, manifest


def select_development_panel(root: Path, evaluation: dict, seed: int) -> tuple[list[dict], dict]:
    if evaluation["scope"] != "development_validation_only":
        raise ValueError("此入口仅用于开发评测，正式测试入口需在方案冻结后实现")
    per_task, gsm_count = evaluation["statistics_per_task"], evaluation["gsm8k_count"]
    if per_task < 0 or gsm_count < 0 or per_task + gsm_count == 0:
        raise ValueError("开发题目数必须非负且不全为零")
    statistics, stat_manifest = load_validation_release(root / "statistics_full", "statreason_synthetic")
    gsm8k, gsm_manifest = load_validation_release(root / "gsm8k", "openai/gsm8k")
    rng = random.Random(seed)
    records = []
    for task in TASK_TYPES:
        pool = sorted((record for record in statistics if record["task_type"] == task), key=lambda record: record["id"])
        records.extend(rng.sample(pool, per_task))
    records.extend(rng.sample(sorted(gsm8k, key=lambda record: record["id"]), gsm_count))
    rng.shuffle(records)
    maximum = evaluation["max_examples"]
    if maximum < 1:
        raise ValueError("max_examples 必须为正")
    records = records[:maximum]
    return records, {"scope": "development_validation_only", "seed": seed,
                     "selected_ids": [record["id"] for record in records],
                     "panel_fingerprint": content_fingerprint(records),
                     "source_fingerprints": {"statistics": stat_manifest["statistics"]["fingerprint"],
                                              "gsm8k": gsm_manifest["statistics"]["fingerprint"]},
                     "official_test_accessed": False}
