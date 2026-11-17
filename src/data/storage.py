"""保存不可静默覆盖的数据版本、逐文件哈希和划分清单。"""
from __future__ import annotations

import json
from pathlib import Path

from src.utils.artifacts import file_sha256, utc_now, write_json
from .schema import canonical_json, validate_records


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_dataset(destination: Path, records: list[dict], metadata: dict) -> dict:
    stats = validate_records(records)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["statistics"]["fingerprint"] != stats["fingerprint"]:
            raise FileExistsError("该数据版本内容已有变化，请使用新的输出版本目录")
        for filename, expected_hash in existing["file_sha256"].items():
            if file_sha256(destination / filename) != expected_hash:
                raise ValueError(f"已有文件哈希不匹配: {filename}")
        return existing
    destination.mkdir(parents=True, exist_ok=True)
    files = {}
    split_ids = {}
    for split in sorted(stats["by_split"]):
        subset = [record for record in records if record["split"] == split]
        filename = f"{split}.jsonl"
        path = destination / filename
        if path.exists():
            raise FileExistsError("目录存在未完成数据，请先检查或指定新版本目录")
        path.write_text("".join(canonical_json(record) + "\n" for record in subset), encoding="utf-8")
        files[filename] = file_sha256(path)
        split_ids[split] = [record["id"] for record in subset]
    write_json(destination / "split_ids.json", split_ids)
    files["split_ids.json"] = file_sha256(destination / "split_ids.json")
    manifest = {"schema_version": 1, "created_at_utc": utc_now(), "metadata": metadata,
                "statistics": stats, "file_sha256": files}
    write_json(manifest_path, manifest)
    return manifest
