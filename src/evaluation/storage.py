"""逐题追加、可恢复 JSONL 和完整 CSV 导出。"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil


def append_prediction(path: Path, row: dict) -> None:
    payload = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
    with path.open("ab+") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell():
            stream.seek(-1, os.SEEK_END)
            last = stream.read(1)
            stream.seek(0, os.SEEK_END)
            if last != b"\n":
                stream.write(b"\n")
        stream.write(payload.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())


def read_predictions(path: Path, *, repair_partial_last_line: bool = False) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_bytes().splitlines()
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line.decode("utf-8")))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if not repair_partial_last_line or index != len(lines) - 1:
                raise ValueError("预测 JSONL 存在损坏，禁止忽略中间记录")
            backup = path.with_name(path.stem + ".interrupted.jsonl")
            if backup.exists():
                raise FileExistsError("已有中断备份，请先检查，不能覆盖")
            shutil.copyfile(path, backup)
            path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def export_csv(path: Path, rows: list[dict]) -> None:
    columns = ["id", "mode", "source", "task_type", "evaluation_stratum", "question", "answer", "completion", "raw_completion",
               "extracted_answer", "parsed_answer", "format_valid", "numeric_correct", "strict_correct", "reward",
               "truncated", "stop_reason", "prompt_token_count", "generated_token_count", "text_token_count",
               "generation_seconds", "peak_allocated_bytes", "peak_reserved_bytes", "seed", "model_revision"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, **row["verification"]})
