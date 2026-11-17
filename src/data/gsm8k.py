"""GSM8K 官方训练文件获取、答案提取和无交叉划分；不下载官方测试文件。"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import random
import re

from src.rewards.numeric import parse_numeric
from src.utils.artifacts import file_sha256
from .schema import question_key, validate_records

REPOSITORY = "openai/gsm8k"
TRAIN_FILENAME = "main/train-00000-of-00001.parquet"


def acquire_train_file(cache_dir: Path, *, revision: str | None = None, local_train: Path | None = None) -> tuple[Path, dict]:
    """固定 commit 后只下载 train parquet；本地导入必须提供来源 commit。"""
    if local_train is not None:
        if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("离线导入需要提供 40 位 GSM8K 来源 commit")
        if not local_train.is_file() or local_train.suffix != ".parquet":
            raise ValueError("离线导入只接受官方训练 parquet 文件")
        path = local_train.resolve()
        provenance = "user_supplied_official_train_file_revision_not_remotely_verified"
    else:
        from huggingface_hub import HfApi, hf_hub_download
        if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
            cached = cache_dir / "datasets--openai--gsm8k" / "snapshots" / revision / TRAIN_FILENAME
        else:
            cached = None
        if cached is not None and cached.is_file():
            path = cached.resolve()
            provenance = "official_hf_commit_cache"
        else:
            info = HfApi(token=False).dataset_info(REPOSITORY, revision=revision or "main", timeout=15)
            if not info.sha or not re.fullmatch(r"[0-9a-f]{40}", info.sha):
                raise ValueError("无法获得完整数据 commit")
            revision = info.sha
            if TRAIN_FILENAME not in {item.rfilename for item in info.siblings}:
                raise ValueError("上游训练文件结构发生变化，需要核对后更新路径")
            path = Path(hf_hub_download(REPOSITORY, TRAIN_FILENAME, repo_type="dataset", revision=revision,
                                         cache_dir=cache_dir, token=False, etag_timeout=15))
            provenance = "official_huggingface_hub_commit_pinned"
    return path, {"repository": REPOSITORY, "revision": revision, "filename": TRAIN_FILENAME,
                  "source_split": "train", "raw_sha256": file_sha256(path),
                  "provenance": provenance, "official_test_downloaded": False}


def extract_answer(solution: str) -> tuple[str, str]:
    """GSM8K 使用唯一 #### 分隔最终答案，保留前面的原始解题过程。"""
    if not isinstance(solution, str) or solution.count("####") != 1:
        raise ValueError("GSM8K 答案必须包含唯一 #### 分隔符")
    reasoning, final = solution.split("####")
    value = parse_numeric(final.strip(), allow_percentage=False)
    return reasoning.strip(), str(value)


def process_train_rows(rows: list[dict], counts: dict[str, int], *, seed: int = 42) -> list[dict]:
    if set(counts) - {"sft", "rl", "validation", "reserved"} or any(count < 0 for count in counts.values()):
        raise ValueError("GSM8K 处理只允许使用官方 train 的四个开发划分")
    if sum(counts.values()) != len(rows):
        raise ValueError("请求的划分数与原始训练集行数不符")
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[question_key(row["question"])].append((index, row))
    # 相同题目始终作为一个组分配，不能因相同行号不同而跨 split 泄漏。
    grouped = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(grouped)
    grouped.sort(key=len, reverse=True)  # 稳定排序，先安置重复题组，单行组填充尾部。
    remaining = dict(counts)
    records = []
    for group in grouped:
        split = next((name for name, available in remaining.items() if available >= len(group)), None)
        if split is None:
            raise ValueError("重复题组不能在固定配额下完整安置，请调整划分计划")
        remaining[split] -= len(group)
        for index, row in group:
            reasoning, answer = extract_answer(row["answer"])
            value = Decimal(answer)
            integer = value == value.to_integral_value()
            records.append({
                "id": f"gsm8k-train-{index:05d}", "source": REPOSITORY,
                "task_type": "gsm8k_math", "template_family": "human_authored_unknown_family",
                "question": row["question"].strip(), "parameters": {"official_train_row": index},
                "solution": reasoning + f"\nFINAL_ANSWER: {answer}", "original_solution": row["answer"],
                "answer": answer, "answer_type": "scalar", "answer_unit": "number",
                "absolute_tolerance": 0.0 if integer else 1e-9, "relative_tolerance": 0.0,
                "allow_percentage": False, "split": split,
            })
    if any(remaining.values()):
        raise ValueError("未达到精确划分配额")
    validate_records(records)
    return records


def prepare_gsm8k(cache_dir: Path, counts: dict[str, int], *, seed: int = 42,
                  revision: str | None = None, local_train: Path | None = None) -> tuple[list[dict], dict]:
    import pyarrow.parquet as parquet
    path, provenance = acquire_train_file(cache_dir, revision=revision, local_train=local_train)
    rows = parquet.read_table(path, columns=["question", "answer"]).to_pylist()
    if len(rows) != 7473:
        raise ValueError(f"官方 GSM8K main train 应有 7473 行，实际为 {len(rows)}")
    return process_train_rows(rows, counts, seed=seed), {**provenance, "seed": seed, "raw_row_count": len(rows)}

