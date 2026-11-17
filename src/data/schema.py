"""题目 schema、去重和模板族隔离校验。"""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
import unicodedata

REQUIRED_FIELDS = {
    "id", "source", "task_type", "template_family", "question", "parameters",
    "solution", "answer", "answer_type", "absolute_tolerance", "relative_tolerance", "split",
    "answer_unit", "allow_percentage",
}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def question_key(question: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", question).casefold().split())


def content_fingerprint(records: list[dict]) -> str:
    """与文件顺序无关，覆盖标准答案、容差及 split 等所有字段。"""
    content = "\n".join(canonical_json(record) for record in sorted(records, key=lambda record: record["id"]))
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_records(records: list[dict], *, isolate_families: bool = True) -> dict:
    ids = set()
    question_splits = defaultdict(set)
    family_splits = defaultdict(set)
    parameter_splits = defaultdict(set)
    for record in records:
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(f"缺少题目字段: {sorted(missing)}")
        if record["id"] in ids:
            raise ValueError(f"重复 id: {record['id']}")
        ids.add(record["id"])
        if record["answer_type"] != "scalar":
            raise ValueError("当前数据只支持标量答案，区间必须拆分端点")
        if record["split"] not in {"prototype", "sft", "rl", "validation", "test", "reserved"}:
            raise ValueError("未知数据划分")
        if not record["question"] or not record["solution"] or not isinstance(record["parameters"], dict):
            raise ValueError("题目、解答和参数不能为空或类型错误")
        for name in ("answer", "absolute_tolerance", "relative_tolerance"):
            if not Decimal(str(record[name])).is_finite():
                raise ValueError("答案或容差出现非有限值")
        if min(float(record["absolute_tolerance"]), float(record["relative_tolerance"])) < 0:
            raise ValueError("容差为负")
        if record["allow_percentage"] and record["answer_unit"] != "probability":
            raise ValueError("只有明确为概率的题目允许百分数")
        if record["answer_unit"] == "probability" and not 0 <= Decimal(str(record["answer"])) <= 1:
            raise ValueError("概率答案越界")
        question_splits[question_key(record["question"])].add(record["split"])
        if record["source"] == "statreason_synthetic":
            family_splits[record["template_family"]].add(record["split"])
            parameter_splits[(record["task_type"], canonical_json(record["parameters"]))].add(record["split"])
    if any(len(splits) > 1 for splits in question_splits.values()):
        raise ValueError("同一道题跨 split 泄漏")
    if any(len(splits) > 1 for splits in parameter_splits.values()):
        raise ValueError("相同参数题跨 split 泄漏")
    if isolate_families and any(len(splits) > 1 for splits in family_splits.values()):
        raise ValueError("统计题模板族跨 split 泄漏")
    by_split = dict(Counter(record["split"] for record in records))
    by_task = dict(Counter(record["task_type"] for record in records))
    by_split_task = {split: dict(Counter(record["task_type"] for record in records if record["split"] == split)) for split in sorted(by_split)}
    return {"count": len(records), "by_split": by_split, "by_task": by_task,
            "by_split_task": by_split_task, "template_families": len(family_splits),
            "duplicate_question_rows": len(records) - len(question_splits),
            "cross_split_question_overlap": 0, "cross_split_family_overlap": 0,
            "fingerprint": content_fingerprint(records)}

