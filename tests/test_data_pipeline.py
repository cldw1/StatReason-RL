"""数据公式复算、可复现、划分隔离、持久化与 GSM8K 处理测试。"""
from copy import deepcopy

import pytest

from src.data.gsm8k import acquire_train_file, extract_answer, process_train_rows
from src.data.quality import audit_statistics
from src.data.schema import content_fingerprint, question_key, validate_records
from src.data.statistics import FAMILY_DESCRIPTIONS, TASK_TYPES, generate_statistics
from src.data.storage import read_jsonl, save_dataset

COUNTS = {"sft": 200, "rl": 400, "validation": 100, "test": 200}


@pytest.fixture(scope="module")
def full_records():
    return generate_statistics(COUNTS, seed=42)


def test_prototype_100_covers_all_task_types():
    records = generate_statistics({"prototype": 100}, seed=42)
    assert len(records) == 100
    assert set(record["task_type"] for record in records) == set(TASK_TYPES)
    assert audit_statistics(records)["records_checked"] == 100


def test_full_scale_split_counts_and_coverage(full_records):
    statistics = validate_records(full_records)
    assert statistics["count"] == 900 and statistics["by_split"] == COUNTS
    assert statistics["template_families"] == 28
    assert statistics["duplicate_question_rows"] == 0
    for counts in statistics["by_split_task"].values():
        assert set(counts) == set(TASK_TYPES) and min(counts.values()) >= 14
    for descriptions in FAMILY_DESCRIPTIONS.values():
        assert len(descriptions) == len(set(descriptions)) == 4


def test_all_900_answers_and_reference_completions_independently_recomputed(full_records):
    quality = audit_statistics(full_records)
    assert quality["independent_recompute_failures"] == quality["reference_completion_failures"] == 0
    assert quality["max_absolute_recompute_error"] < 1e-8


def test_seed_reproducibility_and_seed_sensitivity(full_records):
    assert content_fingerprint(generate_statistics(COUNTS, seed=42)) == content_fingerprint(full_records)
    assert content_fingerprint(generate_statistics(COUNTS, seed=43)) != content_fingerprint(full_records)


def test_cross_split_question_and_family_leakage_rejected(full_records):
    first = deepcopy(full_records[0])
    copied = deepcopy(first)
    copied.update(id="other-id", split="test")
    with pytest.raises(ValueError, match="泄漏"):
        validate_records([first, copied])
    copied["question"] += "new wording"
    with pytest.raises(ValueError, match="泄漏"):
        validate_records([first, copied])
    copied["parameters"]["extra"] = "different-parameters"
    with pytest.raises(ValueError, match="模板族"):
        validate_records([first, copied])


def test_schema_rejects_intervals_and_ambiguous_percentage(full_records):
    modified = deepcopy(full_records[0])
    modified["answer_type"] = "interval"
    with pytest.raises(ValueError, match="标量"):
        validate_records([modified])
    modified["answer_type"] = "scalar"
    modified["answer_unit"] = "number"
    with pytest.raises(ValueError, match="百分数"):
        validate_records([modified])


def test_storage_is_idempotent_but_rejects_changes_and_tampering(full_records, tmp_path):
    destination = tmp_path / "release"
    records = full_records[:5]
    first = save_dataset(destination, records, {"test_fixture": True})
    second = save_dataset(destination, records, {"test_fixture": True})
    assert first == second
    assert read_jsonl(destination / "sft.jsonl") == records
    changed = deepcopy(records)
    changed[0]["answer"] = "0.5"
    with pytest.raises(FileExistsError):
        save_dataset(destination, changed, {})
    (destination / "sft.jsonl").write_text("corrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="哈希"):
        save_dataset(destination, records, {})


@pytest.mark.parametrize("raw,expected", [("work\n#### 1,234", "1234"), ("work\n#### -2.5", "-2.5"), ("work\n#### 1/4", "0.25")])
def test_gsm8k_separator_and_numeric_extraction(raw, expected):
    reasoning, answer = extract_answer(raw)
    assert reasoning == "work" and answer == expected


@pytest.mark.parametrize("raw", ["no separator", "#### 1\n#### 2", "#### NaN", "#### 1/0", "#### 25%"])
def test_gsm8k_bad_reference_rejected(raw):
    with pytest.raises(ValueError):
        extract_answer(raw)


def test_gsm8k_group_duplicates_stay_in_one_split_and_seed_reproduces():
    # 人工测试夹具，不是下载的 GSM8K 数据，也不进入数据集结果。
    rows = [{"question": f"fixture question {index}", "answer": f"fixture work\n#### {index}"} for index in range(12)]
    rows.append(deepcopy(rows[0]))
    counts = {"sft": 3, "rl": 5, "validation": 2, "reserved": 3}
    records = process_train_rows(rows, counts, seed=42)
    assert validate_records(records)["by_split"] == counts
    assert content_fingerprint(records) == content_fingerprint(process_train_rows(rows, counts, seed=42))
    keys = {}
    for record in records:
        keys.setdefault(question_key(record["question"]), set()).add(record["split"])
    assert all(len(splits) == 1 for splits in keys.values())
    assert not any(record["split"] == "test" for record in records)
    with pytest.raises(ValueError):
        process_train_rows(rows, {"test": 13})


def test_offline_gsm8k_requires_revision_before_reading_file(tmp_path):
    with pytest.raises(ValueError, match="commit"):
        acquire_train_file(tmp_path, local_train=tmp_path / "train.parquet")


def test_confidence_interval_questions_have_explicit_assumptions(full_records):
    for record in full_records:
        if record["task_type"] == "normal_mean_ci_endpoint":
            assert "正态" in record["question"] or "N(μ" in record["question"]
            assert "上限" in record["question"] or "下限" in record["question"] or "上置信限" in record["question"]
            assert record["answer_type"] == "scalar"
            if record["parameters"]["variant"] == 1:
                assert "n-1" in record["question"] and "未知" in record["question"]
