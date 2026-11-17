"""不依赖 Qwen 下载的评测逻辑测试；构造预测不进入模型结果。"""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from src.data.storage import save_dataset
from src.data.statistics import generate_statistics
from src.evaluation.metrics import summarize_predictions, wilson_interval
from src.evaluation.panel import select_development_panel
from src.evaluation.runner import assemble_prediction
from src.evaluation.storage import append_prediction, export_csv, read_predictions
from src.models.inference import example_seed, extract_completion
from src.models.prompts import build_messages
from src.models.snapshot import ModelNotCachedError, resolve_snapshot, snapshot_complete
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def fixture_record():
    return generate_statistics({"prototype": 1}, seed=7)[0]


def generated_fixture(completion, *, truncated=False):
    return {"completion": completion, "raw_completion": completion, "truncated": truncated,
            "stop_reason": "max_new_tokens" if truncated else "eos", "generation_seconds": 0.1,
            "prompt_token_count": 10, "generated_token_count": 4, "text_token_count": 3,
            "peak_allocated_bytes": 100, "peak_reserved_bytes": 200, "seed": 42,
            "prompt_text": "FINAL_ANSWER: 999 (this is a prompt fixture, never a response)"}


def test_prompt_contains_question_not_reference_or_solution():
    record = fixture_record()
    record.update(question="unique question sentinel", answer="secret-reference-sentinel", solution="secret-solution-sentinel")
    messages = build_messages(record)
    combined = json.dumps(messages)
    assert messages[-1]["content"] == record["question"]
    assert "secret-reference-sentinel" not in combined and "secret-solution-sentinel" not in combined


@pytest.mark.parametrize("sequence,prompt_length,eos,max_new,expected_tokens,truncated,reason", [
    ([1, 2, 99, 3, 2, 0], 3, [2], 3, [3, 2], False, "eos"),
    ([7, 8, 9], 1, [9], 2, [8, 9], False, "eos"),
    ([7, 8, 9], 1, [2], 2, [8, 9], True, "max_new_tokens"),
    ([7, 8], 1, [2], 3, [8], True, "stopped_without_eos"),
    ([7, 2], 1, [2, 3], 1, [2], False, "eos"),
])
def test_prompt_slicing_and_eos_boundaries(sequence, prompt_length, eos, max_new, expected_tokens, truncated, reason):
    boundary = extract_completion(sequence, prompt_length, eos, max_new)
    assert boundary.token_ids == expected_tokens and boundary.truncated == truncated and boundary.stop_reason == reason
    assert boundary.text_token_ids == (expected_tokens[:-1] if not truncated else expected_tokens)


def test_seeds_depend_on_identity_not_evaluation_order():
    assert example_seed(42, "a") == example_seed(42, "a")
    assert len({example_seed(42, "a"), example_seed(43, "a"), example_seed(42, "b"), example_seed(42, "a", 1)}) == 4


def test_prediction_builder_grades_only_completion():
    record = fixture_record()
    output = generated_fixture("No final answer here.")
    output["prompt_text"] = record["solution"]
    wrong = assemble_prediction(record, output, mode="non-thinking", model_name="test_fixture", revision="fixture", signature="fixture")
    assert wrong["verification"]["reward"] == 0
    correct = assemble_prediction(record, generated_fixture(record["solution"]), mode="non-thinking", model_name="test_fixture", revision="fixture", signature="fixture")
    assert correct["verification"]["reward"] == 1
    truncated = assemble_prediction(record, generated_fixture(record["solution"], truncated=True), mode="non-thinking", model_name="test_fixture", revision="fixture", signature="fixture")
    assert truncated["verification"]["reward"] == 0


def test_wilson_known_boundaries_and_empty():
    assert wilson_interval(0, 0) is None
    assert wilson_interval(0, 10)[1] == pytest.approx(0.2775327999)
    assert wilson_interval(10, 10)[0] == pytest.approx(0.7224672001)
    assert wilson_interval(5, 10)[0] < 0.5 < wilson_interval(5, 10)[1]
    with pytest.raises(ValueError):
        wilson_interval(11, 10)


def test_metrics_keep_modes_separate_and_reject_duplicate_question():
    record = fixture_record()
    correct = assemble_prediction(record, generated_fixture(record["solution"]), mode="non-thinking", model_name="test_fixture", revision="fixture", signature="fixture")
    wrong = assemble_prediction(record, generated_fixture("wrong"), mode="thinking", model_name="test_fixture", revision="fixture", signature="fixture")
    summary = summarize_predictions([correct, wrong])
    assert summary["by_mode"]["non-thinking"]["sampled_pass_at_1"] == 1
    assert summary["by_mode"]["thinking"]["sampled_pass_at_1"] == 0
    assert "greedy_accuracy" in summarize_predictions([correct], sampled=False)["by_mode"]["non-thinking"]
    with pytest.raises(ValueError, match="一次生成"):
        summarize_predictions([correct, correct])


def test_jsonl_recovery_and_csv_preserve_multiline_text(tmp_path):
    record = fixture_record()
    row = assemble_prediction(record, generated_fixture(record["solution"]), mode="non-thinking", model_name="test_fixture", revision="fixture", signature="fixture")
    path = tmp_path / "predictions.jsonl"
    append_prediction(path, row)
    with path.open("ab") as stream:
        stream.write(b'{"partial":"\xe4\xb8')
    with pytest.raises(ValueError):
        read_predictions(path)
    assert read_predictions(path, repair_partial_last_line=True) == [row]
    assert (tmp_path / "predictions.interrupted.jsonl").exists()
    path.write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
    second = deepcopy(row)
    second["mode"] = "thinking"
    append_prediction(path, second)
    assert read_predictions(path) == [row, second]
    csv_path = tmp_path / "predictions.csv"
    export_csv(csv_path, [row])
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        exported = list(csv.DictReader(stream))
    assert exported[0]["completion"] == row["completion"]


def test_cache_probe_is_offline_and_checks_complete_shards(tmp_path):
    model = {"name_or_path": "Qwen/Qwen3-0.6B", "hf_home": str(tmp_path), "revision": None}
    with pytest.raises(ModelNotCachedError):
        resolve_snapshot(model)
    revision = "a" * 40
    folder = tmp_path / "hub/models--Qwen--Qwen3-0.6B/snapshots" / revision
    folder.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
        (folder / name).write_text('{}', encoding="utf-8")
    (folder / "model.safetensors.index.json").write_text('{"weight_map":{"a":"part.safetensors"}}', encoding="utf-8")
    assert not snapshot_complete(folder)
    (folder / "part.safetensors").write_bytes(b"fixture-not-real-weights")
    assert snapshot_complete(folder)
    assert resolve_snapshot(model).revision == revision
    # 这里只验证缓存文件选择；测试夹具不能被描述为真实模型下载/加载成功。
    (folder / "model.safetensors.index.json").write_text('{"weight_map":{"a":"../outside.safetensors"}}', encoding="utf-8")
    assert not snapshot_complete(folder)


def test_panel_selection_uses_only_validation_and_is_reproducible(tmp_path):
    records = generate_statistics({"validation": 14}, seed=42)
    save_dataset(tmp_path / "statistics_full", records, {"fixture": True})
    gsm = deepcopy(records[0])
    gsm.update(id="gsm-fixture", source="openai/gsm8k", task_type="gsm8k_math", template_family="unknown", question="fixture GSM8K validation only")
    save_dataset(tmp_path / "gsm8k", [gsm], {"fixture": True})
    config = load_config(ROOT / "configs/evaluation.yaml")["evaluation"]
    first, metadata = select_development_panel(tmp_path, config, 42)
    second, second_metadata = select_development_panel(tmp_path, config, 42)
    assert first == second and metadata == second_metadata
    assert len(first) == 8 and all(record["split"] == "validation" for record in first)
    assert metadata["official_test_accessed"] is False
    config["scope"] = "test"
    with pytest.raises(ValueError):
        select_development_panel(tmp_path, config, 42)
