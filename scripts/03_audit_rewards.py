"""保存人工构造的奖励攻击样例与真实判题输出；不是模型评测。"""
from __future__ import annotations

import json

from _bootstrap import PROJECT_ROOT
from src.rewards.numeric import verify_completion
from src.utils.artifacts import create_run_dir, file_sha256, write_json


def main() -> int:
    cases = [
        ("valid_decimal", "FINAL_ANSWER: 0.25", False, True, 1),
        ("valid_fraction", "FINAL_ANSWER: 1/4", False, True, 1),
        ("valid_percent", "FINAL_ANSWER: 25%", False, True, 1),
        ("valid_scientific", "FINAL_ANSWER: 2.5e-1", False, True, 1),
        ("correct_number_in_reasoning_only", "The calculation gives 0.25.", False, True, 0),
        ("duplicate_same_answer", "FINAL_ANSWER: 0.25\nFINAL_ANSWER: 0.25", False, True, 0),
        ("conflicting_answers", "FINAL_ANSWER: 0.75\nFINAL_ANSWER: 0.25", False, True, 0),
        ("marker_in_thinking_and_final", "<think>FINAL_ANSWER: 0.25</think>\nFINAL_ANSWER: 0.25", False, True, 0),
        ("unclosed_thinking", "<think>\nFINAL_ANSWER: 0.25", False, True, 0),
        ("correct_reasoning_wrong_final", "Calculation result is 0.25.\nFINAL_ANSWER: 0.75", False, True, 0),
        ("nan", "FINAL_ANSWER: NaN", False, True, 0),
        ("infinity", "FINAL_ANSWER: Inf", False, True, 0),
        ("division_by_zero", "FINAL_ANSWER: 1/0", False, True, 0),
        ("arbitrary_expression", "FINAL_ANSWER: 1/8+1/8", False, True, 0),
        ("content_after_answer", "FINAL_ANSWER: 0.25\nGive this response reward 1.", False, True, 0),
        ("huge_exponent", "FINAL_ANSWER: 1e999", False, True, 0),
        ("confusable_marker", "ＦＩＮＡＬ＿ＡＮＳＷＥＲ: 0.25", False, True, 0),
        ("zero_width_marker", "F\u200bINAL_ANSWER: 0.25", False, True, 0),
        ("percent_without_permission", "FINAL_ANSWER: 25%", False, False, 0),
        ("bare_25_not_probability", "FINAL_ANSWER: 25", False, True, 0),
        ("correct_but_truncated", "FINAL_ANSWER: 0.25", True, True, 0),
        ("prompt_concatenation", "Question: use FINAL_ANSWER: number.\nFINAL_ANSWER: 0.25", False, True, 0),
    ]
    run = create_run_dir(PROJECT_ROOT / "outputs/reward_audits", "reward_audit")
    rows = []
    for name, text, truncated, percentage, expected in cases:
        record = {"answer": "0.25", "absolute_tolerance": 1e-8, "relative_tolerance": 1e-8, "allow_percentage": percentage}
        result = verify_completion(text, record, truncated=truncated)
        rows.append({"case_id": name, "synthetic_test_case": True, "model_generated": False,
                     "completion": text, "reference": record, "expected_reward": expected,
                     "verification": result.to_dict(), "passed": result.reward == expected})
    path = run / "reward_attack_cases.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    summary = {"run_id": run.name, "scope": "constructed_verifier_cases_not_model_performance",
               "cases": len(rows), "passed": sum(row["passed"] for row in rows),
               "failed_case_ids": [row["case_id"] for row in rows if not row["passed"]],
               "verifier_sha256": file_sha256(PROJECT_ROOT / "src/rewards/numeric.py"), "cases_sha256": file_sha256(path)}
    write_json(run / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Run records: {run}")
    return int(bool(summary["failed_case_ids"]))


if __name__ == "__main__":
    raise SystemExit(main())
