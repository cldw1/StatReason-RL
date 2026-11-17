"""题目级开发评测指标；不将不同模式或多回答混成独立题目。"""
from __future__ import annotations

from collections import Counter
import math
from statistics import NormalDist


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> list[float] | None:
    if total < 0 or successes < 0 or successes > total or not 0 < confidence < 1:
        raise ValueError("Wilson 区间参数非法")
    if total == 0:
        return None
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    proportion = successes / total
    divisor = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / divisor
    half = z / divisor * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    return [max(0.0, center - half), min(1.0, center + half)]


def _summarize(rows: list[dict], metric: str) -> dict:
    total = len(rows)
    correct = sum(row["verification"]["strict_correct"] for row in rows)
    return {"question_count": total, "strict_correct_count": correct,
            metric: correct / total if total else None,
            "wilson_95_interval": wilson_interval(correct, total),
            "format_valid_rate": sum(row["verification"]["format_valid"] for row in rows) / total if total else None,
            "numeric_correct_rate": sum(row["verification"]["numeric_correct"] for row in rows) / total if total else None,
            "parse_failure_rate": sum(not row["verification"]["parse_success"] for row in rows) / total if total else None,
            "truncation_rate": sum(row["truncated"] for row in rows) / total if total else None,
            "mean_generated_tokens": sum(row["generated_token_count"] for row in rows) / total if total else None,
            "generation_seconds": sum(row["generation_seconds"] for row in rows),
            "total_generated_tokens": sum(row["generated_token_count"] for row in rows)}


def summarize_predictions(rows: list[dict], *, sampled: bool = True) -> dict:
    metric = "sampled_pass_at_1" if sampled else "greedy_accuracy"
    grouped = {}
    seen = set()
    for row in rows:
        key = (row["mode"], row["id"])
        if key in seen:
            raise ValueError("同一模式每题只允许一次生成；多回答需要单独的 pass@k 聚合")
        seen.add(key)
        grouped.setdefault(row["mode"], []).append(row)
    result = {}
    for mode, subset in grouped.items():
        summary = _summarize(subset, metric)
        summary["by_task_type"] = {task: _summarize([row for row in subset if row["task_type"] == task], metric)
                                   for task in sorted({row["task_type"] for row in subset})}
        summary["by_source"] = {source: _summarize([row for row in subset if row["source"] == source], metric)
                                for source in sorted({row["source"] for row in subset})}
        summary['by_evaluation_stratum'] = {s:_summarize([r for r in subset if r.get('evaluation_stratum','unspecified')==s],metric)
                                            for s in sorted({r.get('evaluation_stratum','unspecified') for r in subset})}
        result[mode] = summary
    return {"scope": "development_not_final_benchmark", "statistical_unit": "question_within_mode",
            "metric": metric, "by_mode": result,
            "limitations": "small_selected_validation_panel_not_test_accuracy_or_training_seed_uncertainty"}
