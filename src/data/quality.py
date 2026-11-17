"""独立数值复算和训练题抽查材料；不评测模型、不展示测试题答案。"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
from scipy.stats import binom, norm, t

from src.rewards.numeric import verify_completion
from .statistics import TASK_TYPES


def independent_answer(record: dict) -> float:
    """用不同数值路径复算，例如生成器 Fraction 求和对照 SciPy binom。"""
    p = record["parameters"]
    variant, task = p["variant"], record["task_type"]
    if task == "conditional_probability":
        if variant == 0:
            return float(Fraction(p["joint_count"], p["b_count"]))
        if variant == 1:
            return p["joint_percent"] / p["marginal_percent"]
        if variant == 2:
            counts = p["counts_ab_notab_anotb_neither"]
            return counts[0] / sum(counts[:2])
        return 1 - p["blue"] / (p["red"] + p["blue"] - 1)
    if task == "bayes":
        if variant == 0:
            joint = np.array([p["prior"] * p["likelihood_a"], (1 - p["prior"]) * p["likelihood_b"]])
            return float((joint / joint.sum())[0])
        if variant == 1:
            joint = np.array(p["source_weights"]) * np.array(p["event_rates"])
            return float((joint / joint.sum())[1])
        if variant == 2:
            numerator = p["odds_numerator"] * p["likelihood_ratio_numerator"]
            other = p["odds_denominator"] * p["likelihood_ratio_denominator"]
            return numerator / (numerator + other)
        joint = [p["prior"] * p["s1"] * p["s2"], (1 - p["prior"]) * p["f1"] * p["f2"]]
        return joint[0] / math.fsum(joint)
    if task == "binomial_probability":
        n, probability = p["n"], p["p"]
        if variant == 0:
            return float(binom.pmf(p["k"], n, probability))
        if variant == 1:
            return float(binom.cdf(p["k"], n, probability))
        if variant == 2:
            return float(binom.sf(p["k"] - 1, n, probability))
        return float(binom.cdf(p["upper"], n, probability) - binom.cdf(p["lower"] - 1, n, probability))
    if task == "discrete_expectation" and variant == 3:
        outcomes = [pair[0] + 2 * pair[1] for pair in p["pairs"]]
        return float(np.average(outcomes, weights=p["joint_weights"]))
    if task == "discrete_variance" and variant == 1:
        second = Fraction(p["second_numerator"], p["second_denominator"])
        mean = Fraction(p["mean_numerator"], p["mean_denominator"])
        return float(second - mean ** 2)
    if task == "discrete_variance" and variant == 3:
        return float(np.average((np.array([0, 1]) - p["p"]) ** 2, weights=[1 - p["p"], p["p"]]))
    if task in ("discrete_expectation", "discrete_variance"):
        values, weights = np.array(p["values"], dtype=float), np.array(p["weights"], dtype=float)
        mean = np.average(values, weights=weights)
        if variant == 2:
            values = p["coefficient"] * values + p["intercept"]
            mean = np.average(values, weights=weights)
        return float(mean if task == "discrete_expectation" else np.average((values - mean) ** 2, weights=weights))
    if task == "standard_error":
        if variant == 0:
            return math.sqrt(p["population_variance"]) / math.sqrt(p["n"])
        if variant == 1:
            return float(np.linalg.norm([p["sigma"] / math.sqrt(p["n"]), p["sigma2"] / math.sqrt(p["n2"])]))
        if variant == 2:
            return math.sqrt(p["coefficient"] ** 2 * p["sigma"] ** 2 / p["n"])
        variance_of_sum = p["n"] * p["sigma"] ** 2 + p["n"] * (p["n"] - 1) * p["rho"] * p["sigma"] ** 2
        return math.sqrt(variance_of_sum) / p["n"]
    if task == "normal_mean_ci_endpoint":
        confidence, n, mean = p["confidence"], p["n"], p["mean"]
        if variant == 0:
            return float(norm.interval(confidence, loc=mean, scale=p["sigma"] / math.sqrt(n))[0])
        if variant == 1:
            return float(t.interval(confidence, df=n - 1, loc=mean, scale=p["sample_standard_deviation"] / math.sqrt(n))[1])
        if variant == 2:
            scale = math.hypot(p["sigma"] / math.sqrt(n), p["sigma2"] / math.sqrt(p["n2"]))
            return float(norm.interval(confidence, loc=mean - p["mean2"], scale=scale)[0])
        return float(norm.isf(1 - confidence, loc=mean, scale=p["sigma"] / math.sqrt(n)))
    raise ValueError("无独立复算函数")


def audit_statistics(records: list[dict]) -> dict:
    max_error = 0.0
    failures = []
    for record in records:
        expected = float(record["answer"])
        independent = independent_answer(record)
        error = abs(expected - independent)
        max_error = max(max_error, error)
        tolerance = max(record["absolute_tolerance"], record["relative_tolerance"] * abs(expected))
        if error > tolerance or not verify_completion(record["solution"], record).strict_correct:
            failures.append(record["id"])
    if failures:
        raise ValueError(f"独立复算或标准解答判题失败: {failures[:10]}")
    return {"records_checked": len(records), "independent_recompute_failures": len(failures),
            "reference_completion_failures": 0, "max_absolute_recompute_error": max_error,
            "method": "independent_numpy_scipy_paths_and_reference_completion_verification",
            "scope": "programmatic_check_not_human_review_or_model_accuracy"}


def training_review_markdown(records: list[dict]) -> str:
    lines = ["# 统计题训练样本抽查", "", "仅展示 SFT/RL 训练题，每题型各选一条。程序复算不等于人工审题完成；这里是待人工复核材料。", ""]
    for task in TASK_TYPES:
        for split in ("sft", "rl"):
            record = next((record for record in records if record["task_type"] == task and record["split"] == split), None)
            if record:
                lines.extend([f"## {task} / {split}", "", f"ID：{record['id']}", "", record["question"], "", record["solution"], "", f"独立复算：{independent_answer(record):.12g}", ""])
    return "\n".join(lines)

