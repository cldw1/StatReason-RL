"""确定性统计题生成器：公式计算标准答案，按数学模板族隔离划分。"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import math
import random

from scipy.stats import norm, t

from .schema import canonical_json, question_key, validate_records

TASK_TYPES = (
    "conditional_probability", "bayes", "binomial_probability", "discrete_expectation",
    "discrete_variance", "standard_error", "normal_mean_ci_endpoint",
)
FAMILY_DESCRIPTIONS = {
    "conditional_probability": ["conditional_count", "joint_and_marginal", "two_way_table", "without_replacement"],
    "bayes": ["two_source_mixture", "three_source_mixture", "prior_odds_likelihood_ratio", "two_conditional_independent_signals"],
    "binomial_probability": ["point_probability", "cdf_at_most", "upper_tail", "inclusive_interval"],
    "discrete_expectation": ["pmf_mean", "finite_population_mean", "affine_transform_mean", "joint_linear_mean"],
    "discrete_variance": ["pmf_variance", "first_second_moments", "affine_transform_variance", "bernoulli_indicator"],
    "standard_error": ["known_variance_mean", "difference_independent_means", "affine_mean", "equicorrelated_mean"],
    "normal_mean_ci_endpoint": ["known_sigma_two_sided_lower", "unknown_sigma_t_upper", "independent_means_known_variances_lower", "known_sigma_one_sided_upper"],
}
SPLIT_VARIANTS = {"sft": 0, "rl": 1, "validation": 2, "test": 3, "prototype": 0}


def _probability(rng: random.Random) -> Fraction:
    return Fraction(rng.randint(10, 90), 100)


def _decimal(value) -> str:
    return f"{float(value):.2f}"


def _distribution(rng: random.Random):
    values = sorted(rng.sample(range(-5, 17), 3))
    weights = [rng.randint(1, 9) for _ in values]
    total = sum(weights)
    probabilities = [Fraction(weight, total) for weight in weights]
    mean = sum((value * probability for value, probability in zip(values, probabilities, strict=True)), Fraction(0))
    second = sum((value * value * probability for value, probability in zip(values, probabilities, strict=True)), Fraction(0))
    return values, weights, probabilities, mean, second - mean * mean


def _make_task(task: str, variant: int, rng: random.Random) -> tuple[dict, str, str, float, str]:
    params = {"variant": variant}
    unit = "probability" if task in TASK_TYPES[:3] else "number"
    if task == "conditional_probability":
        if variant == 0:
            total_b = rng.randint(30, 300)
            joint = rng.randint(2, total_b - 2)
            params.update(b_count=total_b, joint_count=joint)
            question = f"在一个有限总体中，满足事件 B 的对象有 {total_b} 个，其中同时满足 A 的有 {joint} 个。从满足 B 的对象中等概率抽取一个，求 P(A|B)。"
            answer = joint / total_b
            solution = f"条件样本空间是 B 内的 {total_b} 个对象，P(A|B)={joint}/{total_b}。"
        elif variant == 1:
            marginal = rng.randint(25, 90)
            joint = rng.randint(1, marginal - 1)
            params.update(marginal_percent=marginal, joint_percent=joint)
            question = f"给定 P(B)={marginal}%，P(A∩B)={joint}%，求条件概率 P(A|B)。"
            answer = joint / marginal
            solution = f"P(A|B)=P(A∩B)/P(B)=({joint}/100)/({marginal}/100)。"
        elif variant == 2:
            counts = [rng.randint(5, 90) for _ in range(4)]
            ab, not_a_b, a_not_b, neither = counts
            params["counts_ab_notab_anotb_neither"] = counts
            question = f"一个分类表有四格计数：A且B={ab}，非A且B={not_a_b}，A且非B={a_not_b}，非A且非B={neither}。从全体等概率选一个，已知它属于 B，求它属于 A 的概率。"
            answer = ab / (ab + not_a_b)
            solution = f"在 B 条件下只保留前两格，概率={ab}/({ab}+{not_a_b})。"
        else:
            red, blue = rng.randint(3, 60), rng.randint(3, 60)
            params.update(red=red, blue=blue)
            question = f"袋中有 {red} 个红球和 {blue} 个蓝球，逐个等概率不放回抽取两个球。已知第一个是红球，第二个仍是红球的概率是多少？"
            answer = (red - 1) / (red + blue - 1)
            solution = f"抽走一个红球后剩余红球 {red-1} 个、总球 {red+blue-1} 个，条件概率=({red}-1)/({red}+{blue}-1)。"
    elif task == "bayes":
        if variant == 0:
            prior, likelihood_a, likelihood_b = [_probability(rng) for _ in range(3)]
            params.update(prior=float(prior), likelihood_a=float(likelihood_a), likelihood_b=float(likelihood_b))
            question = f"一批物品来自 A、B 两条产线，来自 A 的比例为 {_decimal(prior)}。事件 E 在 A、B 产线的条件概率分别为 {_decimal(likelihood_a)} 和 {_decimal(likelihood_b)}。随机物品发生 E 后，求其来自 A 的后验概率。"
            evidence = prior * likelihood_a + (1 - prior) * likelihood_b
            answer = prior * likelihood_a / evidence
            solution = f"由全概率公式 P(E)=P(A)P(E|A)+P(B)P(E|B)，贝叶斯公式 P(A|E)=P(A)P(E|A)/P(E)，其中 P(E)={float(evidence):.10f}。"
        elif variant == 1:
            weights = [rng.randint(1, 9) for _ in range(3)]
            rates = [_probability(rng) for _ in range(3)]
            params.update(source_weights=weights, event_rates=list(map(float, rates)))
            question = f"从三个来源 A、B、C 抽取物品的概率之比为 {weights[0]}:{weights[1]}:{weights[2]}，事件 E 的三个条件概率依次是 {', '.join(_decimal(rate) for rate in rates)}。已观察到 E，求来源为 B 的后验概率。"
            answer = weights[1] * rates[1] / sum(weight * rate for weight, rate in zip(weights, rates, strict=True))
            solution = "用来源权重乘各自似然，B 的权重似然除以三个来源权重似然之和；共同归一化系数相消。"
        elif variant == 2:
            odds_numerator, odds_denominator = rng.randint(1, 12), rng.randint(1, 12)
            ratio = Fraction(rng.randint(1, 30), rng.randint(1, 10))
            params.update(odds_numerator=odds_numerator, odds_denominator=odds_denominator, likelihood_ratio_numerator=ratio.numerator, likelihood_ratio_denominator=ratio.denominator)
            question = f"假设 H 的先验赔率 P(H):P(非H)={odds_numerator}:{odds_denominator}。观察 E 的似然比 P(E|H)/P(E|非H)={ratio.numerator}/{ratio.denominator}。求后验概率 P(H|E)。"
            odds = Fraction(odds_numerator, odds_denominator) * ratio
            answer = odds / (1 + odds)
            solution = f"后验赔率=先验赔率×似然比={odds}，再用概率=赔率/(1+赔率)。"
        else:
            prior, s1, s2, f1, f2 = [_probability(rng) for _ in range(5)]
            params.update(prior=float(prior), s1=float(s1), s2=float(s2), f1=float(f1), f2=float(f2))
            question = f"P(H)={_decimal(prior)}。给定 H 时，事件 E1、E2 条件独立，概率为 {_decimal(s1)}、{_decimal(s2)}；给定非H时它们也条件独立，概率为 {_decimal(f1)}、{_decimal(f2)}。已同时观察 E1 和 E2，求 P(H|E1∩E2)。"
            numerator = prior * s1 * s2
            answer = numerator / (numerator + (1 - prior) * f1 * f2)
            solution = "分别在 H 和非H条件下将两个似然相乘，再用贝叶斯公式归一化；不假设两个事件无条件独立。"
    elif task == "binomial_probability":
        n = rng.randint(6, 20)
        probability = Fraction(rng.randint(2, 8), 10)
        center = max(1, min(n - 1, round(n * probability)))
        k = max(1, min(n - 1, center + rng.randint(-1, 1)))
        lower, upper = max(0, center - rng.randint(1, 3)), min(n, center + rng.randint(1, 3))
        params.update(n=n, p=float(probability))
        def point(index):
            return math.comb(n, index) * probability ** index * (1 - probability) ** (n - index)
        if variant == 0:
            params["k"] = k
            question = f"进行 {n} 次相互独立的伯努利试验，每次成功概率为 {_decimal(probability)}。令 X 为成功次数，求 P(X={k})。"
            answer = point(k)
            solution = f"X~Bin({n},{_decimal(probability)})，使用二项分布点概率 C({n},{k})p^{k}(1-p)^{n-k}。"
        elif variant == 1:
            params["k"] = k
            question = f"X 服从二项分布 Bin(n={n},p={_decimal(probability)})。求成功次数不超过 {k} 的概率 P(X≤{k})。"
            answer = sum(point(index) for index in range(k + 1))
            solution = f"将二项分布从 0 到 {k} 的点概率求和，包含端点 {k}。"
        elif variant == 2:
            params["k"] = k
            question = f"一个随机变量 X~Bin({n},{_decimal(probability)})，求至少出现 {k} 次成功的概率 P(X≥{k})。"
            answer = sum(point(index) for index in range(k, n + 1))
            solution = f"计算 1-P(X≤{k-1})，或从 {k} 到 {n} 对点概率求和。"
        else:
            params.update(lower=lower, upper=upper)
            question = f"X~Bin({n},{_decimal(probability)})。求 {lower}≤X≤{upper} 的概率，两个端点均包含。"
            answer = sum(point(index) for index in range(lower, upper + 1))
            solution = f"将 k={lower},...,{upper} 的二项点概率相加，区间为闭区间。"
    elif task in ("discrete_expectation", "discrete_variance"):
        values, weights, probabilities, mean, variance = _distribution(rng)
        params.update(values=values, weights=weights)
        distribution_text = "，".join(f"P(X={value})={probability.numerator}/{probability.denominator}" for value, probability in zip(values, probabilities, strict=True))
        if task == "discrete_expectation":
            if variant == 0:
                question, answer = f"离散变量 X 的分布为：{distribution_text}。求 E(X)。", mean
                solution = "期望为各取值与对应概率乘积之和：E(X)=ΣxP(X=x)。"
            elif variant == 1:
                question = f"盒中三种数值标签依次为 {values}，各标签的张数依次为 {weights}。等概率抽取一张，求标签数值的期望。"
                answer = mean
                solution = f"每类标签概率=张数/{sum(weights)}，因此按张数加权求平均。"
            elif variant == 2:
                coefficient, intercept = rng.choice([-3, -2, 2, 3]), rng.randint(-8, 8)
                params.update(coefficient=coefficient, intercept=intercept)
                question = f"已知 {distribution_text}。令 Y={coefficient}X+({intercept})，求 E(Y)。"
                answer = coefficient * mean + intercept
                solution = f"先计算 E(X)={float(mean):.10f}，再使用 E(aX+b)=aE(X)+b。"
            else:
                pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
                joint_weights = [rng.randint(1, 12) for _ in pairs]
                params = {"variant": variant, "pairs": pairs, "joint_weights": joint_weights}
                question = f"(X,Y) 仅取 (0,0)、(0,1)、(1,0)、(1,1)，对应概率之比为 {':'.join(map(str,joint_weights))}。求 E(X+2Y)。"
                answer = Fraction(sum((x + 2 * y) * weight for (x, y), weight in zip(pairs, joint_weights, strict=True)), sum(joint_weights))
                solution = "对四个联合取值分别计算 x+2y，再按归一化后的联合权重求加权和，不需要独立性。"
        elif variant == 0:
            question, answer = f"离散变量 X 的分布为：{distribution_text}。求 Var(X)，使用总体方差定义。", variance
            solution = f"E(X)={float(mean):.10f}，Var(X)=Σp(x)(x-E(X))²=E(X²)-[E(X)]²。"
        elif variant == 1:
            second = variance + mean * mean
            params = {"variant": variant, "mean_numerator": mean.numerator, "mean_denominator": mean.denominator, "second_numerator": second.numerator, "second_denominator": second.denominator}
            question = f"已知 E(X)={mean.numerator}/{mean.denominator}，E(X²)={second.numerator}/{second.denominator}。求总体方差 Var(X)。"
            answer, solution = variance, "用 Var(X)=E(X²)-[E(X)]²，注意均值需要先平方。"
        elif variant == 2:
            coefficient, intercept = rng.choice([-4, -2, 2, 4]), rng.randint(-8, 8)
            params.update(coefficient=coefficient, intercept=intercept)
            question = f"{distribution_text}。令 Y={coefficient}X+({intercept})，求 Var(Y)。"
            answer = coefficient ** 2 * variance
            solution = f"计算 Var(X)={float(variance):.10f}，再用 Var(aX+b)=a²Var(X)，平移项不改变方差。"
        else:
            probability = _probability(rng)
            params = {"variant": variant, "p": float(probability)}
            question = f"事件 A 发生的概率为 {_decimal(probability)}。I_A 为其指示变量（发生取1，否则取0），求 Var(I_A)。"
            answer, solution = probability * (1 - probability), "I_A 服从伯努利分布，E(I_A)=p 且 I_A²=I_A，所以方差为 p(1-p)。"
    elif task == "standard_error":
        n, sigma = rng.randint(8, 180), rng.randint(5, 200) / 10
        params.update(n=n, sigma=sigma)
        if variant == 0:
            variance = Fraction(round(sigma * 10) ** 2, 100)
            params["population_variance"] = float(variance)
            question = f"X1,...,X{n} 相互独立同分布，总体方差为 {float(variance):.2f}。求样本均值的标准误（标准差）。"
            answer = math.sqrt(float(variance) / n)
            solution = "独立同分布条件下 Var(X̄)=σ²/n，标准误为 sqrt(σ²/n)。"
        elif variant == 1:
            n2, sigma2 = rng.randint(8, 180), rng.randint(5, 200) / 10
            params.update(n2=n2, sigma2=sigma2)
            question = f"两组样本彼此独立，组内独立同分布。第一组 n1={n}、总体标准差 σ1={sigma:.1f}；第二组 n2={n2}、σ2={sigma2:.1f}。求两个样本均值之差的标准误。"
            answer = math.sqrt(sigma * sigma / n + sigma2 * sigma2 / n2)
            solution = "独立时均值之差的方差为 σ1²/n1+σ2²/n2，取平方根得到标准误。"
        elif variant == 2:
            coefficient, intercept = rng.choice([-3, -2, 2, 3]), rng.randint(-8, 8)
            params.update(coefficient=coefficient, intercept=intercept)
            question = f"X1,...,X{n} 独立同分布，总体标准差 σ={sigma:.1f}。令 Yi={coefficient}Xi+({intercept})。求 Y 的样本均值的标准误。"
            answer = abs(coefficient) * sigma / math.sqrt(n)
            solution = "Y 的标准差为 |a|σ，样本均值标准误为 |a|σ/sqrt(n)。"
        else:
            rho = rng.randint(1, 30) / 100
            params["rho"] = rho
            question = f"{n} 个变量具有共同方差 {sigma:.1f}²，任意两个不同变量的相关系数均为 {rho:.2f}，它们并不独立。求这 {n} 个变量的算术平均值的标准误。"
            answer = sigma * math.sqrt((1 + (n - 1) * rho) / n)
            solution = "将所有方差与协方差相加再除以 n²，Var(X̄)=σ²[1+(n-1)ρ]/n，最后取平方根。"
    elif task == "normal_mean_ci_endpoint":
        n, mean, sigma = rng.randint(8, 90), rng.randint(-200, 900) / 10, rng.randint(5, 180) / 10
        confidence = rng.choice([0.9, 0.95, 0.99])
        alpha = 1 - confidence
        params.update(n=n, mean=mean, sigma=sigma, confidence=confidence)
        if variant == 0:
            critical = float(norm.ppf(1 - alpha / 2))
            question = f"独立样本来自 N(μ,σ²)，已知总体标准差 σ={sigma:.1f}，n={n}，样本均值为 {mean:.1f}。求 μ 的双侧 {confidence:.0%} 置信区间下限。给定标准正态临界值 z={critical:.10f}。"
            answer = mean - critical * sigma / math.sqrt(n)
            solution = "总体正态且 σ 已知，双侧区间下限为 x̄-zσ/sqrt(n)。"
        elif variant == 1:
            critical = float(t.ppf(1 - alpha / 2, df=n - 1))
            params["sample_standard_deviation"] = params.pop("sigma")
            question = f"独立样本来自正态总体，σ 未知。n={n}，样本均值 {mean:.1f}，按 n-1 作分母的样本标准差 s={sigma:.1f}。求 μ 的双侧 {confidence:.0%} t 置信区间上限，给定 df={n-1} 的临界值 t={critical:.10f}。"
            answer = mean + critical * sigma / math.sqrt(n)
            solution = "正态总体、σ未知，使用 t_(n-1) 临界值，上限=x̄+t×s/sqrt(n)，s 按 n-1 定义。"
        elif variant == 2:
            n2, mean2, sigma2 = rng.randint(8, 90), rng.randint(-200, 900) / 10, rng.randint(5, 180) / 10
            params.update(n2=n2, mean2=mean2, sigma2=sigma2)
            critical = float(norm.ppf(1 - alpha / 2))
            question = f"两组独立样本分别来自正态总体，总体标准差已知。n1={n}、x̄1={mean:.1f}、σ1={sigma:.1f}；n2={n2}、x̄2={mean2:.1f}、σ2={sigma2:.1f}。求 μ1-μ2 的双侧 {confidence:.0%} 置信区间下限，给定 z={critical:.10f}。"
            answer = mean - mean2 - critical * math.sqrt(sigma * sigma / n + sigma2 * sigma2 / n2)
            solution = "均值差减去 z 倍标准误，标准误为 sqrt(σ1²/n1+σ2²/n2)。"
        else:
            critical = float(norm.ppf(confidence))
            question = f"独立样本来自 N(μ,σ²)，σ={sigma:.1f} 已知，n={n}，x̄={mean:.1f}。求 μ 的单侧上置信限 U，使 (-∞,U] 的覆盖率为 {confidence:.0%}。给定 z_({confidence})={critical:.10f}。"
            answer = mean + critical * sigma / math.sqrt(n)
            solution = "单侧上限使用 z_(1-α)，不是双侧的 z_(1-α/2)：U=x̄+z_(1-α)σ/sqrt(n)。"
        params["critical_value"] = critical
    else:
        raise ValueError(f"未知题型: {task}")
    return params, question, solution, float(answer), unit


def generate_statistics(counts: dict[str, int], *, seed: int = 42) -> list[dict]:
    """每个 split 各有七种题型；split 和题型使用独立 seed，便于版本复现。

    四个正式 split 分别对应每题型的四个不同数学模板。该方案是模板族外推
    评测，不能简单解释成同分布随机划分的正确率。
    """
    records, questions, parameter_keys = [], set(), set()
    for split, count in counts.items():
        if split not in SPLIT_VARIANTS or count < 0:
            raise ValueError("未知划分或负样本数")
        variant = SPLIT_VARIANTS[split]
        for task_index, task in enumerate(TASK_TYPES):
            task_count = count // len(TASK_TYPES) + int(task_index < count % len(TASK_TYPES))
            seed_text = f"statreason-v1:{seed}:{split}:{task}"
            rng = random.Random(int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:8], "big"))
            for _ in range(task_count):
                for _attempt in range(10000):
                    parameters, question, solution, answer, unit = _make_task(task, variant, rng)
                    key = question_key(question)
                    parameter_key = (task, canonical_json(parameters))
                    if key not in questions and parameter_key not in parameter_keys and (unit != "probability" or 1e-5 < answer < 1 - 1e-5):
                        break
                else:
                    raise RuntimeError(f"{task}/{split} 唯一题目空间不足")
                questions.add(key)
                parameter_keys.add(parameter_key)
                final_answer = f"{answer:.8f}".rstrip("0").rstrip(".")
                question += " 最终答案请给出数值，至少保留8位小数（精确分数也可）。"
                if unit == "probability":
                    question += "概率可写为0到1的小数，或带%符号的百分数。"
                digest = hashlib.sha256((task + ":" + canonical_json(parameters)).encode()).hexdigest()[:20]
                family = FAMILY_DESCRIPTIONS[task][variant]
                records.append({
                    "id": f"stat-{split}-{digest}", "source": "statreason_synthetic", "task_type": task,
                    "template_family": f"{task}/{family}" + ("/prototype_only" if split == "prototype" else ""),
                    "question": question, "parameters": parameters, "solution": solution + f"\nFINAL_ANSWER: {final_answer}",
                    "answer": format(answer, ".15g"), "answer_type": "scalar", "answer_unit": unit,
                    "absolute_tolerance": 1e-8 if unit == "probability" else 1e-6,
                    "relative_tolerance": 1e-8, "allow_percentage": unit == "probability", "split": split,
                })
    validate_records(records)
    return records
