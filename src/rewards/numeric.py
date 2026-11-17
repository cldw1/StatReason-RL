"""严格标量答案判题：不执行模型生成的表达式，只解析白名单数值语法。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation, localcontext
import re
import unicodedata
from typing import Any

MARKER = "FINAL_ANSWER"
NUMBER = re.compile(r"[+-]?(?:(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]{1,3})?", re.ASCII)
MAX_NUMBER_CHARS = 128
MAX_COMPLETION_CHARS = 65536


class NumericParseError(ValueError):
    """数值不满足有限标量语法。"""


def _decimal_atom(token: str) -> Decimal:
    if not NUMBER.fullmatch(token):
        raise NumericParseError("invalid_numeric_syntax")
    if sum(char.isdigit() for char in token) > 64:
        raise NumericParseError("too_many_digits")
    if "e" in token.lower() and abs(int(token.lower().split("e")[1])) > 100:
        raise NumericParseError("exponent_out_of_range")
    try:
        value = Decimal(token.replace(",", ""))
    except InvalidOperation as exc:
        raise NumericParseError("invalid_decimal") from exc
    if not value.is_finite() or abs(value) > Decimal("1e100"):
        raise NumericParseError("nonfinite_or_out_of_range")
    return value


def parse_numeric(token: str, *, allow_percentage: bool = False) -> Decimal:
    """支持整数、小数、科学记数、规范千分位及分数；百分号必须由题目授权。

    百分数只进行明确的 /100 转换；不把无百分号的 25 自动当成 0.25。
    限制字符串长度、有效位和指数，防止大数或表达式消耗异常资源。
    """
    if not isinstance(token, str) or len(token) > MAX_NUMBER_CHARS:
        raise NumericParseError("invalid_numeric_length")
    token = token.strip(" \t")
    percentage = token.endswith("%")
    if percentage:
        if not allow_percentage:
            raise NumericParseError("percentage_not_allowed")
        token = token[:-1].strip(" \t")
    parts = token.split("/")
    if len(parts) > 2:
        raise NumericParseError("invalid_fraction")
    with localcontext() as context:
        context.prec = 80
        value = _decimal_atom(parts[0].strip(" \t"))
        if len(parts) == 2:
            denominator = _decimal_atom(parts[1].strip(" \t"))
            if denominator == 0:
                raise NumericParseError("zero_denominator")
            value /= denominator
        if percentage:
            value /= 100
        if not value.is_finite() or abs(value) > Decimal("1e100"):
            raise NumericParseError("nonfinite_or_out_of_range")
        return +value


@dataclass(frozen=True)
class Verification:
    """格式、解析、数值正确和严格正确分开记录。"""
    marker_count: int
    structure_valid: bool
    format_valid: bool
    parse_success: bool
    extracted_answer: str | None
    parsed_answer: str | None
    numeric_correct: bool
    strict_correct: bool
    truncated: bool
    reward: float
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


def _inside_explicit_thinking(text: str, position: int) -> bool:
    depth = 0
    for match in re.finditer(r"</?think>", text[:position], re.IGNORECASE):
        if match.group(0).lower() == "<think>":
            depth += 1
        else:
            # Qwen 的开始标签可能位于 prompt 中，completion 可只有闭合标签。
            depth = max(0, depth - 1)
    return depth > 0


def verify_completion(completion: str, record: dict[str, Any], *, truncated: bool = False) -> Verification:
    """只接受模型实际生成的 completion，调用者不得传入 prompt+completion。

    唯一标记必须独占最后一行，并且位于显式 think 块之外。单标记行后有额外
    内容时可保留数值诊断，但严格奖励为 0。多标记时不挑选其中的正确答案。
    截断样本即使含答案也严格判错；损失掩码和长度惩罚由训练器另行处理。
    """
    if not isinstance(completion, str):
        raise TypeError("completion 必须是生成文本字符串")
    expected = Decimal(str(record["answer"]))
    absolute = Decimal(str(record["absolute_tolerance"]))
    relative = Decimal(str(record["relative_tolerance"]))
    if not all(value.is_finite() for value in (expected, absolute, relative)) or min(absolute, relative) < 0:
        raise ValueError("标准答案和容差必须有限，容差不能为负")
    candidate = None
    parsed = None
    numeric_correct = False
    structure = False
    reason = "missing_final_marker"
    normalized = unicodedata.normalize("NFKC", completion)
    marker_count = len(re.findall(MARKER, normalized, re.IGNORECASE))
    if len(completion) > MAX_COMPLETION_CHARS:
        reason = "completion_too_long"
    elif any(unicodedata.category(char) in ("Cc", "Cf") and char not in "\n\r\t" for char in completion):
        reason = "forbidden_control_character"
    elif marker_count != 1:
        reason = "multiple_final_markers" if marker_count else "missing_final_marker"
    else:
        lines = completion.rstrip(" \t\r\n").lstrip("\r\n").splitlines()
        candidates = [(index, re.fullmatch(r"FINAL_ANSWER:[ \t]*(.+?)[ \t]*", line, re.ASCII)) for index, line in enumerate(lines)]
        matches = [(index, match) for index, match in candidates if match]
        if len(matches) != 1:
            reason = "invalid_final_line"
        else:
            index, match = matches[0]
            position = completion.find(MARKER)
            if _inside_explicit_thinking(completion, position):
                reason = "final_marker_inside_thinking"
            else:
                candidate = match.group(1)
                structure = index == len(lines) - 1
                try:
                    parsed = parse_numeric(candidate, allow_percentage=record.get("allow_percentage", False))
                    with localcontext() as context:
                        context.prec = 80
                        numeric_correct = abs(parsed - expected) <= max(absolute, relative * abs(expected))
                    reason = "correct" if numeric_correct else "wrong_numeric_answer"
                    if not structure:
                        reason = "content_after_final_answer"
                except NumericParseError as exc:
                    reason = str(exc)
    format_valid = structure and parsed is not None
    strict = format_valid and numeric_correct and not truncated
    if truncated:
        reason = "truncated_completion"
    return Verification(marker_count, structure, format_valid, parsed is not None, candidate,
                        str(parsed) if parsed is not None else None, numeric_correct,
                        strict, bool(truncated), float(strict), reason)


def completion_text(completion: str | list[dict]) -> str:
    """TRL 可传字符串或单条 assistant 消息，拒绝混入 user/system 消息。"""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and len(completion) == 1:
        message = completion[0]
        if isinstance(message, dict) and message.get("role") == "assistant" and isinstance(message.get("content"), str):
            return message["content"]
    raise TypeError("期望 completion 字符串或唯一 assistant 消息，不能传入完整对话")


def score_completions(completions: list, records: list[dict], *, truncated: list[bool] | None = None) -> list[Verification]:
    if len(completions) != len(records) or (truncated is not None and len(truncated) != len(records)):
        raise ValueError("completion、标准答案和截断标志长度必须一致")
    flags = truncated if truncated is not None else [False] * len(records)
    return [verify_completion(completion_text(text), record, truncated=flag) for text, record, flag in zip(completions, records, flags, strict=True)]

