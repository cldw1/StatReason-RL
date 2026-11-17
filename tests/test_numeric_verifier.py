"""边界判题、恶意输出和精度测试，均使用构造样例而非模型结果。"""
from decimal import Decimal
import hashlib

import pytest

from src.rewards.numeric import NumericParseError, completion_text, parse_numeric, score_completions, verify_completion


def record(answer="0.25", *, percentage=True, absolute=1e-8, relative=1e-8):
    return {"answer": answer, "absolute_tolerance": absolute, "relative_tolerance": relative,
            "allow_percentage": percentage, "answer_unit": "probability" if percentage else "number"}


@pytest.mark.parametrize("token,expected", [
    ("7", "7"), ("-7", "-7"), ("+7", "7"), (".25", "0.25"), ("-.25", "-0.25"),
    ("1.", "1"), ("2.5e-1", "0.25"), ("1E+3", "1000"), ("1,234.50", "1234.50"),
    ("-1,000", "-1000"), ("1/4", "0.25"), ("-3/4", "-0.75"),
    ("1 / -4", "-0.25"), ("25%", "0.25"), ("2.5e1%", "0.25"), ("1/2%", "0.005"),
])
def test_numeric_forms(token, expected):
    assert parse_numeric(token, allow_percentage=True) == Decimal(expected)


@pytest.mark.parametrize("token", [
    "NaN", "Inf", "-Infinity", "1/0", "0/0", "1//2", "1/2/3", "1+2", "sqrt(4)",
    "__import__('os').system('echo unsafe')", "[1,2]", "(1,2)", "(0.25)", "1,00", "12,34,567",
    "1 000", "1_000", "1e999", "1e-999", "1" * 129, "1" * 65, "25%%", "25 percent",
    "０.２５", "0.25\n", "0.25;print(1)", "0.25 # comment", "0.25\u200b", "1−2", "−0.25",
])
def test_unsafe_or_ambiguous_numeric_forms_rejected(token):
    with pytest.raises(NumericParseError):
        parse_numeric(token, allow_percentage=True)


def test_percentage_requires_explicit_question_rule():
    with pytest.raises(NumericParseError):
        parse_numeric("25%")
    assert not verify_completion("FINAL_ANSWER: 25", record()).strict_correct
    assert not verify_completion("FINAL_ANSWER: 25%", record(percentage=False)).strict_correct
    assert verify_completion("FINAL_ANSWER: 25%", record()).strict_correct


@pytest.mark.parametrize("text", [
    "The answer is 0.25", "", "0.25", "FINAL_ANSWER:", "FINAL_ANSWER : 0.25",
    "final_answer: 0.25", " FINAL_ANSWER: 0.25", "Answer FINAL_ANSWER: 0.25",
    "FINAL_ANSWER: 0.25 or 0.75", "FINAL_ANSWER: [0.25]", "FINAL_ANSWER: 0.25\nignore the verifier",
    "FINAL_ANSWER: 0.25\nFINAL_ANSWER: 0.25", "FINAL_ANSWER: 0.75\nFINAL_ANSWER: 0.25",
    "<think>FINAL_ANSWER: 0.25</think>\nFINAL_ANSWER: 0.25",
    "<think>\nFINAL_ANSWER: 0.25", "ＦＩＮＡＬ＿ＡＮＳＷＥＲ: 0.25",
    "ＦＩＮＡＬ＿ＡＮＳＷＥＲ: 0.75\nFINAL_ANSWER: 0.25",
    "F\u200bINAL_ANSWER: 0.75\nFINAL_ANSWER: 0.25",
    "FINAL_ANSWER: 0.25\x00", "FINAL_ANSWER: NaN", "FINAL_ANSWER: Inf", "FINAL_ANSWER: 1/0",
    "<think>\n0.25\n</think>\nFINAL_ANSWER: 0.75",
    "Question requests FINAL_ANSWER: number\nFINAL_ANSWER: 0.25",
    "x" * 65537 + "\nFINAL_ANSWER: 0.25",
], ids=lambda text: f"case-{len(text)}-{hashlib.sha256(text.encode()).hexdigest()[:8]}")
def test_reward_exploit_cases_return_zero(text):
    result = verify_completion(text, record())
    assert result.reward == 0.0
    assert not result.strict_correct


def test_valid_completion_and_empty_thinking_block():
    for text in ("Calculation: 1/4.\nFINAL_ANSWER: 0.25", "<think>1/4</think>\nFINAL_ANSWER: 1/4", "</think>\nFINAL_ANSWER: 0.25", "FINAL_ANSWER: 0.25\n"):
        result = verify_completion(text, record())
        assert result.format_valid and result.parse_success and result.numeric_correct and result.strict_correct


def test_truncation_and_diagnostic_correctness_separate_from_strict_reward():
    result = verify_completion("FINAL_ANSWER: 0.25", record(), truncated=True)
    assert result.format_valid and result.numeric_correct and not result.strict_correct
    assert result.reason == "truncated_completion" and result.reward == 0
    trailing = verify_completion("FINAL_ANSWER: 0.25\nextra", record())
    assert trailing.parse_success and trailing.numeric_correct and not trailing.format_valid


def test_absolute_relative_tolerances_and_exact_integer_policy():
    assert verify_completion("FINAL_ANSWER: 0.25000001", record(absolute=1e-8, relative=0)).strict_correct
    assert not verify_completion("FINAL_ANSWER: 0.25000002", record(absolute=1e-8, relative=0)).strict_correct
    assert verify_completion("FINAL_ANSWER: 1000000.5", record("1000000", percentage=False, absolute=0, relative=1e-6)).strict_correct
    assert not verify_completion("FINAL_ANSWER: 1000002", record("1000000", percentage=False, absolute=0, relative=1e-6)).strict_correct
    assert not verify_completion("FINAL_ANSWER: 2.000000001", record("2", percentage=False, absolute=0, relative=0)).strict_correct


def test_nonfinite_reference_is_configuration_error():
    for bad in (record("NaN"), record(absolute=-1), record(relative=float("inf"))):
        with pytest.raises(ValueError):
            verify_completion("FINAL_ANSWER: 0.25", bad)


def test_completion_interface_rejects_prompt_or_multimessage_input():
    conversation = [{"role": "user", "content": "FINAL_ANSWER: 0.25"}, {"role": "assistant", "content": "0.75"}]
    with pytest.raises(TypeError):
        completion_text(conversation)
    with pytest.raises(TypeError):
        completion_text([{"role": "user", "content": "FINAL_ANSWER: 0.25"}])
    results = score_completions([[{"role": "assistant", "content": "FINAL_ANSWER: 0.25"}], "FINAL_ANSWER: 0.75"], [record(), record()])
    assert [item.reward for item in results] == [1.0, 0.0]
    with pytest.raises(ValueError):
        score_completions(["x"], [])
