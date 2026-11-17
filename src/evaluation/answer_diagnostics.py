"""仅诊断明确的最终答案，不改变 numeric.py 中的原始二元奖励。"""
from __future__ import annotations

from decimal import Decimal, localcontext
import re

from src.rewards.numeric import NumericParseError, parse_numeric, verify_completion


def final_response(completion: str, *, thinking: bool) -> tuple[str, str]:
    closing = list(re.finditer(r'</think>', completion, re.I))
    if thinking:
        if len(closing) != 1:
            return '', 'unclosed_or_ambiguous_thinking'
        answer = completion[closing[0].end():].strip()
        if re.search(r'</?think>', answer, re.I):
            return '', 'ambiguous_thinking'
        return answer, 'after_thinking_close'
    if re.search(r'</?think>', completion, re.I):
        return '', 'unexpected_thinking_tags'
    return completion.strip(), 'non_thinking_response'


def _latex_scalar(text: str) -> str:
    text = text.strip()
    if text.startswith('$') and text.endswith('$'):
        text = text[1:-1].strip()
    text = text.replace(r'\%', '%')
    match = re.fullmatch(r'([+-]?)\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}', text)
    if match:
        sign, numerator, denominator = match.groups()
        numerator_value = parse_numeric(numerator)
        if sign == '-':
            numerator_value = -numerator_value
        return f'{numerator_value}/{denominator}'
    return text


def diagnose_answer(completion: str, record: dict, *, truncated: bool = False,
                    thinking: bool = False) -> dict:
    """只读唯一末尾标记或 boxed，不从过程选数；容差保持原值。"""
    response, boundary = final_response(completion, thinking=thinking)
    strict = verify_completion(response, record, truncated=truncated)
    markers = list(re.finditer('FINAL_ANSWER', response, re.I))
    boxes = list(re.finditer(r'\\boxed\s*\{', response))
    candidate, source, failure = None, None, 'no_unique_terminal_answer'
    if len(markers) == 1 and not boxes:
        match = re.search(r'FINAL_ANSWER:[ \t]*([^\n\r]+)\s*\Z', response)
        if match:
            candidate, source = match.group(1).strip(), 'terminal_marker'
    elif not markers and len(boxes) == 1:
        start = boxes[0].end()
        depth, end = 1, start
        while end < len(response) and depth:
            if response[end] == '{':
                depth += 1
            elif response[end] == '}':
                depth -= 1
            end += 1
        if depth == 0 and response[end:].strip() in ('', '$', '$$', '.', '。', r'\)', r'\]'):
            candidate, source = response[start:end - 1], 'terminal_boxed'
    parsed, correct = None, False
    if candidate is not None:
        try:
            parsed = parse_numeric(_latex_scalar(candidate), allow_percentage=record.get('allow_percentage', False))
            with localcontext() as context:
                context.prec = 80
                expected = Decimal(str(record['answer']))
                tolerance = max(Decimal(str(record['absolute_tolerance'])),
                                Decimal(str(record['relative_tolerance'])) * abs(expected))
                correct = abs(parsed - expected) <= tolerance
            failure = 'correct' if correct else 'wrong_numeric_answer'
        except NumericParseError as error:
            failure = str(error)
    return {'strict': strict.to_dict(), 'final_response': response, 'response_boundary': boundary,
            'diagnostic_candidate': candidate, 'diagnostic_source': source,
            'diagnostic_parsed': str(parsed) if parsed is not None else None,
            'diagnostic_parse_success': parsed is not None,
            'numeric_correct_at_same_tolerance': correct and not truncated,
            'candidate_correct_ignoring_truncation': correct,
            'diagnostic_reason': failure, 'tolerance_changed': False,
            'diagnostic_only_not_training_reward': True}
