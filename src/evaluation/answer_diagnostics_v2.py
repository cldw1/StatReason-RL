"""事后诊断v2：兼容唯一末尾boxed后的数学闭合符与句号，不改严格判题。

冻结的生成记录继续保存v1结果。此模块用于所有模型的统一重分析，
不参与采样、训练奖励、原始身份签名或主严格分数。
"""
from __future__ import annotations

import re

from .answer_diagnostics import diagnose_answer


PROTOCOL = 'terminal_boxed_v2_math_delimiter_and_punctuation'
ALLOWED_ENDINGS = {delimiter + punctuation
                   for delimiter in ('', '$', '$$', r'\)', r'\]')
                   for punctuation in ('', '.', '。')}


def diagnose_answer_v2(completion: str, record: dict, *, truncated=False, thinking=False) -> dict:
    original = diagnose_answer(completion, record, truncated=truncated, thinking=thinking)
    result = {**original, 'posthoc_protocol': PROTOCOL, 'strict_reward_changed': False,
              'normalized_terminal_closure': None}
    response = original['final_response']
    if original['diagnostic_parse_success'] or re.search('FINAL_ANSWER', response, re.I):
        return result
    boxes = list(re.finditer(r'\\boxed\s*\{', response))
    if len(boxes) != 1:
        return result
    start = boxes[0].end()
    depth, end = 1, start
    while end < len(response) and depth:
        if response[end] == '{':
            depth += 1
        elif response[end] == '}':
            depth -= 1
        end += 1
    ending = response[end:].strip()
    if depth != 0 or ending not in ALLOWED_ENDINGS:
        return result
    normalized = diagnose_answer(response[:end], record, truncated=truncated, thinking=False)
    for key in ('diagnostic_candidate', 'diagnostic_source', 'diagnostic_parsed', 'diagnostic_parse_success',
                'numeric_correct_at_same_tolerance', 'candidate_correct_ignoring_truncation', 'diagnostic_reason'):
        result[key] = normalized[key]
    result['normalized_terminal_closure'] = ending
    return result
