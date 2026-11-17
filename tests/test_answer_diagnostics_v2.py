import pytest

from src.evaluation.answer_diagnostics import diagnose_answer
from src.evaluation.answer_diagnostics_v2 import diagnose_answer_v2


def record():
    return {'answer': '37', 'absolute_tolerance': 0, 'relative_tolerance': 0, 'allow_percentage': False}


@pytest.mark.parametrize('text', [r'Thus \(\boxed{37}\).', r'\[\boxed{37}\].', r'$\boxed{37}$.',
                                 r'$$\boxed{37}$$。', r'\boxed{\frac{74}{2}}\).'])
def test_math_closure_and_sentence_punctuation_remain_diagnostic_only(text):
    result = diagnose_answer_v2(text, record())
    assert result['numeric_correct_at_same_tolerance']
    assert result['strict'] == diagnose_answer(text, record())['strict']
    assert not result['strict']['strict_correct']
    assert not result['strict_reward_changed']


@pytest.mark.parametrize('text', [r'\boxed{37}\). or 38', r'\boxed{37} dollars', r'\boxed{1}\boxed{37}\).',
                                 r'FINAL_ANSWER: 1 and \boxed{37}\).', r'\boxed{__import__("os")}\).'])
def test_no_prose_multiple_answers_or_executable_expressions(text):
    assert not diagnose_answer_v2(text, record())['numeric_correct_at_same_tolerance']


def test_truncation_and_thinking_boundaries_are_unchanged():
    text = r'Thought with 37</think>\(\boxed{37}\).'
    assert diagnose_answer_v2(text, record(), thinking=True)['numeric_correct_at_same_tolerance']
    assert not diagnose_answer_v2(text, record(), thinking=True, truncated=True)['numeric_correct_at_same_tolerance']
    assert not diagnose_answer_v2(r'Thought \boxed{37}\).', record(), thinking=True)['numeric_correct_at_same_tolerance']
