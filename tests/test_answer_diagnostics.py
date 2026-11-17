import pytest

from src.evaluation.answer_diagnostics import diagnose_answer, final_response
from src.rewards.numeric import verify_completion


def record(answer='0.5'):
    return {'answer': answer, 'absolute_tolerance': 1e-8,
            'relative_tolerance': 1e-8, 'allow_percentage': True}


@pytest.mark.parametrize('answer', ['1/2', '0.5', '50%', '5e-1', r'\frac{1}{2}'])
def test_terminal_boxed_equivalence_is_diagnostic_only(answer):
    completion = 'Compute the probability.\\boxed{' + answer + '}'
    result = diagnose_answer(completion, record())
    assert result['numeric_correct_at_same_tolerance']
    assert not result['strict']['strict_correct']
    assert not verify_completion(completion, record()).strict_correct


@pytest.mark.parametrize('completion', [
    '0.5 is an intermediate result. FINAL_ANSWER: 2',
    r'\boxed{0.5} followed by another statement',
    r'\boxed{0.5} \boxed{1}',
    'FINAL_ANSWER: 0.5\nFINAL_ANSWER: 1',
    r'\boxed{__import__("os")}',
    r'\boxed{1/0}',
    'FINAL_ANSWER: 0.5 and maybe 1',
])
def test_diagnostic_rejects_answer_search_and_reward_holes(completion):
    assert not diagnose_answer(completion, record())['numeric_correct_at_same_tolerance']


def test_inline_marker_recovers_only_terminal_scalar():
    result = diagnose_answer('p=1/2. FINAL_ANSWER: 0.5', record())
    assert result['numeric_correct_at_same_tolerance']
    assert not result['strict']['strict_correct']


def test_thought_answers_are_never_used():
    assert final_response('FINAL_ANSWER: 0.5', thinking=True)[0] == ''
    result = diagnose_answer('FINAL_ANSWER: 0.5</think>\nFINAL_ANSWER: 2', record(), thinking=True)
    assert not result['numeric_correct_at_same_tolerance']
    assert result['strict']['format_valid']
    result = diagnose_answer('wrong thought</think>\nFINAL_ANSWER: 0.5', record(), thinking=True)
    assert result['strict']['strict_correct']


def test_truncated_correct_number_is_still_incorrect():
    result = diagnose_answer('FINAL_ANSWER: 0.5', record(), truncated=True)
    assert result['candidate_correct_ignoring_truncation']
    assert not result['numeric_correct_at_same_tolerance']
    assert not result['strict']['strict_correct']


def test_tolerance_and_percentage_rules_do_not_change():
    assert not diagnose_answer('FINAL_ANSWER: 0.5001', record())['numeric_correct_at_same_tolerance']
    value = record()
    value['allow_percentage'] = False
    assert not diagnose_answer(r'\boxed{50%}', value)['numeric_correct_at_same_tolerance']
