import pytest
from src.evaluation.ablation_analysis import holm_adjust, weighted


def test_holm_preserves_names_and_sorted_monotonic_adjustment():
    adjusted = holm_adjust({'third': .03, 'first': .001, 'second': .02, 'last': .5})
    assert adjusted == pytest.approx({'first': .004, 'second': .06, 'third': .06, 'last': .5})
    assert holm_adjust({}) == {}


def test_holm_rejects_invalid_values():
    for value in [-.1, 1.1, float('nan')]:
        with pytest.raises(ValueError):
            holm_adjust({'a': value})


def test_token_weighted_mean_does_not_average_microbatches_equally():
    assert weighted([{'valid_tokens': 1, 'entropy': 1.}, {'valid_tokens': 9, 'entropy': 3.}], 'entropy') == pytest.approx(2.8)
    with pytest.raises(ValueError):
        weighted([{'valid_tokens': 0, 'entropy': 3.}], 'entropy')
