from copy import deepcopy
import json

import pytest

from src.data.schema import canonical_json, content_fingerprint
from src.evaluation.capability_round import experiment_arms, messages_for, prepare_panel, summarize_arm, paired_recipe_comparison
from src.models.snapshot import resolve_snapshot
from src.utils.artifacts import file_sha256


def prediction(identifier='a', correct=True):
    return {'id': identifier, 'question': 'Question', 'answer': '1', 'absolute_tolerance': 0,
            'relative_tolerance': 0, 'verification': {'strict_correct': correct, 'format_valid': True},
            'answer_diagnostics': {'numeric_correct_at_same_tolerance': correct}, 'truncated': False,
            'generated_token_count': 10, 'generation_seconds': 1., 'peak_allocated_bytes': 100,
            'peak_reserved_bytes': 200, 'source': 'fixture', 'task_type': 'fixture', 'evaluation_stratum': 'fixture'}


def test_matched_model_arm_has_same_prompt_sampler_and_budget():
    arms = {a['name']: a for a in experiment_arms()}
    a = arms['qwen3_06b_nonthink_512']
    for name in ('qwen3_17b_nonthink_512', 'qwen25_math_15b_common_512'):
        b = arms[name]
        assert {k: v for k, v in a.items() if k not in ('name', 'model')} == {k: v for k, v in b.items() if k not in ('name', 'model')}
    assert {a['max_new_tokens'] for a in arms.values() if a['model'] == 'qwen3_06b'} == {256, 512, 1024, 2048}
    assert arms['qwen3_06b_think_1024']['top_p'] == .95


def test_prompt_builder_never_reads_reference_solution():
    value = {'question': 'visible', 'answer': 'secret answer', 'solution': 'secret solution'}
    for arm in experiment_arms():
        messages = messages_for(value, arm)
        assert messages[-1]['content'] == 'visible'
        assert 'secret' not in str(messages)


def test_summary_keeps_strict_and_numeric_separate_and_counts_failures():
    a, b = prediction(), prediction('b', False)
    b['answer_diagnostics']['numeric_correct_at_same_tolerance'] = True
    b['verification']['format_valid'] = False
    b['truncated'] = True
    result = summarize_arm([a, b])
    assert result['strict_accuracy'] == .5
    assert result['numeric_accuracy'] == 1.
    assert result['truncated'] == 1
    assert result['total_generated_tokens'] == 20
    with pytest.raises(ValueError):
        summarize_arm([a, a])


def test_recipe_bootstrap_requires_same_questions_and_tolerance():
    a, b = [prediction()], [prediction(correct=False)]
    result = paired_recipe_comparison(a, b, resamples=200)
    assert result['paired_bootstrap_95'] == [-1., -1.]
    b[0]['absolute_tolerance'] = .1
    with pytest.raises(ValueError):
        paired_recipe_comparison(a, b)


def test_frozen_panel_refuses_modified_file(tmp_path):
    rows = [{'id': str(i), 'split': 'validation'} for i in range(40)]
    path = tmp_path / 'panel.jsonl'
    path.write_text(''.join(canonical_json(r) + '\n' for r in rows), encoding='utf-8')
    manifest = {'panel_fingerprint': content_fingerprint(rows), 'file_sha256': file_sha256(path)}
    (tmp_path / 'panel_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    assert prepare_panel(tmp_path / 'no_data_needed', tmp_path)[0] == rows
    path.write_text(path.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='指纹'):
        prepare_panel(tmp_path / 'no_data_needed', tmp_path)


def test_qwen25_math_cache_is_allowed_without_network(tmp_path):
    revision = 'a' * 40
    path = tmp_path / 'hub/models--Qwen--Qwen2.5-Math-1.5B-Instruct/snapshots' / revision
    path.mkdir(parents=True)
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors'):
        (path / name).write_text('{}', encoding='utf-8')
    snapshot = resolve_snapshot({'name_or_path': 'Qwen/Qwen2.5-Math-1.5B-Instruct', 'revision': revision, 'hf_home': str(tmp_path)})
    assert snapshot.path == path
