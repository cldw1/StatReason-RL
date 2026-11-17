"""Protect frozen holdouts, budget matching and structural metadata without GPU."""
import json
from pathlib import Path

import pytest

from src.data.statistics import FAMILY_DESCRIPTIONS
from src.evaluation.ablation_protocol import planned_holdout_ids, structural_difficulty
from src.utils.artifacts import file_sha256


def metadata_fixture(tmp_path):
    for name, split, ids in [('statistics_full', 'test', ['stat-1', 'stat-2']),
                             ('gsm8k', 'reserved', ['gsm-1', 'gsm-2', 'gsm-3'])]:
        directory = tmp_path / name
        directory.mkdir()
        index = directory / 'split_ids.json'
        index.write_text(json.dumps({split: ids, 'sft': ['train-' + name], 'rl': [], 'validation': []}), encoding='utf-8')
        (directory / (split + '.jsonl')).write_text('sealed body must not be parsed during preparation', encoding='utf-8')
        (directory / 'manifest.json').write_text(json.dumps({'file_sha256': {'split_ids.json': file_sha256(index),
                                                                           split + '.jsonl': 'hash_checked_only_when_executing'}}), encoding='utf-8')


def test_prepare_ids_does_not_read_final_question_bodies(tmp_path, monkeypatch):
    metadata_fixture(tmp_path)
    original = Path.read_text

    def checked(self, *args, **kwargs):
        if self.name in ['test.jsonl', 'reserved.jsonl']:
            pytest.fail('preparation accessed a final question body')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_text', checked)
    ids, sources = planned_holdout_ids(tmp_path, count_per_source=2)
    again, _ = planned_holdout_ids(tmp_path, count_per_source=2)
    assert ids == again and len(set(ids)) == 4
    assert set(sources['statistics_full']['selected_ids']) == {'stat-1', 'stat-2'}
    assert sources['gsm8k']['split'] == 'reserved'


def test_all_sealed_statistics_questions_must_be_retained(tmp_path):
    metadata_fixture(tmp_path)
    with pytest.raises(ValueError, match='entire sealed'):
        planned_holdout_ids(tmp_path, count_per_source=1)


def test_changed_id_index_is_rejected(tmp_path):
    metadata_fixture(tmp_path)
    (tmp_path / 'statistics_full/split_ids.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='index changed'):
        planned_holdout_ids(tmp_path, count_per_source=2)


def test_structure_labels_cover_known_families_but_do_not_guess_gsm_difficulty():
    for task, families in FAMILY_DESCRIPTIONS.items():
        for variant, family in enumerate(families):
            record = {'source': 'statreason_synthetic', 'task_type': task, 'template_family': task + '/' + family,
                      'parameters': {'variant': variant}}
            assert structural_difficulty(record) in ['basic', 'medium', 'complex']
    assert structural_difficulty({'source': 'openai/gsm8k'}) == 'unspecified'
    with pytest.raises(ValueError):
        structural_difficulty({'source': 'statreason_synthetic', 'task_type': 'bayes',
                               'template_family': 'bayes/unknown', 'parameters': {'variant': 0}})
