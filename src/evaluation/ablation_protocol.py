"""Freeze models, question IDs, sampling and structural labels before test access."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from src.data.schema import canonical_json, content_fingerprint, question_key
from src.data.statistics import FAMILY_DESCRIPTIONS
from src.data.storage import read_jsonl
from src.utils.artifacts import file_sha256, write_json, utc_now
from src.utils.environment import package_versions


STRUCTURAL_DIFFICULTY = {
    'conditional_probability': ['basic', 'basic', 'basic', 'basic'],
    'bayes': ['medium', 'medium', 'medium', 'complex'],
    'binomial_probability': ['medium', 'medium', 'medium', 'complex'],
    'discrete_expectation': ['basic', 'basic', 'medium', 'complex'],
    'discrete_variance': ['medium', 'medium', 'medium', 'basic'],
    'standard_error': ['basic', 'medium', 'medium', 'complex'],
    'normal_mean_ci_endpoint': ['medium', 'complex', 'complex', 'medium'],
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def structural_difficulty(record):
    if record['source'] != 'statreason_synthetic':
        return 'unspecified'
    task = record['task_type']
    names = FAMILY_DESCRIPTIONS[task]
    family = record['template_family']
    variant = record['parameters']['variant']
    if not isinstance(variant, int) or not 0 <= variant < len(names) or family != task + '/' + names[variant]:
        raise ValueError('unknown statistics structure: no silent difficulty guess')
    return STRUCTURAL_DIFFICULTY[task][variant]


def planned_holdout_ids(dataset, seed=42, count_per_source=200):
    sources, ids = {}, []
    rng = random.Random(seed)
    for directory, split in [('statistics_full', 'test'), ('gsm8k', 'reserved')]:
        manifest = read(dataset / directory / 'manifest.json')
        index = dataset / directory / 'split_ids.json'
        if file_sha256(index) != manifest['file_sha256']['split_ids.json']:
            raise ValueError('split-ID index changed')
        split_ids = read(index)
        pool = sorted(split_ids[split])
        if directory == 'statistics_full':
            if len(pool) != count_per_source:
                raise ValueError('retain the entire sealed statistics test set')
            selected = pool
        else:
            selected = rng.sample(pool, count_per_source)
        forbidden = {i for name in ['sft', 'rl', 'validation'] for i in split_ids.get(name, [])}
        if set(selected) & forbidden:
            raise ValueError('held-out IDs overlap training or development')
        ids.extend(selected)
        filename = split + '.jsonl'
        sources[directory] = {'split': split, 'file': filename, 'file_sha256': manifest['file_sha256'][filename],
                              'selected_ids': selected, 'index_sha256': file_sha256(index)}
    if len(set(ids)) != len(ids):
        raise ValueError('duplicate held-out IDs')
    rng.shuffle(ids)
    return ids, sources


def prepare_plan(root):
    training = read(root / 'reports/ablation_round1/training_results.json')
    dataset = root / training['common_config']['paths']['dataset_release']
    ids, sources = planned_holdout_ids(dataset)
    generation = {'max_prompt_tokens': 1536, 'max_new_tokens': 256, 'max_time_per_completion_seconds': 60,
                  'do_sample': True, 'temperature': .7, 'top_p': 1., 'top_k': 0, 'min_p': 0.,
                  'num_return_sequences': 1, 'batch_size': 4, 'thinking': False}
    paths = ['scripts/26_evaluate_ablations.py', 'src/evaluation/ablation_protocol.py',
             'src/models/inference.py', 'src/models/prompts.py', 'src/models/lora.py', 'src/models/snapshot.py',
             'src/rewards/numeric.py', 'src/evaluation/runner.py', 'src/evaluation/storage.py', 'src/evaluation/metrics.py',
             'src/training/dapo_mechanisms.py', 'src/utils/environment.py', 'src/data/schema.py',
             'src/data/statistics.py', 'src/data/storage.py', 'src/utils/config.py']
    value = {'schema_version': 1, 'dataset_release': dataset.relative_to(root).as_posix(),
             'ordered_question_ids': ids, 'sources': sources, 'models': training['models'],
             'model_config': training['common_config']['model'], 'generation': generation, 'seed': 42,
             'source_sha256': {p: file_sha256(root / p) for p in paths}, 'versions': package_versions(),
             'structural_difficulty': STRUCTURAL_DIFFICULTY,
             'difficulty_scope': 'predefined by solution structure, not model accuracy; GSM8K left unspecified',
             'gsm8k_scope': 'held-out reserved rows of official train split, not official GSM8K test',
             'independence_scope': 'held-out records and v1 train/RL family isolation; related structures exist in historical expanded development',
             'cost_comparisons': training['cost_comparisons'],
             'comparison_protocol': {'common_step': 30, 'available_endpoint_step': 100,
                                     'E2_step100': 'missing_budget_stopped_do_not_impute',
                                     'cost_match': 'nearest saved step below E0 total cost; report residual budget'},
             'final_question_bodies_read_at_preparation': False}
    signature = hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()
    folder = root / 'outputs/ablation_evaluation_round1'
    path = folder / 'frozen_plan.json'
    if path.exists():
        previous = read(path)
        if previous['signature'] != signature:
            raise ValueError('frozen evaluation protocol changed; preserve previous study and review before using a new version')
        return previous
    folder.mkdir(parents=True, exist_ok=True)
    plan = {**value, 'signature': signature, 'frozen_at_utc': utc_now()}
    write_json(path, plan)
    return plan


def load_frozen_panel(root, plan):
    dataset = root / plan['dataset_release']
    by_id = {}
    for directory, info in plan['sources'].items():
        path = dataset / directory / info['file']
        if file_sha256(path) != info['file_sha256']:
            raise ValueError('held-out source file changed')
        expected_source = 'statreason_synthetic' if directory == 'statistics_full' else 'openai/gsm8k'
        for row in read_jsonl(path):
            if row['split'] != info['split'] or row['source'] != expected_source:
                raise ValueError('wrong source or split in held-out file')
            if row['id'] in info['selected_ids']:
                if row['id'] in by_id:
                    raise ValueError('duplicate selected row')
                by_id[row['id']] = row
    if set(by_id) != set(plan['ordered_question_ids']):
        raise ValueError('frozen evaluation panel is incomplete')
    # Exact-question leakage is an error, never a reason to silently drop items.
    forbidden_questions = set()
    for directory in ['statistics_full', 'gsm8k']:
        for split in ['sft', 'rl', 'validation']:
            forbidden_questions.update(question_key(r['question']) for r in read_jsonl(dataset / directory / (split + '.jsonl')))
    extra = root / 'outputs/capability_round1/panel.jsonl'
    if extra.exists():
        forbidden_questions.update(question_key(r['question']) for r in read_jsonl(extra))
    previous_dev = root / 'data/processed/statreason-diagnostic-v2/statistics_full/validation.jsonl'
    if previous_dev.exists():
        forbidden_questions.update(question_key(r['question']) for r in read_jsonl(previous_dev))
    panel = []
    for key in plan['ordered_question_ids']:
        row = dict(by_id[key])
        if question_key(row['question']) in forbidden_questions:
            raise ValueError(f'exact question overlaps past training/development: {key}; preserve source, do not delete failures')
        row.update(difficulty=structural_difficulty(row),
                   evaluation_stratum='statistics_final' if row['source'] == 'statreason_synthetic' else 'gsm8k_reserved_final')
        panel.append(row)
    return panel, {'fingerprint': content_fingerprint(panel), 'question_count': len(panel),
                   'by_source': dict(Counter(r['source'] for r in panel)),
                   'by_structural_difficulty': dict(Counter(r['difficulty'] for r in panel)),
                   'exact_question_overlap_observed': 0, 'official_gsm8k_test': False}
