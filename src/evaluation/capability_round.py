"""第二阶段第一轮：冻结开发题组与跨模型、生成预算对照。

本模块不读取任何 test.jsonl。新题组只作开发选择，不宣称最终泛化能力。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from src.data.diagnostic import parameter_digest, rewrite_question, worked_solution
from src.data.quality import audit_statistics
from src.data.schema import canonical_json, content_fingerprint, question_key
from src.data.statistics import TASK_TYPES, _make_task
from src.data.storage import read_jsonl
from src.evaluation.panel import load_validation_release
from src.models.prompts import SYSTEM_PROMPT
from src.utils.artifacts import file_sha256, write_json


MODEL_NAMES = {'qwen3_06b': 'Qwen/Qwen3-0.6B', 'qwen3_17b': 'Qwen/Qwen3-1.7B',
               'qwen25_math_15b': 'Qwen/Qwen2.5-Math-1.5B-Instruct'}
OFFICIAL_MATH_PROMPT = r'Please reason step by step, and put your final answer within \boxed{}.'


def experiment_arms() -> list[dict]:
    arms = []
    for model in ('qwen3_06b', 'qwen3_17b'):
        for thinking, budgets in ((False, (256, 512)), (True, (1024, 2048))):
            for budget in budgets:
                arms.append({'name': f'{model}_{"think" if thinking else "nonthink"}_{budget}',
                             'model': model, 'thinking': thinking, 'max_new_tokens': budget,
                             'prompt': 'common_strict', 'do_sample': True,
                             'temperature': .6 if thinking else .7,
                             'top_p': .95 if thinking else .8, 'top_k': 20, 'min_p': 0.})
    # Common 512-token arm matches Qwen3 non-thinking prompt, sampler and budget.
    arms.append({'name': 'qwen25_math_15b_common_512', 'model': 'qwen25_math_15b',
                 'thinking': False, 'max_new_tokens': 512, 'prompt': 'common_strict',
                 'do_sample': True, 'temperature': .7, 'top_p': .8, 'top_k': 20, 'min_p': 0.})
    # Native CoT prompt is a separate recipe experiment, not a pure model comparison.
    arms.append({'name': 'qwen25_math_15b_native_512', 'model': 'qwen25_math_15b',
                 'thinking': False, 'max_new_tokens': 512, 'prompt': 'official_boxed',
                 'do_sample': True, 'temperature': .7, 'top_p': .8, 'top_k': 20, 'min_p': 0.})
    return arms


def messages_for(record: dict, arm: dict) -> list[dict]:
    prompt = SYSTEM_PROMPT if arm['prompt'] == 'common_strict' else OFFICIAL_MATH_PROMPT
    return [{'role': 'system', 'content': prompt}, {'role': 'user', 'content': record['question']}]


def _read_checked_split(root: Path, split: str) -> list[dict]:
    if split not in ('sft', 'rl', 'validation'):
        raise ValueError('能力开发入口禁止读取测试集')
    path = root / f'{split}.jsonl'
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if file_sha256(path) != manifest['file_sha256'][path.name]:
        raise ValueError('数据文件哈希不匹配')
    rows = read_jsonl(path)
    if any(row['split'] != split for row in rows):
        raise ValueError('数据 split 标签不匹配')
    return rows


def prepare_panel(project: Path, destination: Path, seed: int = 20261117) -> tuple[list[dict], dict]:
    """40题：新核心14、此前未生成过回答的原迁移14、未用GSM8K验证12。"""
    panel_path, manifest_path = destination / 'panel.jsonl', destination / 'panel_manifest.json'
    if panel_path.exists() or manifest_path.exists():
        if not panel_path.exists() or not manifest_path.exists():
            raise ValueError('不完整的冻结题组，禁止自动覆盖')
        rows = read_jsonl(panel_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if content_fingerprint(rows) != manifest['panel_fingerprint'] or file_sha256(panel_path) != manifest['file_sha256']:
            raise ValueError('冻结题组指纹改变')
        if len(rows) != 40 or any(row['split'] != 'validation' for row in rows):
            raise ValueError('冻结题组规模或 split 不符')
        return rows, manifest
    seen_ids, seen_questions, seen_parameters = set(), set(), set()
    scanned = {}
    # Only files containing generated model predictions are exposure evidence.
    for path in sorted((project / 'outputs/evaluations').glob('*/evaluation_predictions.jsonl')):
        rows = read_jsonl(path)
        scanned[str(path.relative_to(project))] = file_sha256(path)
        for row in rows:
            seen_ids.add(row['id'])
            seen_questions.add(question_key(row['question']))
    source_hashes = {}
    for release in ('statreason-v1', 'statreason-diagnostic-v2', 'statreason-diagnostic-v2_1'):
        root = project / 'data/processed' / release / 'statistics_full'
        for split in ('sft', 'rl', 'validation'):
            path = root / f'{split}.jsonl'
            if path.exists():
                for row in _read_checked_split(root, split):
                    seen_parameters.add(parameter_digest(row['task_type'], row['parameters']))
                source_hashes[str(path.relative_to(project))] = file_sha256(path)
    # Existing v1 IDs contain the parameter digest; use only stored ID metadata,
    # never the held-out question body, to exclude exact test parameter reuse.
    split_ids_path = project / 'data/processed/statreason-v1/statistics_full/split_ids.json'
    split_ids = json.loads(split_ids_path.read_text(encoding='utf-8'))
    seen_parameters.update(identifier.rsplit('-', 1)[-1] for ids in split_ids.values() for identifier in ids)
    old_statistics, _ = load_validation_release(project / 'data/processed/statreason-v1/statistics_full', 'statreason_synthetic')
    gsm, _ = load_validation_release(project / 'data/processed/statreason-v1/gsm8k', 'openai/gsm8k')
    rng = random.Random(seed)
    chosen = []
    for task in TASK_TYPES:
        pool = sorted((r for r in old_statistics if r['task_type'] == task and r['id'] not in seen_ids
                       and question_key(r['question']) not in seen_questions), key=lambda r: r['id'])
        if len(pool) < 2:
            raise ValueError(f'未使用的迁移开发题不足: {task}')
        chosen.extend({**r, 'evaluation_stratum': 'unseen_transfer_development'} for r in rng.sample(pool, 2))
        for index in range(2):
            for _ in range(10000):
                parameters, question, _, answer, unit = _make_task(task, 0, rng)
                digest = parameter_digest(task, parameters)
                if digest not in seen_parameters:
                    break
            else:
                raise RuntimeError('新开发参数不足')
            seen_parameters.add(digest)
            steps, exact = worked_solution(task, parameters)
            terminal = exact if exact is not None else f'{answer:.8f}'
            question = rewrite_question(task, parameters, question, 'validation')
            question += ' 最终答案至少保留8位小数，精确分数也可。'
            if unit == 'probability':
                question += '概率可写为0到1的小数，或带%符号的百分数。'
            chosen.append({'id': f'stat-capability-dev-{digest}', 'source': 'statreason_synthetic',
                           'task_type': task, 'template_family': f'{task}/capability_dev_input_representation_0',
                           'parameters': parameters, 'question': question,
                           'solution': steps + f'\nFINAL_ANSWER: {terminal}', 'answer': format(answer, '.15g'),
                           'answer_type': 'scalar', 'answer_unit': unit, 'allow_percentage': unit == 'probability',
                           'absolute_tolerance': 1e-8 if unit == 'probability' else 1e-6,
                           'relative_tolerance': 1e-8, 'split': 'validation',
                           'evaluation_stratum': 'fresh_core_development'})
    gsm_pool = sorted((r for r in gsm if r['id'] not in seen_ids and question_key(r['question']) not in seen_questions), key=lambda r: r['id'])
    chosen.extend({**r, 'evaluation_stratum': 'unseen_gsm8k_development'} for r in rng.sample(gsm_pool, 12))
    rng.shuffle(chosen)
    if any(r['id'] in seen_ids or question_key(r['question']) in seen_questions for r in chosen):
        raise ValueError('新开发题与已生成模型回答重叠')
    audit = audit_statistics([r for r in chosen if r['source'] == 'statreason_synthetic'])
    destination.mkdir(parents=True, exist_ok=True)
    panel_path.write_text(''.join(canonical_json(r) + '\n' for r in chosen), encoding='utf-8')
    manifest = {'created_before_generation': True, 'seed': seed, 'count': len(chosen),
                'panel_fingerprint': content_fingerprint(chosen), 'file_sha256': file_sha256(panel_path),
                'by_source': dict(Counter(r['source'] for r in chosen)),
                'by_stratum': dict(Counter(r['evaluation_stratum'] for r in chosen)),
                'prior_prediction_file_hashes': scanned, 'source_file_hashes': source_hashes,
                'independent_answer_audit': audit, 'previous_prediction_id_overlap': 0,
                'test_body_read': False, 'test_id_metadata_read_for_exclusion': True,
                'split': 'validation', 'purpose': 'development_model_and_recipe_selection',
                'independence_limit': 'unseen exact questions; shared generator families, not an independent final test'}
    write_json(manifest_path, manifest)
    return chosen, manifest


def summarize_arm(rows: list[dict]) -> dict:
    if not rows:
        return {'count': 0}
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('同一实验每题只能有一个回答')
    def metrics(group):
        n = len(group)
        return {'count': n, 'strict_correct': sum(r['verification']['strict_correct'] for r in group),
                'strict_accuracy': sum(r['verification']['strict_correct'] for r in group) / n,
                'numeric_correct': sum(r['answer_diagnostics']['numeric_correct_at_same_tolerance'] for r in group),
                'numeric_accuracy': sum(r['answer_diagnostics']['numeric_correct_at_same_tolerance'] for r in group) / n,
                'format_valid': sum(r['verification']['format_valid'] for r in group),
                'format_valid_rate': sum(r['verification']['format_valid'] for r in group) / n,
                'truncated': sum(r['truncated'] for r in group),
                'truncation_rate': sum(r['truncated'] for r in group) / n,
                'total_generated_tokens': sum(r['generated_token_count'] for r in group),
                'mean_generated_tokens': sum(r['generated_token_count'] for r in group) / n,
                'generation_seconds': sum(r['generation_seconds'] for r in group),
                'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in group),
                'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in group)}
    result = metrics(rows)
    for field in ('source', 'task_type', 'evaluation_stratum'):
        groups = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        result['by_' + field] = {key: metrics(group) for key, group in groups.items()}
    result['metric_scope'] = 'one_sample_per_question_development_only'
    return result


def paired_recipe_comparison(first: list[dict], second: list[dict], seed=42, resamples=10000) -> dict:
    """题目配对允许不同模板/预算；只解释整个配方差异。"""
    import numpy as np
    a, b = {r['id']: r for r in first}, {r['id']: r for r in second}
    if not a or set(a) != set(b) or len(a) != len(first) or len(b) != len(second):
        raise ValueError('配方比较需要相同非空完整题组')
    for key in a:
        for field in ('question', 'answer', 'absolute_tolerance', 'relative_tolerance'):
            if a[key][field] != b[key][field]:
                raise ValueError('题目或判题标准不一致')
    delta = np.array([int(b[k]['verification']['strict_correct']) - int(a[k]['verification']['strict_correct']) for k in sorted(a)])
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), (resamples, len(delta)))].mean(axis=1)
    return {'n': len(delta), 'second_minus_first': float(delta.mean()),
            'paired_bootstrap_95': np.quantile(draws, [.025, .975]).tolist(),
            'resamples': resamples, 'seed': seed,
            'interpretation': 'whole_generation_recipe_difference_not_pure_model_effect',
            'limitation': 'question_bootstrap_not_template_or_training_seed_uncertainty'}
