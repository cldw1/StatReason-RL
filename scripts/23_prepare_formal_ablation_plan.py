"""Prepare reviewable configs and RL-only fingerprints; launch no GPU training."""
from __future__ import annotations

import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.models.snapshot import resolve_snapshot
from src.training.dapo_mechanisms import validate_ablation
from src.training.rl_data import prepare_rl_data
from src.utils.artifacts import write_json, utc_now, file_sha256
from src.utils.config import load_config


def main():
    from transformers import AutoTokenizer
    root = PROJECT_ROOT
    report = root / 'reports/mechanism_smoke_round1'
    evidence = json.loads((report / 'results.json').read_text(encoding='utf-8'))
    reference = load_config(root / 'configs/ablations/smoke_common.yaml')
    tokenizer = AutoTokenizer.from_pretrained(resolve_snapshot(reference['model']).path, local_files_only=True, padding_side='left')
    settings = {**reference['rl_data'], 'max_train_examples': 400,
                'source_quotas': {'openai/gsm8k': 300, 'statreason_synthetic': 100},
                'selection_note': 'Frozen RL-only source-stratified 400-row pool. No model-success filtering, no development/test selection.'}
    _, _, selection = prepare_rl_data(tokenizer, root / reference['paths']['dataset_release'], settings,
                                      reference['trainer']['max_prompt_length'], reference['reproducibility']['seed'])
    write_json(report / 'formal_rl_selection.json', selection)
    common = {'extends': 'smoke_common.yaml', 'experiment': {'name': 'ablation_100', 'level': 1},
              'data': {'fingerprint': selection['record_fingerprint']}, 'rl_data': settings,
              'resources': {'max_train_seconds': 1800},
              'trainer': {'max_steps': 100, 'save_steps': 10, 'save_total_limit': 11, 'shuffle_dataset': True}}
    directory = root / 'configs/ablations'
    write_json(directory / 'formal_common.yaml', common)
    paths = {}
    projected = {}
    for key, row in evidence['rows'].items():
        config = {'extends': 'formal_common.yaml', 'experiment': {'name': key.lower() + '_ablation100', 'ablation_id': key}}
        if key in ['E1', 'E4']:
            config.setdefault('trainer', {})['epsilon_high'] = .28
        if key in ['E3', 'E4']:
            config.setdefault('trainer', {})['loss_type'] = 'dapo'
        if key in ['E2', 'E4']:
            config.setdefault('mechanisms', {}).setdefault('dynamic_sampling', {})['enabled'] = True
        if key == 'E4':
            config.setdefault('mechanisms', {})['overlong'] = {'enabled': True}
        path = directory / f'{key}_100.yaml'
        write_json(path, config)
        resolved = load_config(path)
        validate_ablation(resolved)
        paths[key] = {'config': path.relative_to(root).as_posix(), 'resolved': resolved,
                      'source_config_sha256': file_sha256(path),
                      'command': f'.venv\\Scripts\\python.exe scripts/07_train_grpo.py --config configs/ablations/{key}_100.yaml'}
        projected[key] = {'naive_100_step_seconds': row['training_seconds'] * 25,
                          'naive_generated_tokens': row['generated_tokens'] * 25,
                          'measured_smoke_peak_allocated_bytes': row['peak_allocated_bytes'],
                          'measured_smoke_peak_reserved_bytes': row['peak_reserved_bytes'],
                          'scope': 'linear extrapolation from 4 steps including warmup; not actual formal results'}
    total = sum(p['naive_100_step_seconds'] for p in projected.values())
    budget = {
        'created_at_utc': utc_now(), 'status': 'prepared_awaiting_user_large_local_gpu_budget_decision',
        'formal_training_started': False, 'paid_services_started': False,
        'hardware': 'local RTX 5060 Laptop 8 GB', 'model': 'Qwen3-0.6B retained SFT',
        'planned_runs': paths, 'planned_optimizer_steps': 500, 'random_seeds': [42],
        'rl_pool_count': 400, 'rl_pool_by_source': selection['selected_by_source'],
        'rl_pool_fingerprint': selection['record_fingerprint'], 'projections': projected,
        'projected_training_minutes': total / 60,
        'projection_limitation': 'Only four steps per arm; warmup, different RL questions, rejection variance and thermal throttling can change cost.',
        'train_time_guard_seconds_per_arm': 1800,
        'time_guard_scope': 'soft boundary check every two optimizer steps; one ongoing generation can overshoot; dynamic token and retry caps remain active',
        'max_resample_rounds_per_batch': 8, 'max_generated_tokens_per_dynamic_batch': 4096,
        'unfilled_batch_behavior': 'stop before optimizer call, save actual progress; never silently count a completed step',
        'evaluation_proposal': {
            'development': 'same existing 100-row dev panel, final checkpoint diagnostics only; no new tuning loop',
            'final_statistical_questions': 200,
            'final_gsm8k_questions': 200,
            'gsm8k_scope': 'held-out reserved rows of official train split; not official GSM8K test score',
            'final_test_bodies_opened_during_preparation': False,
            'primary_models': ['Base', 'SFT', 'E0', 'E1', 'E2', 'E3', 'E4'],
            'extra_cost_matched_checkpoints': ['E2 at nearest saved checkpoint below E0 final generation-token cost',
                                                'E4 at nearest saved checkpoint below E0 final generation-token cost'],
            'cost_match_selection': 'select only by logged generation cost; never by test accuracy; report residual budget difference',
            'generation': {'thinking': False, 'max_new_tokens': 256, 'temperature': .7, 'top_p': 1., 'top_k': 0,
                           'min_p': 0., 'do_sample': True, 'batch_size': 4},
            'pass_at_k': 'one sample per question for pass@1; pass@8 excluded from this bounded round',
            'estimated_minutes_range': [20, 45],
            'statistics': ['Wilson intervals', 'paired question bootstrap', 'McNemar exploratory test', 'source/task stratification'],
            'seed_limit': 'one training seed; no claim of across-seed robustness'},
        'projected_total_minutes_including_eval': [total/60 + 20, total/60 + 45],
        'checkpoint_disk_estimate_gib': [1., 2.],
        'stopping_rule': 'One bounded five-arm round; no repeated hyperparameter search for attractive results.',
        'grade_before_formal_runs': 'C',
    }
    write_json(report / 'formal_budget_proposal.json', budget)
    print(json.dumps({'formal_training_started': False, 'steps_planned': 500,
                      'rl_pool': selection['selected_by_source'], 'fingerprint': selection['record_fingerprint'],
                      'training_minutes_linear_projection': total / 60,
                      'total_minutes_projection': budget['projected_total_minutes_including_eval'],
                      'configs_prepared': list(paths)}, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
