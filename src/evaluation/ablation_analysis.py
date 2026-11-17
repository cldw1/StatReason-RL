"""Question-paired analysis of the frozen ablation study; no inference or training."""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib

from src.data.schema import canonical_json, content_fingerprint
from src.data.quality import audit_statistics
from src.data.storage import read_jsonl
from src.rewards.numeric import verify_completion
from src.utils.artifacts import file_sha256
from .ablation_protocol import read
from .comparison import paired_comparison
from .metrics import summarize_predictions


def holm_adjust(pvalues):
    if any(not 0 <= p <= 1 for p in pvalues.values()):
        raise ValueError('p-values must be finite and within [0,1]')
    ordered = sorted(pvalues, key=pvalues.get)
    result, previous = {}, 0.
    for rank, key in enumerate(ordered):
        previous = min(1., max(previous, (len(ordered) - rank) * pvalues[key]))
        result[key] = previous
    return result


def weighted(rows, key):
    total = sum(r['valid_tokens'] for r in rows)
    if not total:
        raise ValueError('no participating completion tokens')
    return sum(r[key] * r['valid_tokens'] for r in rows) / total


def training_series(root, row):
    run = root / row['run']
    policy = read_jsonl(run / 'policy_audit.jsonl')
    rollout = read_jsonl(run / 'rollout_groups.jsonl')
    attempts = read_jsonl(run / 'sampling_audit.jsonl')
    logs = {r['step']: r for r in read_jsonl(run / 'train_metrics.jsonl') if 'grad_norm' in r}
    grouped = defaultdict(list)
    for entry in policy:
        if not entry['old_logps_immutable'] or entry['independent_loss_absolute_error'] > 2e-5:
            raise ValueError('policy probability or independent loss audit failed')
        grouped[entry['optimizer_step'] + 1].append(entry)
    if set(grouped) != set(range(1, row['optimizer_steps'] + 1)):
        raise ValueError('optimizer step series is incomplete')
    by_id = {r['rollout_id']: r for r in rollout}
    series = []
    for step, entries in sorted(grouped.items()):
        ids = {i for p in entries for i in p['rollout_id']}
        completions = [c for i in ids for c in by_id[i]['completions']]
        point = {'step': step, 'grad_norm': logs[step]['grad_norm'], 'loss': logs[step]['loss'],
                 'raw_reward': sum(c['accuracy_reward'] for c in completions) / len(completions),
                 'shaped_reward': sum(c['reward'] for c in completions) / len(completions),
                 'mean_length': sum(len(c['completion_ids']) for c in completions) / len(completions),
                 'truncation_rate': sum(c['verification']['truncated'] for c in completions) / len(completions),
                 'ratio_min': min(p['ratio_min'] for p in entries), 'ratio_max': max(p['ratio_max'] for p in entries),
                 'cumulative_generation_tokens_before_update': sum(r['generated_tokens'] for r in rollout if r['optimizer_step'] < step)}
        kinds = {p['loss_type'] for p in entries}
        if len(kinds) != 1:
            raise ValueError('mixed loss aggregation within one optimizer update')
        # Batch size is one in this controlled study. DAPO microbatch policy
        # terms already share the global token denominator and must be summed.
        policy_sum = sum(p['unscaled_policy_loss'] for p in entries)
        point['policy_loss'] = policy_sum / len(entries) if kinds == {'grpo'} else policy_sum
        for field in ['policy_entropy_raw', 'policy_entropy_temperature', 'policy_entropy_sampling', 'kl',
                      'effective_low_clip', 'effective_high_clip', 'ratio_low_outside', 'ratio_high_outside']:
            point[field] = weighted(entries, field)
        series.append(point)
    return {'updates': series, 'sampling': attempts,
            'source_sha256': {p: file_sha256(run / p) for p in ['policy_audit.jsonl', 'rollout_groups.jsonl', 'sampling_audit.jsonl', 'train_metrics.jsonl']}}


def audit_and_analyze(root, resamples=10000, audit_only=False):
    folder = root / 'outputs/ablation_evaluation_round1'
    plan = read(folder / 'frozen_plan.json')
    if read(folder / 'study_status.json')['status'] != 'completed':
        raise ValueError('frozen evaluation is incomplete; resume script 26 first')
    for path, expected in plan['source_sha256'].items():
        if file_sha256(root / path) != expected:
            raise ValueError(f'frozen generation or verifier source changed: {path}')
    panel = read_jsonl(folder / 'panel.jsonl')
    if [r['id'] for r in panel] != plan['ordered_question_ids'] or len(panel) != 400:
        raise ValueError('question panel differs from frozen selection')
    panel_meta = read(folder / 'panel_identity.json')
    if content_fingerprint(panel) != panel_meta['fingerprint']:
        raise ValueError('question text, answer, tolerance or metadata changed')
    gold_audit = audit_statistics([r for r in panel if r['source'] == 'statreason_synthetic'])
    questions = {r['id']: r for r in panel}
    predictions, metrics, evidence = {}, {}, {}
    for name, policy in plan['models'].items():
        directory = folder / name
        if read(directory / 'run_status.json')['status'] != 'completed':
            raise ValueError(f'{name}: missing completed state')
        signature = hashlib.sha256(canonical_json({'plan': plan['signature'], 'policy': policy, 'name': name}).encode()).hexdigest()
        identity = read(directory / 'evaluation_identity.json')
        if identity['signature'] != signature or identity['plan_signature'] != plan['signature']:
            raise ValueError('checkpoint evaluation identity mismatch')
        if policy['adapter']:
            adapter = root / policy['adapter']
            if file_sha256(adapter / 'adapter_model.safetensors') != policy['adapter_safetensors_sha256'] or file_sha256(adapter / 'adapter_config.json') != policy['adapter_config_sha256']:
                raise ValueError('evaluated policy weights changed')
        rows = read_jsonl(directory / 'evaluation_predictions.jsonl')
        if len(rows) != len(panel) or {r['id'] for r in rows} != set(questions):
            raise ValueError('missing, duplicate or foreign predicted question')
        for r in rows:
            reference = questions[r['id']]
            for field in ['question', 'answer', 'absolute_tolerance', 'relative_tolerance', 'source', 'task_type', 'template_family', 'difficulty']:
                if r[field] != reference[field]:
                    raise ValueError(f'prediction row changed ground truth: {field}')
            if r['evaluation_signature'] != signature or r['plan_signature'] != plan['signature']:
                raise ValueError('mixed prediction signatures')
            if r['mode'] != 'non-thinking' or r['enable_thinking'] or r['generated_token_count'] != len(r['completion_token_ids']):
                raise ValueError('generation mode or token accounting differs from the protocol')
            for field in ['max_new_tokens', 'do_sample', 'temperature', 'top_p', 'top_k', 'min_p']:
                if r['generation_config'][field] != plan['generation'][field]:
                    raise ValueError('saved generation configuration differs from the frozen protocol')
            actual = verify_completion(r['completion'], reference, truncated=r['truncated']).to_dict()
            if actual != r['verification']:
                raise ValueError('fresh strict reward verification disagrees with saved score')
        base = summarize_predictions(rows)['by_mode']['non-thinking']
        base['by_structural_difficulty'] = {
            difficulty: summarize_predictions([r for r in rows if r['difficulty'] == difficulty])['by_mode']['non-thinking']
            for difficulty in sorted({r['difficulty'] for r in rows})}
        saved = read(directory / 'evaluation_summary.json')['by_mode']['non-thinking']
        if base != saved:
            raise ValueError('saved summary differs from question-level recomputation')
        predictions[name], metrics[name] = rows, base
        resource = read(directory / 'resource_metrics.json')
        attempts = read_jsonl(directory / 'generation_attempts.jsonl')
        actual_tokens = sum(a['generated_tokens'] for a in attempts)
        if actual_tokens != resource['total_returned_attempt_tokens_including_recomputed_batches']:
            raise ValueError('returned generation cost mismatch')
        evidence[name] = {'prediction_sha256': file_sha256(directory / 'evaluation_predictions.jsonl'),
                          'resource_sha256': file_sha256(directory / 'resource_metrics.json'),
                          'attempt_generated_tokens': actual_tokens, 'attempt_seconds': sum(a['seconds'] for a in attempts),
                          'peak_allocated_bytes': resource['peak_allocated_bytes'], 'peak_reserved_bytes': resource['peak_reserved_bytes']}
    training = read(root / 'reports/ablation_round1/training_results.json')
    if not training['controls_verified']:
        raise ValueError('controlled training audit is missing')
    common_config = None
    for key, row in training['rows'].items():
        run = root / row['run']
        status = read(run / 'run_status.json')
        identity = read(run / 'training_identity.json')
        config = deepcopy(identity['config'])
        expected_mechanisms = {'E0': ('grpo', .2, False, False), 'E1': ('grpo', .28, False, False),
                               'E2': ('grpo', .2, True, False), 'E3': ('dapo', .2, False, False),
                               'E4': ('dapo', .28, True, True)}
        actual = (config['trainer']['loss_type'], config['trainer']['epsilon_high'],
                  config['mechanisms']['dynamic_sampling']['enabled'], config['mechanisms']['overlong']['enabled'])
        if actual != expected_mechanisms[key] or status['optimizer_steps'] != row['optimizer_steps']:
            raise ValueError('actual mechanism configuration or step count differs from report')
        config['experiment'].pop('name'); config['experiment'].pop('ablation_id')
        config['trainer'].pop('loss_type'); config['trainer'].pop('epsilon_high')
        for mechanism in ['dynamic_sampling', 'overlong']:
            config['mechanisms'][mechanism].pop('enabled')
        if common_config is not None and config != common_config:
            raise ValueError('uncontrolled training parameter change')
        common_config = config
        if identity['selection'] != training['selection']:
            raise ValueError('RL data selection changed')
        for field, expected in training['common_code_identity'].items():
            if identity[field] != expected:
                raise ValueError('training code or package versions differed across arms')
        update = read(run / 'parameter_updates.json')
        reload = read(run / 'reload_verification.json')
        reference = read(run / 'reference_verification.json')
        if update['changed_tensor_count'] != 224 or not update['all_non_lora_parameters_frozen']:
            raise ValueError('real LoRA update/frozen base evidence missing')
        if not reload['adapter_weights_exact_match'] or not reload['probe_logits_exact_match'] or not reference['reference_hashes_unchanged']:
            raise ValueError('reload or independent reference audit failed')
        if file_sha256(run / 'sampling_audit.jsonl') != row['sampling_audit_sha256']:
            raise ValueError('training sampling log changed')
    series = {key: training_series(root, row) for key, row in training['rows'].items()}
    result = {'scope': 'frozen_400_question_holdout_one_training_seed_one_answer_per_question',
              'plan_signature': plan['signature'], 'panel_identity': panel_meta, 'gold_audit': gold_audit,
              'prediction_count': sum(len(v) for v in predictions.values()), 'unique_question_count': len(panel),
              'models': plan['models'], 'generation': plan['generation'], 'metrics': metrics, 'training': training,
              'training_series': series, 'evaluation_evidence': evidence,
              'total_evaluation_returned_tokens': sum(v['attempt_generated_tokens'] for v in evidence.values()),
              'total_evaluation_returned_seconds': sum(v['attempt_seconds'] for v in evidence.values()),
              'independence_limit': plan['independence_scope'], 'gsm8k_scope': plan['gsm8k_scope'],
              'gpu_work_performed_by_analysis': False, 'audit_passed': True}
    if audit_only:
        return result
    pairs = [('Base', 'SFT')]
    pairs += [('SFT', k + '_step100') for k in ['E0', 'E1', 'E3', 'E4']]
    pairs += [('E0_step100', k + '_step100') for k in ['E1', 'E3', 'E4']]
    pairs += [('E0_step30', k + '_step30') for k in ['E1', 'E2', 'E3', 'E4']]
    pairs += [('E0_step100', info['model']) for info in plan['cost_comparisons'].values()]
    comparisons = {}
    for first, second in pairs:
        comparison = paired_comparison(predictions[first], predictions[second], resamples=resamples)
        comparison.update(first_model=first, second_model=second, scope=result['scope'],
                          multiplicity='exploratory; unadjusted intervals plus Holm-adjusted McNemar across this comparison family')
        if (first, second) in [('SFT', 'E4_step100'), ('E0_step100', 'E4_step100')]:
            comparison['by_source'] = {}
            for source in ['openai/gsm8k', 'statreason_synthetic']:
                subset = paired_comparison([r for r in predictions[first] if r['source'] == source],
                                           [r for r in predictions[second] if r['source'] == source], resamples=resamples)
                subset['scope'] = result['scope']
                comparison['by_source'][source] = subset
        comparisons[second + '_vs_' + first] = comparison
    adjusted = holm_adjust({k: v['mcnemar_exact_two_sided_p'] for k, v in comparisons.items()})
    for key, value in comparisons.items():
        value['mcnemar_holm_adjusted_p'] = adjusted[key]
    result['comparisons'] = comparisons
    result['comparison_family_size'] = len(comparisons)
    return result
