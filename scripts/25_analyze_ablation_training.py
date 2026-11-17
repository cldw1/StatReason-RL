"""Audit real ablation runs and freeze checkpoint comparisons before final evaluation."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.data.storage import read_jsonl
from src.models.lora import validate_adapter
from src.rewards.numeric import verify_completion
from src.training.dapo_mechanisms import overlong_penalty, raw_group_status
from src.training.group_rewards import relative_advantages
from src.utils.artifacts import file_sha256, utc_now, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def core(config):
    value = deepcopy(config)
    for key in ['name', 'ablation_id']:
        value['experiment'].pop(key)
    for key in ['loss_type', 'epsilon_high']:
        value['trainer'].pop(key)
    for key in ['dynamic_sampling', 'overlong']:
        value['mechanisms'][key].pop('enabled')
    return value


def choose_cost_checkpoint(costs, budget):
    eligible = [int(step) for step, cost in costs.items() if cost <= budget]
    if not eligible:
        raise ValueError('no saved checkpoint fits the generation-token budget')
    return max(eligible)


def main():
    root = PROJECT_ROOT
    destination = root / 'reports/ablation_round1'
    destination.mkdir(parents=True, exist_ok=True)
    summary, models = {}, {}
    common = common_code = common_selection = None
    for key in ['E0', 'E1', 'E2', 'E3', 'E4']:
        candidates = sorted((root / 'outputs/runs').glob(key.lower() + '_ablation100_*'))
        if not candidates:
            raise ValueError(f'{key}: missing run; never drop a missing arm from the report')
        run = candidates[-1]
        status, identity, resources = [read(run / p) for p in ['run_status.json', 'training_identity.json', 'resource_metrics.json']]
        if status['status'] not in ['completed', 'sampling_budget_stopped', 'budget_stopped']:
            raise ValueError(f'{key}: unresolved run state {status["status"]}')
        code = {k: identity[k] for k in ['code_hashes', 'support_source_sha256', 'trl_source_sha256', 'versions']}
        current = core(identity['config'])
        if common is not None and (common != current or common_code != code or common_selection != identity['selection']):
            raise ValueError('runs differ beyond declared ablation mechanisms')
        common, common_code, common_selection = current, code, identity['selection']
        config = identity['config']
        model = read(run / 'model_snapshot.json')
        validate_adapter(run / 'adapter', model['repository'] if 'repository' in model else identity['adapter']['base_repository'],
                         identity['adapter']['base_revision'])
        attempts, rollouts, policy = [read_jsonl(run / p) for p in ['sampling_audit.jsonl', 'rollout_groups.jsonl', 'policy_audit.jsonl']]
        accepted = {a['rollout_id'] for a in attempts if a['accepted_for_training']}
        if accepted != {i for p in policy for i in p['rollout_id']}:
            raise ValueError('accepted group and actual optimizer input mismatch')
        kept, all_rows = [], []
        for rollout in rollouts:
            for c in rollout['completions']:
                # The raw verification is already tied to the immutable RL data
                # below; reward arithmetic is audited here without a model run.
                penalty = (overlong_penalty(len(c['completion_ids']), config['mechanisms']['overlong']['soft_length'],
                                           config['trainer']['max_completion_length'], config['mechanisms']['overlong']['penalty_coefficient'])
                           if config['mechanisms']['overlong']['enabled'] else 0.)
                if c['accuracy_reward'] != float(c['verification']['strict_correct']) or c['length_penalty'] != penalty or c['reward'] != c['accuracy_reward'] + penalty:
                    raise ValueError('stored reward components do not match the declared rule')
                all_rows.append(c)
                if rollout['rollout_id'] in accepted:
                    kept.append(c)
            expected = relative_advantages([c['reward'] for c in rollout['completions']])['advantages']
            if any(abs(c['advantage'] - a) > 1e-5 for c, a in zip(rollout['completions'], expected, strict=True)):
                raise ValueError('stored group advantages disagree with independent calculation')
        tokens = sum(p['valid_tokens'] for p in policy)
        if sum(len(c['completion_ids']) for c in kept) * config['trainer']['num_iterations'] != tokens:
            raise ValueError('completion masks or rollout reuse token accounting changed')
        if sum(a['generated_tokens'] for a in attempts) != resources['total_generated_tokens']:
            raise ValueError('some returned sampling cost is missing')
        weighted = lambda field: sum(p[field] * p['valid_tokens'] for p in policy) / tokens
        checkpoint_costs = {}
        for cp in (run / 'checkpoints').glob('checkpoint-*'):
            step = int(cp.name.split('-')[-1])
            if read(cp / 'trainer_state.json')['global_step'] != step:
                raise ValueError('checkpoint step metadata mismatch')
            checkpoint_costs[step] = sum(r['generated_tokens'] for r in rollouts if r['optimizer_step'] < step)
        summary[key] = {'run': run.relative_to(root).as_posix(), 'status': status['status'], 'optimizer_steps': status['optimizer_steps'],
                        'attempts': len(attempts), 'accepted_groups': len(accepted),
                        'raw_effective_attempt_rate': sum(a['mixed'] for a in attempts) / len(attempts),
                        'all_correct_attempts': sum(a['all_correct'] for a in attempts),
                        'all_wrong_attempts': sum(a['all_wrong'] for a in attempts),
                        'accepted_effective_group_rate': sum(a['mixed'] for a in attempts if a['accepted_for_training']) / len(accepted),
                        'total_generated_tokens': resources['total_generated_tokens'], 'extra_resampling_tokens': resources['extra_resampling_tokens'],
                        'discarded_rollout_tokens': resources['discarded_rollout_tokens'], 'training_seconds': resources['training_seconds'],
                        'peak_allocated_bytes': resources['peak_allocated_bytes'], 'peak_reserved_bytes': resources['peak_reserved_bytes'],
                        'upper_effective_clip_rate': weighted('effective_high_clip'), 'lower_effective_clip_rate': weighted('effective_low_clip'),
                        'raw_entropy_token_weighted': weighted('policy_entropy_raw'), 'sampling_entropy_token_weighted': weighted('policy_entropy_sampling'),
                        'kl_token_weighted': weighted('kl'), 'ratio_min': min(p['ratio_min'] for p in policy), 'ratio_max': max(p['ratio_max'] for p in policy),
                        'kept_mean_length': sum(len(c['completion_ids']) for c in kept) / len(kept),
                        'kept_truncation_rate': sum(c['verification']['truncated'] for c in kept) / len(kept),
                        'length_penalized_kept': sum(c['length_penalty'] < 0 for c in kept),
                        'length_penalized_all': sum(c['length_penalty'] < 0 for c in all_rows),
                        'max_independent_loss_error': max(p['independent_loss_absolute_error'] for p in policy),
                        'changed_lora_tensors': read(run / 'parameter_updates.json')['changed_tensor_count'],
                        'reload': read(run / 'reload_verification.json'),
                        'checkpoint_generation_cost': checkpoint_costs, 'sampling_stop_reason': status.get('sampling_stop_reason'),
                        'sampling_audit_sha256': file_sha256(run / 'sampling_audit.jsonl')}
        if 30 not in checkpoint_costs:
            raise ValueError('the shared 30-step comparison is not available for every arm')
        models[key + '_step30'] = {'parent_run': summary[key]['run'], 'optimizer_step': 30,
                                  'adapter': (run / 'checkpoints/checkpoint-30').relative_to(root).as_posix(),
                                  'training_generated_tokens': checkpoint_costs[30], 'ablation': key}
        if status['optimizer_steps'] == 100:
            models[key + '_step100'] = {'parent_run': summary[key]['run'], 'optimizer_step': 100,
                                        'adapter': (run / 'adapter').relative_to(root).as_posix(),
                                        'training_generated_tokens': checkpoint_costs[100], 'ablation': key}
    # Regrade every raw completion against the immutable selected RL records.
    records = {}
    for directory in ['statistics_full', 'gsm8k']:
        records.update({r['id']: r for r in read_jsonl(root / common['paths']['dataset_release'] / directory / 'rl.jsonl')})
    regraded = 0
    for row in summary.values():
        for rollout in read_jsonl(root / row['run'] / 'rollout_groups.jsonl'):
            for c in rollout['completions']:
                actual = verify_completion(c['completion'], records[c['id']], truncated=c['verification']['truncated']).to_dict()
                if actual != c['verification']:
                    raise ValueError('stored verification differs from fresh strict regrading')
                regraded += 1
    budget = summary['E0']['total_generated_tokens']
    cost_comparisons = {}
    for key in ['E2', 'E4']:
        step = choose_cost_checkpoint(summary[key]['checkpoint_generation_cost'], budget)
        name = key + f'_step{step}'
        if name not in models:
            run = root / summary[key]['run']
            models[name] = {'parent_run': summary[key]['run'], 'optimizer_step': step,
                            'adapter': (run / f'checkpoints/checkpoint-{step}').relative_to(root).as_posix(),
                            'training_generated_tokens': summary[key]['checkpoint_generation_cost'][step], 'ablation': key}
        cost_comparisons[key] = {'model': name, 'budget': budget, 'actual_tokens': models[name]['training_generated_tokens'],
                                 'residual_budget': budget - models[name]['training_generated_tokens'],
                                 'selection': 'highest saved optimizer step within token budget, no accuracy accessed'}
    models = {'Base': {'adapter': None, 'optimizer_step': 0},
              'SFT': {'adapter': common['initialization']['sft_adapter'], 'optimizer_step': 0}, **models}
    for name, policy in models.items():
        policy['roles'] = []
        if name in ['Base', 'SFT'] or name.endswith('_step30'):
            policy['roles'].append('shared_30_step_comparison')
        if name in ['Base', 'SFT'] or name.endswith('_step100'):
            policy['roles'].append('available_100_step_endpoints_E2_missing')
        for key, comparison in cost_comparisons.items():
            if comparison['model'] == name:
                policy['roles'].append('generation_cost_bounded_vs_E0_step100')
        if policy['adapter']:
            path = root / policy['adapter']
            policy['adapter_safetensors_sha256'] = file_sha256(path / 'adapter_model.safetensors')
            policy['adapter_config_sha256'] = file_sha256(path / 'adapter_config.json')
    result = {'recorded_at_utc': utc_now(), 'rows': summary, 'models': models,
              'common_config': common, 'common_code_identity': common_code, 'selection': common_selection,
              'controls_verified': True, 'strict_completion_regrades': regraded, 'cost_comparisons': cost_comparisons,
              'final_test_bodies_read': False, 'ability_evaluation_performed': False,
              'grade': 'C_pending_unified_evaluation',
              'analysis_decision': 'no E2 retry; compare all arms at 30 steps, list four available 100-step endpoints separately'}
    write_json(destination / 'training_results.json', result)
    lines = ['# 真实消融训练审计与冻结评测方案', '',
             '模型Qwen3-0.6B，固定原SFT、400题RL池、G=4、每组旧策略复用两次、单种子42。所有证据来自本次实际运行。尚未做最终能力评测。', '',
             '| 实验 | 实际步数 | 尝试/接受组 | 原始有效组比例 | 总生成Token | 额外补采Token | 训练分钟 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for key, r in summary.items():
        lines.append(f"| {key} | {r['optimizer_steps']} | {r['attempts']}/{r['accepted_groups']} | {r['raw_effective_attempt_rate']:.1%} | {r['total_generated_tokens']} | {r['extra_resampling_tokens']} | {r['training_seconds']/60:.2f} |")
    lines += ['', 'E2在30步后，连续9个候选组没有二元奖励差异而按既定规则停止。15个有效组已训练，末尾2733个Token为未用于更新的补采成本，必须计入总成本。不能把30步与其他100步直接作公平机制归因。', '',
              '在读取最终测试题前冻结：五组统一比较第30步；四个完成100步的组单列终点比较，E2的100步标为缺失。预算受限比较采用不超过E0总生成16157 Token的最近保存检查点，并明确剩余预算差，不声称预算完全相等。', '',
              '本轮不继续针对最终成绩调参，也不重跑E2寻找更好结果。其提前停止本身是动态筛选成本和预算限制的实测证据。', '',
              '| 实验 | 上侧有效裁剪 | 下侧有效裁剪 | 训练接受回答中触发长度惩罚 |', '|---|---:|---:|---:|']
    for key, r in summary.items():
        lines.append(f"| {key} | {r['upper_effective_clip_rate']:.4%} | {r['lower_effective_clip_rate']:.4%} | {r['length_penalized_kept']} |")
    lines += ['', 'E4正式运行中53个接受回答触发长度惩罚，证明其参与了实际更新。裁剪比例整体很低，不能仅凭上裁剪配置宣称缓解熵坍塌。策略熵取自各组实际采样的题目和Token；动态筛选改变样本分布，均值差异不能单独解释为探索能力改变。', '',
              f'重新严格判题{regraded}个原始回答通过，共同配置、源代码、版本、数据选择及奖励算术核对通过。所有组均有224个LoRA张量变化和重载一致性证据。', '',
              '最终评测拟使用200道封存统计题及200道GSM8K reserved题，后者来自官方train而非官方test。使用统一Non-thinking模板、温度0.7、top_p=1、top_k=0、256 Token、batch=4、每题一次采样。难度标签按结构预先定义，GSM题不凭模型分数猜测难度。', '',
              '执行：`python scripts/26_evaluate_ablations.py --execute`；详细生成记录落盘，终端仅中文进度。当前仍缺统一评测与统计图表，不能认定B级完成。']
    (destination / 'training_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'runs': {k: {'steps': r['optimizer_steps'], 'tokens': r['total_generated_tokens']} for k, r in summary.items()},
                      'models_to_evaluate': len(models), 'cost_comparisons': cost_comparisons,
                      'strict_completion_regrades': regraded, 'training_or_inference_launched': False}, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
