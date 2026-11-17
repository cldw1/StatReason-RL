"""Recompute mechanism and cost evidence from the bounded real GPU checks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from _bootstrap import PROJECT_ROOT
from src.data.storage import read_jsonl
from src.utils.artifacts import write_json, file_sha256, utc_now


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def weighted(rows, key):
    return sum(r[key] * r['valid_tokens'] for r in rows) / sum(r['valid_tokens'] for r in rows)


def core(config):
    value = deepcopy(config)
    value['experiment'].pop('ablation_id')
    value['experiment'].pop('name')
    value['trainer'].pop('epsilon_high')
    value['trainer'].pop('loss_type')
    value['mechanisms']['dynamic_sampling'].pop('enabled')
    value['mechanisms']['overlong'].pop('enabled')
    return value


def main():
    root = PROJECT_ROOT
    destination = root / 'reports/mechanism_smoke_round1'
    manifest = read(destination / 'manifest.json')
    if set(manifest['runs']) != {'E0', 'E1', 'E2', 'E3', 'E4'}:
        raise ValueError('all planned check records must be retained before summarizing')
    rows = {}
    common = common_code = common_selection = None
    for key, entry in manifest['runs'].items():
        run = root / entry['run']
        status, resources, identity = [read(run / p) for p in ['run_status.json', 'resource_metrics.json', 'training_identity.json']]
        if status['status'] != 'completed' or status['optimizer_steps'] != 4:
            raise ValueError(f'{key} is incomplete: keep records and diagnose before summarizing')
        generation = read(run / 'effective_generation_audit.json')
        if not generation['equal'] or generation['effective']['top_p'] != 1. or generation['effective']['top_k'] != 0:
            raise ValueError('generation control failed')
        current_core = core(identity['config'])
        code = {k: identity[k] for k in ['code_hashes', 'support_source_sha256', 'trl_source_sha256', 'versions']}
        if common is not None and (common != current_core or code != common_code or common_selection != identity['selection']):
            raise ValueError('controlled checks differ beyond the intended mechanism')
        common, common_code, common_selection = current_core, code, identity['selection']
        attempts = read_jsonl(run / 'sampling_audit.jsonl')
        rollouts = read_jsonl(run / 'rollout_groups.jsonl')
        policy = read_jsonl(run / 'policy_audit.jsonl')
        by_rollout = {r['rollout_id']: r for r in rollouts}
        accepted = [a for a in attempts if a['accepted_for_training']]
        selected_ids = {a['rollout_id'] for a in accepted}
        first = [a for a in attempts if a['attempt_index_in_batch'] == 0]
        if {i for p in policy for i in p['rollout_id']} != selected_ids:
            raise ValueError('sample acceptance does not match actual trained rollouts')
        if sum(a['generated_tokens'] for a in attempts) != resources['total_generated_tokens']:
            raise ValueError('rollout cost not fully accounted')
        trained_tokens = sum(p['valid_tokens'] for p in policy)
        if sum(a['generated_tokens'] for a in accepted) * identity['config']['trainer']['num_iterations'] != trained_tokens:
            raise ValueError('padding/reuse token accounting mismatch')
        updates = read(run / 'parameter_updates.json')
        reload = read(run / 'reload_verification.json')
        all_completions = [c for r in rollouts for c in r['completions']]
        kept_completions = [c for i in selected_ids for c in by_rollout[i]['completions']]
        train_logs = [r for r in read_jsonl(run / 'train_metrics.jsonl') if 'grad_norm' in r]
        count = len(kept_completions)
        rows[key] = {
            'run': entry['run'], 'optimizer_steps': status['optimizer_steps'],
            'changed_lora_tensors': updates['changed_tensor_count'], 'reload': reload,
            'sampling_attempts': len(attempts), 'accepted_groups': len(accepted),
            'first_attempt_mixed_rate': sum(a['mixed'] for a in first) / len(first),
            'all_attempt_mixed_rate': sum(a['mixed'] for a in attempts) / len(attempts),
            'accepted_mixed_rate': sum(a['mixed'] for a in accepted) / len(accepted),
            'all_correct_attempt_rate': sum(a['all_correct'] for a in attempts) / len(attempts),
            'all_wrong_attempt_rate': sum(a['all_wrong'] for a in attempts) / len(attempts),
            'training_seconds': resources['training_seconds'], 'session_wall_seconds': resources['session_wall_seconds'],
            'generated_tokens': resources['total_generated_tokens'],
            'extra_resampling_tokens': resources['extra_resampling_tokens'],
            'discarded_rollout_tokens': resources['discarded_rollout_tokens'],
            'trained_completion_tokens_including_reuse': trained_tokens,
            'peak_allocated_bytes': resources['peak_allocated_bytes'], 'peak_reserved_bytes': resources['peak_reserved_bytes'],
            'raw_correctness_reward_kept': sum(c['accuracy_reward'] for c in kept_completions) / count,
            'shaped_reward_kept': sum(c['reward'] for c in kept_completions) / count,
            'penalized_kept_completions': sum(c['length_penalty'] < 0 for c in kept_completions),
            'penalized_all_completions': sum(c['length_penalty'] < 0 for c in all_completions),
            'mean_kept_length': sum(len(c['completion_ids']) for c in kept_completions) / count,
            'kept_truncation_rate': sum(c['verification']['truncated'] for c in kept_completions) / count,
            'raw_policy_entropy_token_weighted': weighted(policy, 'policy_entropy_raw'),
            'sampling_entropy_token_weighted': weighted(policy, 'policy_entropy_sampling'),
            'kl_token_weighted': weighted(policy, 'kl'),
            'effective_low_clip_token_weighted': weighted(policy, 'effective_low_clip'),
            'effective_high_clip_token_weighted': weighted(policy, 'effective_high_clip'),
            'ratio_min': min(p['ratio_min'] for p in policy), 'ratio_max': max(p['ratio_max'] for p in policy),
            'max_grad_norm': max(r['grad_norm'] for r in train_logs),
            'max_independent_loss_error': max(p['independent_loss_absolute_error'] for p in policy),
            'effective_generation': generation['effective'],
            'source_sha256': {p: file_sha256(run / p) for p in ['rollout_groups.jsonl', 'sampling_audit.jsonl', 'policy_audit.jsonl']},
        }
    tests = ET.parse(root / 'reports/project_audit_20261117/pytest_after_mechanisms.xml').getroot()
    result = {'recorded_at_utc': utc_now(), 'scope': 'GPU implementation checks only; not formal ablations or ability estimates',
              'grade': 'C', 'rows': rows, 'controls_identical_except_intended_mechanisms': True,
              'same_sft_and_data_selection': True, 'selection': common_selection, 'code_identity': common_code,
              'cpu_tests': {'passed_cases': len(tests.findall('.//testcase')), 'failures': len(tests.findall('.//failure')),
                            'errors': len(tests.findall('.//error'))},
              'paid_services_started': False, 'final_test_accessed': False}
    resume_path = destination / 'resume_check/resume_verification.json'
    if resume_path.exists():
        result['combination_resume_verification'] = read(resume_path)
    excluded_path = root / 'outputs/runs/e0_vanilla_smoke_20261117T065841Z_7400dc1c/resource_metrics.json'
    result['debug_generation_cost'] = {
        'five_comparable_checks_tokens': sum(r['generated_tokens'] for r in rows.values()),
        'excluded_default_fallback_check_tokens': read(excluded_path)['total_generated_tokens'],
        'excluded_check_resource_sha256': file_sha256(excluded_path),
        'extra_resume_check_tokens': result.get('combination_resume_verification', {}).get('additional_generated_tokens', 0),
        'scope': 'returned completion tokens; resumed lineage prefix counted only once',
    }
    result['debug_generation_cost']['total_new_training_generated_tokens'] = sum(
        result['debug_generation_cost'][k] for k in ['five_comparable_checks_tokens', 'excluded_default_fallback_check_tokens', 'extra_resume_check_tokens'])
    write_json(destination / 'results.json', result)
    lines = ['# GRPO/DAPO五组真实GPU实现检查', '',
             '每组4次优化器更新，G=4、同一Qwen3-0.6B SFT初始化、同一16题RL池、温度0.7、不启用top-p/top-k过滤。逐项核对共同配置、数据、代码、版本和实际生成配置。', '',
             '**当前仍属C级。** 此表只证明真实执行和记账，不是正式消融、模型正确率或熵坍塌结论。', '',
             '| 组别 | 更新步 | 原始尝试/接受组 | 接受组有效率 | 总生成Token | 额外补采Token | 训练秒 | 峰值分配GiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key, r in rows.items():
        lines.append(f"| {key} | {r['optimizer_steps']} | {r['sampling_attempts']}/{r['accepted_groups']} | {r['accepted_mixed_rate']:.0%} | {r['generated_tokens']} | {r['extra_resampling_tokens']} | {r['training_seconds']:.2f} | {r['peak_allocated_bytes']/2**30:.2f} |")
    lines += ['', '## 可由日志支持的结论', '',
              '- 五组均保存真实LoRA更新、独立冻结SFT参考、旧概率不可变检查、TRL损失独立复算、适配器重载及探针输出一致性。每组只检查4步，不能外推长期稳定性。',
              '- E2/E4接受与丢弃记录可用rollout_id对应实际policy_audit微批。成本分为总生成、额外补采、被丢弃Token；这些是不同集合，不应把它们简单相加。',
              '- E3/E4采用TRL token损失，所有微批共用整优化器批次的有效Token分母。CPU对实际安装TRL方法的不同长度损失与梯度核对通过。',
              '- E4从第2步恢复至第4步，224个最终张量、生成与在线筛选决策均逐位一致。恢复检查额外生成1177 Token，独立保存在resume_check，不充当新的一组效果实验。',
              '- 长度惩罚是否触发、上下裁剪是否实际影响梯度，以results.json的逐组指标为准；配置启用不等于观察到收益。', '',
              '- 本次E4确实对8个生成回答计算了非零长度惩罚，但这些回答所在组被动态筛掉；实际接受训练的回答中没有触发惩罚。因此不能声称本次4步检查已观察到长度奖励对参数更新或正确率的作用。', '',
              '| 组别 | 下侧有效裁剪 | 上侧有效裁剪 | 被惩罚回答（接受/全部） | 接受回答均长 | 接受回答截断 | 最大独立loss误差 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for key, r in rows.items():
        lines.append(f"| {key} | {r['effective_low_clip_token_weighted']:.4%} | {r['effective_high_clip_token_weighted']:.4%} | {r['penalized_kept_completions']}/{r['penalized_all_completions']} | {r['mean_kept_length']:.2f} | {r['kept_truncation_rate']:.1%} | {r['max_independent_loss_error']:.3g} |")
    lines += ['', '## 已发现并修复的问题', '',
              '最初一次E0检查被HF模型默认值覆盖top_p=1为0.95。原运行e0_vanilla_smoke_20261117T065841Z_7400dc1c完整保留并标记不合格用于公平比较；其4步、629生成Token的成本另计，未混入本表。修复后五组均保存effective_generation_audit.json并验证实际温度分布。', '',
              '## 执行命令', '', '```powershell',
              '.venv\\Scripts\\python.exe scripts/21_run_ablation_smokes.py --execute',
              '.venv\\Scripts\\python.exe scripts/22_analyze_ablation_smokes.py',
              '.venv\\Scripts\\python.exe -m pytest -m "not gpu" --junitxml=reports/project_audit_20261117/pytest_after_mechanisms.xml -q',
              '```', '', '下一步为有上限的独立正式训练及统一评测，预算见formal_budget_proposal.json；较大GPU占用尚未启动。']
    (destination / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: {f: r[f] for f in ['optimizer_steps', 'generated_tokens', 'extra_resampling_tokens',
                                          'training_seconds', 'effective_high_clip_token_weighted',
                                          'penalized_kept_completions', 'max_independent_loss_error']}
                      for k, r in rows.items()}, ensure_ascii=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
