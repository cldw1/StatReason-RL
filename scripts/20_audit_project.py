"""Recheck retained artifacts without opening held-out test question bodies."""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import inspect
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from _bootstrap import PROJECT_ROOT
from src.data.schema import validate_records
from src.data.storage import read_jsonl
from src.utils.artifacts import file_sha256, utc_now, write_json
from src.utils.config import load_config


RUNS = {
    'SFT': 'sft_local_epoch1_20261115T173458Z_7e5291a1',
    'GRPO': 'grpo_local_24_20261116T094046Z_0f8144b9',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def reload_adapters(root, config):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    from src.models.inference import load_qwen
    from src.models.lora import tensor_sha256, validate_adapter
    from src.models.snapshot import resolve_snapshot
    from src.training.sft import _probe_logits

    snapshot = resolve_snapshot(config['model'])
    tokenizer = AutoTokenizer.from_pretrained(snapshot.path, local_files_only=True)
    feature = {'input_ids': tokenizer.encode('Compute 2 + 3. Give one numeric answer.', add_special_tokens=False)}
    results = {}
    for label, name in RUNS.items():
        directory = root / 'outputs/runs' / name
        adapter = directory / 'adapter'
        identity = validate_adapter(adapter, snapshot.repository, snapshot.revision)
        expected = {r['name']: r['final_sha256'] for r in read(directory / 'parameter_updates.json')['parameters']}
        probes = []
        for _ in range(2):
            base, _, _ = load_qwen(snapshot, config['model'])
            model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False, local_files_only=True).eval()
            actual = {n: tensor_sha256(p) for n, p in model.named_parameters() if n in expected}
            if actual != expected:
                raise RuntimeError(f'{label}: retained adapter does not match recorded post-update tensors')
            probes.append(_probe_logits(model, feature))
            del model, base
            gc.collect()
            torch.cuda.empty_cache()
        torch.testing.assert_close(probes[0], probes[1], rtol=0, atol=0)
        results[label] = {**identity, 'recorded_final_tensors_rechecked': len(expected),
                          'two_fresh_loads_logits_exact': True,
                          'max_abs_logits_difference': float((probes[0] - probes[1]).abs().max()),
                          'scope': 'fresh inference reload now; optimizer resume evidence checked separately'}
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reload-gpu', action='store_true')
    args = parser.parse_args()
    root = PROJECT_ROOT
    destination = root / 'reports/project_audit_20261117'
    destination.mkdir(parents=True, exist_ok=True)
    from trl import GRPOConfig, GRPOTrainer
    from src.utils.environment import package_versions

    result = {'audited_at_utc': utc_now(), 'paid_resources_started': False,
              'final_test_bodies_opened': False, 'versions': package_versions()}
    git = subprocess.run(['git', 'status', '--short', '--branch'], cwd=root, capture_output=True, text=True)
    result['git'] = {'is_repository': git.returncode == 0, 'return_code': git.returncode,
                     'message': (git.stdout + git.stderr).strip()}
    result['source_sha256'] = {p.relative_to(root).as_posix(): file_sha256(p)
                              for folder in ['src', 'scripts', 'tests', 'configs']
                              for p in (root / folder).rglob('*') if p.suffix in {'.py', '.yaml'}}
    smoke_path = root / 'reports/mechanism_smoke_round1/manifest.json'
    smoke = read(smoke_path) if smoke_path.exists() else {'runs': {}}
    mechanism_updates = {k: v['status']['optimizer_steps'] for k, v in smoke['runs'].items()
                         if v['status'].get('model_training_performed')}
    mechanism_supported = 'MechanismGRPOTrainer' in (root / 'src/training/grpo.py').read_text(encoding='utf-8')
    fields = ['loss_type', 'scale_rewards', 'beta', 'epsilon', 'epsilon_high', 'num_generations',
              'num_iterations', 'max_completion_length', 'mask_truncated_completions']
    result['installed_trl'] = {'defaults': {f: GRPOConfig.__dataclass_fields__[f].default for f in fields},
                               'trainer_source_sha256': file_sha256(Path(inspect.getfile(GRPOTrainer))),
                               'loss_implementation': 'TRL integration plus independent formula audit',
                               'dynamic_sampling_current_entrypoint': 'controlled_ablation_supported' if mechanism_supported else 'rejected_by_validate_vanilla',
                               'dapo_training_evidence_at_audit': bool(mechanism_updates),
                               'bounded_smoke_optimizer_steps': mechanism_updates,
                               'formal_ablation_training_evidence': False}
    diagnostic = read(root / 'reports/capability_round1/existing_diagnosis.json')
    result['existing_prediction_recheck'] = {k: diagnostic[k] for k in ['Base', 'SFT', 'GRPO', 'SFT_revised']}
    result['errors'] = diagnostic['SFT_semantic_review']
    result['implementation_audit'] = diagnostic['implementation_audit']
    result['runs'] = {}
    for label, name in RUNS.items():
        directory = root / 'outputs/runs' / name
        updates = read(directory / 'parameter_updates.json')
        result['runs'][label] = {'directory': directory.relative_to(root).as_posix(),
                                 'status': read(directory / 'run_status.json'),
                                 'update_summary': {k: v for k, v in updates.items() if k != 'parameters'},
                                 'historical_reload': read(directory / 'reload_verification.json'),
                                 'resources': read(directory / 'resource_metrics.json'),
                                 'adapter_file_sha256': file_sha256(directory / 'adapter/adapter_model.safetensors')}
    grpo = root / 'outputs/runs' / RUNS['GRPO']
    rollouts = read_jsonl(grpo / 'rollout_groups.jsonl')
    groups = [g for r in rollouts for g in r['groups']]
    policy = read_jsonl(grpo / 'policy_audit.jsonl')
    tokens = sum(r['valid_tokens'] for r in policy)
    result['grpo_behavior'] = {**diagnostic['GRPO_rollout_audit'],
                              'effective_group_rate': sum(not g['zero_variance'] for g in groups) / len(groups),
                              'policy_audit_tokens_with_reuse': tokens,
                              'loss_affecting_clip_rate': sum((r['effective_low_clip'] + r['effective_high_clip']) * r['valid_tokens'] for r in policy) / tokens,
                              'ratio_min': min(r['ratio_min'] for r in policy),
                              'ratio_max': max(r['ratio_max'] for r in policy),
                              'reference': read(grpo / 'reference_verification.json'),
                              'resume': read(root / 'reports/grpo/resume_verification.json')}
    result['data'] = {}
    for dataset in ['statistics_full', 'gsm8k']:
        directory = root / 'data/processed/statreason-v1' / dataset
        manifest = read(directory / 'manifest.json')
        non_test = []
        hashes = {}
        for filename, expected in manifest['file_sha256'].items():
            # Hashing a sealed file does not decode or expose its questions.
            path = directory / filename
            actual = file_sha256(path)
            if actual != expected:
                raise RuntimeError(f'data hash mismatch: {path}')
            hashes[filename] = actual
            if path.suffix == '.jsonl' and filename != 'test.jsonl':
                non_test.extend(read_jsonl(path))
        result['data'][dataset] = {'manifest_counts': manifest['statistics']['by_split'],
                                    'non_test_difficulty_counts': dict(Counter(r.get('difficulty', 'unspecified') for r in non_test)),
                                    'non_test_revalidated': validate_records(non_test),
                                    'file_sha256': hashes,
                                    'isolation_limit': 'shared skills remain; family isolation is not universal OOD isolation'}
    round_report = read(root / 'reports/capability_round1/round_results.json')
    result['candidate_comparisons'] = {
        'panel_size': 40, 'scope': 'raw-model development only; prompt and budget controls preserved',
        'arms': {name: {k: values[k] for k in ['count', 'strict_correct', 'numeric_correct',
                                             'posthoc_numeric_v2_correct', 'truncated', 'total_generated_tokens',
                                             'generation_seconds', 'peak_allocated_bytes']}
                 for name, values in round_report['metrics'].items()},
        'previous_selection': round_report['candidate_selection'],
        'new_main_training_choice': 'Qwen3-0.6B retained SFT for bounded mechanism ablations',
        'reason': 'real strict successes and mixed reward groups; compatible SFT and measured local training; avoid new SFT cycle',
    }
    xml = ET.parse(destination / 'pytest_cpu.xml').getroot()
    cases = xml.findall('.//testcase')
    result['cpu_tests'] = {'test_cases': len(cases), 'failures': len(xml.findall('.//failure')),
                           'errors': len(xml.findall('.//error')), 'xml_sha256': file_sha256(destination / 'pytest_cpu.xml')}
    if args.reload_gpu:
        result['fresh_gpu_reload'] = reload_adapters(root, load_config(root / 'configs/grpo.yaml'))
    elif (destination / 'audit.json').exists() and 'SFT' in read(destination / 'audit.json').get('fresh_gpu_reload', {}):
        previous = read(destination / 'audit.json')
        result['fresh_gpu_reload'] = previous['fresh_gpu_reload']
        result['fresh_gpu_reload_retained_from_audit_at_utc'] = previous.get('fresh_gpu_reload_retained_from_audit_at_utc', previous['audited_at_utc'])
    else:
        result['fresh_gpu_reload'] = {'performed': False, 'historical_only': True}
    result['acceptance'] = {'grade': 'C', 'reason': 'no real DAPO mechanism ablations yet',
                            'missing_for_B': ['E0-E3 independently initialized measured training',
                                              'E4 combination actual execution', 'uniform frozen evaluation',
                                              'mechanism and token-cost analysis with uncertainty']}
    write_json(destination / 'audit.json', result)
    rows = [
        ('环境与版本', '可运行；TRL 0.23.1', 'audit.json；pytest_cpu.xml', 'TRL默认loss=dapo；必须显式指定', '保留版本'),
        ('Git状态', '当前目录没有Git仓库', 'audit.json:git', '缺少提交历史；已有文件哈希和代码快照', '先保留快照；不自动推送'),
        ('数据与隔离', 'v1非测试数据重验、文件哈希核对通过', 'audit.json:data；data/processed/statreason-v1/*/manifest.json', '模板族隔离不等同于所有数学技能隔离；原v1无显式难度标签', '报告隔离边界；按预先固定结构补充分层元数据'),
        ('数值奖励', '纯二元严格奖励；边界测试通过', 'src/rewards/numeric.py；tests/test_numeric_verifier.py', 'boxed仅诊断；不能混入主奖励', '保持统一协议'),
        ('SFT', '300步；导出验证loss选中的150步；224张量真实变化', 'outputs/runs/' + RUNS['SFT'], '领域数据200条，GSM1000条；统计能力弱', '复用原检查点'),
        ('SFT标签与模板', '1264条重编码一致；未发现prompt监督/模板错误', 'reports/capability_round1/existing_diagnosis.json', '只核实当前旧Non-thinking流程', '不据此声称Thinking训练兼容'),
        ('Vanilla GRPO', '24步；12组，4组奖励有差异；重载/恢复通过', 'outputs/runs/' + RUNS['GRPO'], '仅短程开发；11/12组为GSM；裁剪极少', '统一新协议后作为历史证据'),
        ('旧策略/KL/更新', '独立冻结SFT参考；旧概率复用；224张量更新', 'policy_audit.jsonl；reference_verification.json；resume_verification.json', '旧top-p/top-k采样与温度策略logprob并非同一分布', '新五组统一取消截断分布过滤'),
        ('DAPO四机制', '已接通，五组短程检查' if mechanism_updates else '仅配置草案；当前训练入口拒绝',
         'reports/mechanism_smoke_round1' if mechanism_updates else 'configs/dapo_*.yaml；src/training/grpo.py',
         '尚无正式消融；每组4步仅属实现检查' if mechanism_updates else '没有任何DAPO训练证据',
         '按冻结配置完成正式对照' if mechanism_updates else '实现、测试并做真实GPU闭环'),
        ('三模型能力对照', '12配置×40题真实推理完成', 'reports/capability_round1/round_results.json；outputs/capability_round1', '开发集；Math boxed数值与严格分数必须区分', '保留；暂不再开新SFT周期'),
        ('评测与统计', '旧100题逐题结果、Wilson/配对Bootstrap可复算', 'src/evaluation/metrics.py；comparison.py；reports/grpo', '100题已用于开发；没有冻结最终评测', '消融冻结后再开最终测试'),
        ('B级验收', 'C级，尚未达到', 'audit.json:acceptance', '缺少多个机制真实训练、公平评测与成本分析', '按下述最短路线执行'),
    ]
    lines = ['# StatReason-RL 当前工程审查', '', f"核查时间：{result['audited_at_utc']}。只依据当前文件；没有打开最终测试题正文，没有启动付费服务。", '',
             '| 功能模块 | 当前状态 | 证据文件 | 存在问题 | 是否需要修改 |', '|---|---|---|---|---|']
    lines += ['| ' + ' | '.join(row) + ' |' for row in rows]
    lines += ['', '## 真实诊断', '',
              '旧100题统一降级为开发集：Base 2/100；SFT 19/100（GSM 16/30、统计3/70）；GRPO 19/100。GRPO相对SFT有3题变对、3题变错，不能声称提升。', '',
              '| SFT错误主类别 | 数量 | 占100题 | 占81道错误 |', '|---|---:|---:|---:|']
    labels = {'formula_or_problem_modeling_error': '公式/建模/读题', 'correct_formula_numeric_error': '公式正确但计算错误',
              'reasoning_truncated': '生成截断', 'answer_parse_failure': '解析失败',
              'insufficient_work_to_disambiguate': '过程不足以细分', 'correct_number_invalid_format': '数值正确但格式不合规'}
    for key, label in labels.items():
        n = result['errors']['counts'][key]
        lines.append(f'| {label} | {n} | {n:.1f}% | {n/81:.1%} |')
    lines += ['', '语义分类只针对SFT逐题人工规则审查，不外推到其他模型。700道非测试统计题及旧开发集70道统计题独立计算未发现数值标准答案错误；GSM8K存在一道人宠物题的隐含不重叠假设，保留题目，不宣称全部GSM题已独立证明。代表性原回答与证据：../capability_round1/sft_error_review.jsonl。', '',
              '## 达到B级的最短路线', '',
              '1. 固定Qwen3-0.6B原SFT导出检查点，不再以换模型或完整重训SFT作为启动消融的前置条件。已有4/12混合奖励组足以启动实现验证，但不足以代表完整训练池。',
              '2. 保留旧24步结果；建立新的共同协议：明确grpo/token损失、非对称裁剪、temperature策略概率、同一冻结SFT参考。新五组取消top-p/top-k过滤，避免过滤采样与logprob不匹配。',
              '3. E0基线、E1仅Clip-Higher、E2仅在线Dynamic Sampling、E3仅Token-level Loss、E4四机制组合。独立长度消融E5为可选，不阻塞最小完成。',
              '4. 数学、padding、组筛选与补采预算、梯度累积归一化及重载需要独立验证。若存在后续实现检查，详见reports/mechanism_smoke_round1/report.md。离线RL选题与在线补采分别记账。',
              '5. 根据新协议的实测吞吐确定有上限的正式训练预算；较大本地GPU任务和任何付费任务先给出具体预算再等待用户决定。统一开发集只用于排错；训练和评测方案冻结后再使用独立最终测试。',
              '6. 使用真实结果判定A/B/C。没有准确率改善可走B路线；没有触发裁剪也如实报告。多次采样采用题目级聚类，报告Token成本和配对区间。', '',
              '## 本阶段执行命令', '', '```powershell',
              '.venv\\Scripts\\python.exe scripts/17_diagnose_existing.py',
              '.venv\\Scripts\\python.exe -m pytest -m "not gpu" --junitxml=reports/project_audit_20261117/pytest_cpu.xml -q',
              '.venv\\Scripts\\python.exe scripts/20_audit_project.py --reload-gpu', '```', '',
              '本阶段没有新增训练。审查基线的参数更新证据属于历史真实运行；本次GPU重载结果另存audit.json:fresh_gpu_reload，不能写成重新训练成功。CPU实际执行192项测试通过，1项GPU测试未在CPU命令中运行。', '',
              '## 参考源码', '',
              '- https://huggingface.co/docs/trl/v0.23.1/grpo_trainer',
              '- https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_trainer.py',
              '- https://github.com/huggingface/open-r1', '- https://github.com/QwenLM/Qwen2.5-Math',
              '- https://dapo-sia.github.io/', '- https://arxiv.org/abs/2503.14476', '',
              '实现归属：基于TRL集成、配置和验证；本项目负责严格数值奖励、数据隔离、审计、机制适配和实验分析，不能称整个GRPOTrainer从零手写。']
    (destination / 'audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'audit': str(destination / 'audit.md'), 'grade': 'C', 'cpu_tests': result['cpu_tests'],
                      'gpu_reload_performed': args.reload_gpu, 'actual_new_training': False}, ensure_ascii=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
