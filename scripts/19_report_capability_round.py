"""仅根据已落盘预测重建本轮报告；缺失候选明确列为未评测。"""
from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

from _bootstrap import PROJECT_ROOT
from src.data.storage import read_jsonl
from src.evaluation.answer_diagnostics import diagnose_answer
from src.evaluation.answer_diagnostics_v2 import diagnose_answer_v2, PROTOCOL
from src.evaluation.capability_round import MODEL_NAMES, experiment_arms, prepare_panel, summarize_arm, paired_recipe_comparison
from src.utils.artifacts import write_json, utc_now, file_sha256


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    project = PROJECT_ROOT
    root = project / 'outputs/capability_round1'
    destination = project / 'reports/capability_round1'
    destination.mkdir(parents=True, exist_ok=True)
    panel, manifest = prepare_panel(project, root)
    planned_path = root / 'planned_arms.json'
    planned_arms = read_json(planned_path) if planned_path.exists() else experiment_arms()
    diagnosis = read_json(destination / 'existing_diagnosis.json')
    rows_by_arm, metrics, states = {}, {}, {}
    posthoc_rows = []
    for arm in planned_arms:
        directory = root / arm['name']
        if (directory / 'status.json').exists():
            states[arm['name']] = read_json(directory / 'status.json')
        else:
            path = root / f"{arm['model']}_availability.json"
            states[arm['name']] = read_json(path) if path.exists() else {'status': 'not_started'}
        path = directory / 'evaluation_predictions.jsonl'
        if not path.exists():
            continue
        rows = read_jsonl(path)
        for row in rows:
            actual = diagnose_answer(row['completion'], row, thinking=row['enable_thinking'], truncated=row['truncated'])
            if actual != row['answer_diagnostics'] or row['verification'] != actual['strict']:
                raise ValueError('新实验判题重算不一致')
        if {r['id'] for r in rows} - {r['id'] for r in panel}:
            raise ValueError('预测中存在面板外题目')
        if states[arm['name']]['status'] == 'completed' and len(rows) != len(panel):
            raise ValueError('完成状态与题目数不符')
        metrics[arm['name']] = summarize_arm(rows)
        numeric_by_source = {}
        for row in rows:
            diagnostics_v2 = diagnose_answer_v2(row['completion'], row, thinking=row['enable_thinking'], truncated=row['truncated'])
            if diagnostics_v2['strict'] != row['verification']:
                raise ValueError('事后诊断不能改变严格分数')
            source = row['source']
            numeric_by_source[source] = numeric_by_source.get(source, 0) + int(diagnostics_v2['numeric_correct_at_same_tolerance'])
            posthoc_rows.append({'arm': arm['name'], 'id': row['id'], 'diagnostics_v2': diagnostics_v2,
                                 'v1_numeric_correct': row['answer_diagnostics']['numeric_correct_at_same_tolerance']})
        metrics[arm['name']]['posthoc_numeric_v2_by_source'] = numeric_by_source
        metrics[arm['name']]['posthoc_numeric_v2_correct'] = sum(numeric_by_source.values())
        metrics[arm['name']]['posthoc_numeric_v2_protocol'] = PROTOCOL
        rows_by_arm[arm['name']] = rows
    (destination / 'posthoc_numeric_v2.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in posthoc_rows), encoding='utf-8')
    comparisons = {}
    pairs = [('qwen3_06b_nonthink_256', 'qwen3_06b_nonthink_512'),
             ('qwen3_06b_think_1024', 'qwen3_06b_think_2048'),
             ('qwen3_06b_nonthink_512', 'qwen3_06b_think_2048'),
             ('qwen3_17b_nonthink_256', 'qwen3_17b_nonthink_512'),
             ('qwen3_17b_think_1024', 'qwen3_17b_think_2048'),
             ('qwen3_06b_think_2048', 'qwen3_17b_think_2048'),
             ('qwen3_06b_nonthink_512', 'qwen3_17b_nonthink_512'),
             ('qwen3_06b_nonthink_512', 'qwen25_math_15b_common_512')]
    for a, b in pairs:
        if a in rows_by_arm and b in rows_by_arm and len(rows_by_arm[a]) == len(rows_by_arm[b]) == len(panel):
            comparisons[b + '_vs_' + a] = paired_recipe_comparison(rows_by_arm[a], rows_by_arm[b])
    all_complete = all(states[arm['name']]['status'] == 'completed' for arm in planned_arms)
    selection_decision = {'status': 'pending_complete_round', 'final_rl_model_frozen': False}
    if all_complete:
        def candidate_key(arm):
            metric = metrics[arm['name']]
            return (metric['posthoc_numeric_v2_by_source'].get('statreason_synthetic', 0),
                    metric['posthoc_numeric_v2_correct'], -metric['generation_seconds'])
        selected = max(planned_arms, key=candidate_key)
        chosen_metrics = metrics[selected['name']]
        selection_decision = {'status': 'short_sft_candidate_selected', 'selected_arm': selected,
            'selected_repository': MODEL_NAMES[selected['model']],
            'model_revision': rows_by_arm[selected['name']][0]['model_revision'],
            'criterion': 'statistics_numeric_v2_correct_then_overall_numeric_v2_then_generation_time',
            'statistics_correct': chosen_metrics['posthoc_numeric_v2_by_source'].get('statreason_synthetic', 0),
            'statistics_denominator': 28, 'overall_numeric_correct': chosen_metrics['posthoc_numeric_v2_correct'],
            'overall_denominator': 40, 'strict_correct': chosen_metrics['strict_correct'],
            'peak_allocated_bytes': chosen_metrics['peak_allocated_bytes'],
            'all_failure_and_truncation_cases_retained': True,
            'choice_scope': 'practical_recipe_on_development_not_pure_model_effect_or_statistical_optimality',
            'final_rl_model_frozen': False, 'new_training_performed': False,
            'requires_thinking_sft_support': selected['thinking'],
            'next_gate': 'actual_training_memory_probe_and_short_sft_domain_and_format_validation'}
    write_json(destination / 'candidate_selection.json', selection_decision)
    total_tokens = total_seconds = 0
    for path in root.glob('*/attempts.jsonl'):
        attempts = read_jsonl(path)
        total_tokens += sum(a['tokens'] for a in attempts)
        total_seconds += sum(a['seconds'] for a in attempts)
    test_path = destination / 'pytest_cpu.xml'
    tests = None
    if test_path.exists():
        suites = ET.parse(test_path).getroot().findall('testsuite')
        tests = {key: sum(int(s.attrib.get(key, 0)) for s in suites) for key in ('tests', 'failures', 'errors', 'skipped')}
    results = {'recorded_at_utc': utc_now(), 'panel': manifest, 'metrics': metrics, 'states': states,
               'comparisons': comparisons, 'cpu_tests': tests, 'candidate_selection': selection_decision,
               'all_recorded_attempt_tokens': total_tokens, 'all_recorded_generation_seconds': total_seconds,
               'paid_service_started': False, 'final_test_read': False,
               'strict_verifier_sha256': file_sha256(project / 'src/rewards/numeric.py'),
               'posthoc_numeric_v2_code_sha256': file_sha256(project / 'src/evaluation/answer_diagnostics_v2.py'),
               'derivation_code_sha256': file_sha256(Path(__file__)),
               'source_prediction_file_sha256': {name: file_sha256(root / name / 'evaluation_predictions.jsonl') for name in rows_by_arm}}
    write_json(destination / 'round_results.json', results)
    lines = ['# StatReason-RL：第二阶段首轮诊断与能力对照', '',
             '本报告由真实逐题日志重新判题生成。旧100题及本轮40题均为开发数据，不是最终测试。', '',
             '## 结论与范围', '',
             '历史标签、模板、数据长度及adapter身份审计通过；低分主要包含真实的建模/公式错误和数值计算错误。候选推理对照后先选择短程SFT模型，再经训练显存及领域/格式验证冻结正式RL基线；完整DAPO消融随后开展。', '',
             '## 历史100题核对', '',
             '| 检查点 | 全部正确 | 统计正确 | GSM8K正确 | 格式有效 | 截断 |',
             '|---|---:|---:|---:|---:|---:|']
    for name in ('Base', 'SFT', 'GRPO', 'SFT_revised'):
        r = diagnosis[name]
        s, g = r['by_source']['statreason_synthetic'], r['by_source']['openai/gsm8k']
        lines.append(f"| {name} | {r['strict_correct']}/100 | {s['correct']}/{s['n']} | {g['correct']}/{g['n']} | {r['format_valid']}% | {r['truncated']}% |")
    counts = diagnosis['SFT_semantic_review']['counts']
    lines += ['', '原SFT的100题逐条语义审阅，81题错误的主分类如下。分类互斥，截断/解析失败优先；不能把这些标签套到另一个模型的回答。', '',
              '| 主分类 | 数量 | 占100题 | 占81个错误 |', '|---|---:|---:|---:|']
    for key, label in [('formula_or_problem_modeling_error', '公式、建模或读题代入错误'),
                       ('correct_formula_numeric_error', '公式正确但数值计算错误'),
                       ('insufficient_work_to_disambiguate', '过程不足，无法细分'),
                       ('reasoning_truncated', '推理截断'), ('answer_parse_failure', '答案解析失败'),
                       ('correct_number_invalid_format', '数值正确但格式错误')]:
        n = counts[key]
        lines.append(f'| {label} | {n} | {n:.1f}% | {n / 81:.1%} |')
    audit = diagnosis['implementation_audit']
    lines += ['', f"实际重新编码{audit['audited_label_rows']}条标签记录；训练组成统计200/GSM8K1000，监督Token {audit['supervised_tokens']:,}。所有编码、mask和保存指纹相符，超长排除0题，400条历史预测的模板不一致0条。",
              '历史SFT实际执行300步，最终按验证损失选择第150步检查点。上面的监督Token是全量入选数据的编码总数，不能当作选中检查点的累计训练Token。',
              '224个LoRA张量有历史更新证据、保存重载logits匹配；当前adapter文件哈希与实际评测加载身份一致。700条非测试统计数据和70条旧开发统计题独立复算通过，未发现标准数值答案错误。未对全部GSM8K题作独立数学证明；宠物题有隐含无重叠假设，保留且注明。',
              '典型错误：写出6.9/√145却给1.12418193（真值约0.57301381）；将失败概率0.2当成功概率；在离散方差中忽略频数。0题属于同容差数值正确、仅格式失分。',
              '精度复核：62道有可解析数值的统计回答中，绝对误差≤0.0001的有3道、≤0.001的有4道。此前口头概述“放宽0.001仍为3道”应更正为4道；这只是误差分箱，严格指标仍是3/70，未更改容差或奖励。',
              '', '旧GRPO真实抽样12题、48回答，其中统计1题、GSM8K11题；有效混合奖励组4、全对组6、全错组2。该预算不足以证明统计计算能力得到训练。',
              '另核对原始192 Token试跑：Thinking为8/8截断、0/8正确；Non-thinking为0/8截断、1/8正确。该8题记录只用于确认预算问题，不作为能力结论。',
              '', '## 新40题首轮实测', '',
              '题组：新核心统计14、未使用的迁移验证14、未使用GSM8K验证12。所有40题此前未产生模型回答，28道统计题独立复算通过。共享生成器族，不能当作独立最终测试。',
              '严格判题保留原FINAL_ANSWER协议与容差；思考模式仅对闭合</think>后的实际答复判题，另存完整completion旧判法。唯一末尾boxed和行内标记仅用于同容差数值诊断，不混入奖励。所有截断均计错，失败样本不删除。', '',
              '| 模型/配置 | 严格正确 | 同容差数值诊断v2 | 统计严格正确 | GSM8K严格正确 | 格式率 | 截断率 | 生成Token | 生成秒 | 峰值allocated GiB |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in planned_arms:
        name = arm['name']
        if name not in metrics:
            continue
        m = metrics[name]
        s, g = m['by_source'].get('statreason_synthetic', {'strict_correct': 0, 'count': 0}), m['by_source'].get('openai/gsm8k', {'strict_correct': 0, 'count': 0})
        suffix = '' if states[name]['status'] == 'completed' else '（未完成）'
        lines.append(f"| {name}{suffix} | {m['strict_correct']}/{m['count']} | {m['posthoc_numeric_v2_correct']}/{m['count']} | {s['strict_correct']}/{s['count']} | {g['strict_correct']}/{g['count']} | {m['format_valid_rate']:.1%} | {m['truncation_rate']:.1%} | {m['total_generated_tokens']:,} | {m['generation_seconds']:.1f} | {m['peak_allocated_bytes'] / 1024**3:.3f} |")
    lines += ['', '数值诊断v2统一重算所有已保存回答，只补充接受唯一末尾boxed后的数学闭合符与句号，例如 `\\(\\boxed{37}\\).`。仍禁止从过程搜索数字，容差和截断政策不变。v1预测及数值指标保留，v2单列保存，不进入奖励或严格Bootstrap。', '',
              '| 模型/配置 | 统计数值正确（诊断v2） | GSM8K数值正确（诊断v2） |', '|---|---:|---:|']
    for name, m in metrics.items():
        source = m['posthoc_numeric_v2_by_source']
        counts = m['by_source']
        lines.append(f"| {name} | {source.get('statreason_synthetic', 0)}/{counts.get('statreason_synthetic', {}).get('count', 0)} | {source.get('openai/gsm8k', 0)}/{counts.get('openai/gsm8k', {}).get('count', 0)} |")
    lines += ['', 'Qwen3及Math匹配采样臂每题每配置一次随机生成；Math的native_greedy_512为一次greedy生成，遵循下载的官方generation_config默认do_sample=false。Qwen3非思考使用temperature=.7/top_p=.8/top_k=20，思考使用.6/.95/20；batch=4、min_p=0、BF16/SDPA、官方模板。生成时间为整批GPU运行时间按题均摊后求和，模型加载时间另存；完整指标还包括reserved显存、分题型、分层、平均长度及每次尝试成本。',
              '', '## 配对不确定性', '', '按相同题目重采样10000次，仅反映题目抽样；不覆盖生成seed、模板族和训练seed差异。跨模式比较是整套配方差异，不能全部归因于模型能力。', '']
    for name, c in comparisons.items():
        low, high = c['paired_bootstrap_95']
        lines.append(f"- {name}：差值{c['second_minus_first']:+.1%}，95%区间[{low:+.1%}, {high:+.1%}]。")
    if all_complete:
        lines += ['', '## 短程SFT候选选择', '',
                  f"本轮先选{selection_decision['selected_repository']}进行训练显存探针和短程SFT，开发配方为{selected['name']}：统计数值诊断{selection_decision['statistics_correct']}/28，全部数值诊断{selection_decision['overall_numeric_correct']}/40，严格正确{selection_decision['strict_correct']}/40。",
                  '选择优先依据统计题同容差数值正确数，其次全部数值正确数，再参考生成耗时。所有失败和截断均保留分母。它是开发阶段的实际配方选择，各原生配方预算可能不同；匹配512 Token控制另表报告，不能归因成纯模型规模提升，也不证明统计学上的唯一最优。',
                  '该选择尚未替换共同SFT或冻结正式RL模型。下一步实际测量LoRA/GRPO训练显存，并验证短程SFT能否改善领域数值正确率和统一答案格式。若选择Thinking模型，先补齐并验证Thinking SFT模板、标签和奖励最终回复边界。']
    lines += ['', '## 候选执行状态', '']
    missing = []
    for key, model_name in MODEL_NAMES.items():
        required = [a['name'] for a in planned_arms if a['model'] == key]
        if not all(states[name]['status'] == 'completed' for name in required):
            availability = read_json(root / f'{key}_availability.json') if (root / f'{key}_availability.json').exists() else {'status': 'not_completed'}
            missing.append(model_name)
            lines.append(f"- {model_name}：{availability['status']}。未将未运行结果记为0分；尚不能测得它的推理/训练峰值显存。")
    if missing:
        lines += ['', '未完成模型的原因以各模型availability.json和下载记录为准。最初会话有网络限制；2026-11-17重连后下载已可访问，并发现Windows首次并发探测符号链接的权限问题，已改为Windows串行下载。未出现真实推理/训练前，不能把下载失败解释为OOM；没有启动云GPU或产生费用。',
                  '', '以下命令可下载并完成/恢复固定开发对照：', '', '```powershell',
                  '.\\.venv\\Scripts\\python.exe scripts\\18_compare_capabilities.py --download-models',
                  '.\\.venv\\Scripts\\python.exe scripts\\19_report_capability_round.py', '```',
                  '', '若只想先下载而不占用GPU，可加`--download-only --models qwen3_17b qwen25_math_15b`。每模型检查下载文件大小上限4GiB，下载后固定真实commit和文件哈希。']
    else:
        lines += ['三个候选的已登记推理配置全部完成。这里只验证了推理可运行；LoRA和GRPO训练显存仍需实际探针。']
    lines += ['', '## 成本、检查与下一步', '',
              f'本轮已返回的全部生成尝试合计{total_tokens:,} Token、{total_seconds:.1f}秒，包含恢复时重复批次。若进程在generate返回前中断，该段成本无法精确统计。没有进行新SFT或GRPO训练，也未读取最终测试题正文或启动付费资源。']
    if tests:
        lines.append(f"CPU测试：{tests['tests']}项，失败{tests['failures']}、错误{tests['errors']}、跳过{tests['skipped']}；另有本轮目标Qwen3真实GPU推理日志，不将随机微型模型当作候选成绩。")
    lines += ['', '下一步对已选短程候选实际测量训练显存，构建经过独立复算和模板族/结构分组隔离的约2000统计+1000GSM8K候选SFT数据；先检验领域数值能力与统一输出格式，再冻结正式RL模型。有效统计能力确认后用训练候选池离线筛选混合奖励组，独立进行100至200步Vanilla GRPO。',
              '旧100题及本轮40题继续作开发集；正式方案冻结前另建封存的最终结构组，原困难测试题保留。DAPO、格式奖励及云GPU均未提前混入本轮实验。',
              '', '## 复现与文件', '',
              '- scripts/17_diagnose_existing.py：历史重算与标签审计。',
              '- scripts/18_compare_capabilities.py --prepare-only：冻结面板。',
              '- scripts/18_compare_capabilities.py --download-models：候选及预算对照。',
              '- python -m pytest -m "not gpu" -q --junitxml=reports/capability_round1/pytest_cpu.xml：CPU回归。',
              '- scripts/19_report_capability_round.py：从真实预测重建本报告。',
              '- reports/capability_round1/sft_error_review.jsonl：100题逐条语义分类及证据。',
              '- outputs/capability_round1/：冻结题组、身份、逐题回答、生成Token、显存、耗时及阻塞记录。',
              '- docs/capability_improvement_round1.md：官方资料、源码核对与本地设计决策。',
              '- docs/reference_project_alignment.md：参考截图的六组实验设计、诊断口径与pass@k定义。',
              '', '官方依据：[Open-R1](https://github.com/huggingface/open-r1)、[Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)、[Qwen3-1.7B采样说明](https://huggingface.co/Qwen/Qwen3-1.7B)、[DAPO论文v2](https://arxiv.org/abs/2503.14476v2)、[官方DAPO recipe](https://github.com/verl-project/verl-recipe/blob/main/dapo/README.md)。']
    (destination / 'capability_round1_report.md').write_text('\n'.join(lines), encoding='utf-8')
    status_path = project / 'reports/project_status.json'
    status = read_json(status_path)
    status.update(recorded_at_utc=utc_now(), next_phase='capability_improvement_before_dapo',
                  next_actions=['complete_candidate_model_development_comparison', 'select_model_and_generation_budget_from_development',
                                'build_structure_isolated_difficulty_stratified_sft_data', 'short_sft_then_training_only_rl_signal_screening'])
    status['capability_round1'] = {'report': 'reports/capability_round1/capability_round1_report.md',
        'historical_diagnosis_completed': True, 'panel_count': 40,
        'completed_arms': [name for name in rows_by_arm if states[name]['status'] == 'completed'],
        'pending_candidates': missing, 'cpu_tests': tests, 'dapo_started': False}
    status['capability_round1']['candidate_selection'] = selection_decision
    if all_complete:
        status['active_workstream'] = 'training_memory_and_statistics_curriculum_before_new_sft'
        status['next_actions'] = ['probe_selected_model_training_memory', 'prepare_difficulty_stratified_statistics_sft',
                                  'validate_short_sft_numeric_accuracy_and_format', 'screen_training_only_rl_signal']
    write_json(status_path, status)
    print(json.dumps({'completed_arms': status['capability_round1']['completed_arms'], 'pending': missing,
                      'tokens': total_tokens, 'seconds': total_seconds, 'tests': tests}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
