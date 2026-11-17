"""CPU-only final audit, paired statistics, figures and experiment report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.evaluation.ablation_analysis import audit_and_analyze
from src.utils.artifacts import file_sha256, utc_now, write_json


COLORS = {'E0': '#2864a0', 'E1': '#d47a27', 'E2': '#40976b', 'E3': '#8c61a0', 'E4': '#ce4f53'}
TRAIN_NOTE = 'Qwen3-0.6B | same SFT, RL pool=400, G=4, reuse=2, seed=42 | E2:30 updates; others:100'
EVAL_NOTE = 'Frozen holdout: statistics 200 + GSM8K train-reserved 200 | non-thinking, 256 tokens, T=0.7, top-p=1, top-k=0'


def percentage(value):
    return f'{value * 100:.2f}%'


def interval(value):
    return f'[{value[0]*100:+.2f}, {value[1]*100:+.2f}]'


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def model_table(result):
    rows = []
    for key, metric in result['metrics'].items():
        model = result['models'][key]
        rows.append({'model': key, 'rl_steps': model['optimizer_step'], 'count': metric['question_count'],
                     'strict_correct': metric['strict_correct_count'], 'strict_accuracy': metric['sampled_pass_at_1'],
                     'wilson_low': metric['wilson_95_interval'][0], 'wilson_high': metric['wilson_95_interval'][1],
                     'gsm8k_correct_of_200': metric['by_source']['openai/gsm8k']['strict_correct_count'],
                     'statistics_correct_of_200': metric['by_source']['statreason_synthetic']['strict_correct_count'],
                     'numeric_accuracy': metric['numeric_correct_rate'], 'format_valid_rate': metric['format_valid_rate'],
                     'parse_failure_rate': metric['parse_failure_rate'], 'truncation_rate': metric['truncation_rate'],
                     'evaluation_generated_tokens': metric['total_generated_tokens'],
                     'evaluation_generation_seconds': metric['generation_seconds'],
                     'evaluation_peak_allocated_bytes': result['evaluation_evidence'][key]['peak_allocated_bytes'],
                     'training_generated_tokens_at_checkpoint': model.get('training_generated_tokens', 0)})
    return rows


def make_summary_figure(result, destination):
    """Render saved results with explicit checkpoint-versus-whole-run token cost."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.7))
    ax.axis('off')
    content = []
    for name in result['models']:
        metric = result['metrics'][name]
        content.append([
            name, str(metric['strict_correct_count']) + '/400',
            str(metric['by_source']['openai/gsm8k']['strict_correct_count']) + '/200',
            str(metric['by_source']['statreason_synthetic']['strict_correct_count']) + '/200',
            percentage(metric['format_valid_rate']), percentage(metric['truncation_rate']),
            str(result['models'][name].get('training_generated_tokens', '-')),
        ])
    table = ax.table(cellText=content,
                     colLabels=['Model', 'Strict', 'GSM8K', 'Stats', 'Format', 'Truncated',
                                'RL tokens\nat checkpoint'], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.55)
    for column in range(7):
        table[0, column].set_height(table[0, column].get_height() * 1.35)
    ax.set_title('Complete ablation evaluation (E2/100 missing; never imputed)')
    checkpoint_tokens = result['models']['E2_step30']['training_generated_tokens']
    whole_run_tokens = result['training']['rows']['E2']['total_generated_tokens']
    tail_tokens = whole_run_tokens - checkpoint_tokens
    fig.text(.5, .075, 'RL token column: generation through each checkpoint; later attempts are excluded.',
             ha='center', fontsize=7)
    fig.text(.5, .045,
             f'E2: {checkpoint_tokens:,} at step 30 + {tail_tokens:,} subsequent untrained tokens '
             f'= {whole_run_tokens:,} whole-run generation cost.', ha='center', fontsize=7)
    fig.text(.5, .015, EVAL_NOTE, ha='center', fontsize=7)
    fig.tight_layout(rect=[0, .13, 1, 1])
    path = destination / '09_ablation_summary.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path


def make_figures(result, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})
    destination.mkdir(parents=True, exist_ok=True)
    files = []

    def save(fig, filename, note=TRAIN_NOTE):
        fig.text(.5, .005, note, ha='center', fontsize=7)
        fig.tight_layout(rect=[0, .04, 1, 1])
        path = destination / filename
        fig.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        files.append(path)

    def style(axes):
        for axis in np.asarray(axes).reshape(-1):
            axis.grid(alpha=.2)
            axis.spines[['top', 'right']].set_visible(False)
            axis.set_xlabel('Optimizer update')

    fig, ax = plt.subplots(figsize=(8.5, 4))
    for key, data in result['training_series'].items():
        points = data['updates']
        ax.plot([p['step'] for p in points], [p['raw_reward'] for p in points], color=COLORS[key], label=key, alpha=.8)
    p = result['training_series']['E4']['updates']
    ax.plot([v['step'] for v in p], [v['shaped_reward'] for v in p], '--', color=COLORS['E4'], label='E4 shaped reward')
    ax.set(title='Accepted-rollout rewards (same group reused twice)', ylabel='Mean reward', ylim=(-1.05, 1.05))
    style([ax]); ax.legend(ncol=3)
    save(fig, '01_reward.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, data in result['training_series'].items():
        p = data['updates']; x = [v['step'] for v in p]
        axes[0].plot(x, [v['policy_entropy_raw'] for v in p], color=COLORS[key], label=key)
        axes[1].plot(x, [v['policy_entropy_sampling'] for v in p], color=COLORS[key], label=key)
    axes[0].set(title='Raw policy entropy', ylabel='Token-weighted entropy (nat)', ylim=(0, None))
    axes[1].set(title='Temperature/sampling entropy', ylabel='Token-weighted entropy (nat)', ylim=(0, None))
    style(axes); axes[0].legend(ncol=3)
    save(fig, '02_entropy.png')

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for key, data in result['training_series'].items():
        attempts = data['sampling']; mixed = kept = kept_mixed = 0
        x, effective, accepted = [], [], []
        for index, a in enumerate(attempts, 1):
            mixed += a['mixed']; kept += a['accepted_for_training']
            kept_mixed += a['accepted_for_training'] and a['mixed']
            x.append(a['optimizer_step_before_batch']); effective.append(100*mixed/index)
            accepted.append(100*kept_mixed/kept if kept else float('nan'))
        axes[0].plot(x, effective, color=COLORS[key], label=key)
        axes[1].plot(x, [100-v for v in effective], color=COLORS[key], label=key)
        axes[2].plot(x, accepted, color=COLORS[key], label=key)
    axes[0].set(title='Cumulative mixed groups / all attempts', ylabel='Effective group percentage', ylim=(0, 105))
    axes[1].set(title='Cumulative zero-variance / all attempts', ylabel='Zero-variance group percentage', ylim=(0, 105))
    axes[2].set(title='Cumulative mixed groups / trained groups', ylabel='Effective group percentage', ylim=(0, 105))
    style(axes); axes[0].legend(ncol=3)
    save(fig, '03_effective_groups.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key in ['E0', 'E1']:
        p = result['training_series'][key]['updates']; x = [v['step'] for v in p]
        axes[0].plot(x, [v['effective_high_clip']*100 for v in p], label=key, color=COLORS[key])
        axes[1].plot(x, [v['effective_low_clip']*100 for v in p], label=key, color=COLORS[key])
    axes[0].set(title='Loss-affecting upper clipping', ylabel='Percent of completion tokens', ylim=(0, None))
    axes[1].set(title='Loss-affecting lower clipping', ylabel='Percent of completion tokens', ylim=(0, None))
    style(axes); axes[0].legend()
    save(fig, '04_clip_higher.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, data in result['training_series'].items():
        p = data['updates']; x = [v['step'] for v in p]
        axes[0].plot(x, [v['mean_length'] for v in p], color=COLORS[key], label=key)
        axes[1].plot(x, [100*v['truncation_rate'] for v in p], color=COLORS[key], label=key)
    axes[0].set(title='Accepted response length (EOS included)', ylabel='Tokens', ylim=(0, 132))
    axes[1].set(title='Accepted response truncation', ylabel='Percent', ylim=(0, 105))
    style(axes); axes[0].legend(ncol=3)
    save(fig, '05_length_truncation.png')

    names = list(result['models'])
    for source, filename, title in [('openai/gsm8k', '06_gsm8k_accuracy.png', 'GSM8K held-out TRAIN-reserved accuracy'),
                                    ('statreason_synthetic', '07_statistics_accuracy.png', 'Statistics held-out accuracy')]:
        fig, ax = plt.subplots(figsize=(12, 4.4))
        parts = [result['metrics'][n]['by_source'][source] for n in names]
        values = np.array([p['sampled_pass_at_1']*100 for p in parts])
        low = values - np.array([p['wilson_95_interval'][0]*100 for p in parts])
        high = np.array([p['wilson_95_interval'][1]*100 for p in parts]) - values
        colors = [COLORS.get(n.split('_')[0], '#7b8793') for n in names]
        bars = ax.bar(range(len(names)), values, yerr=[low, high], color=colors, capsize=3)
        for bar, p in zip(bars, parts):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{p['strict_correct_count']}/200", ha='center', fontsize=8)
        ax.set_xticks(range(len(names)), [n.replace('_step', '\n') for n in names], rotation=25, ha='right')
        ax.set(title=title + ' (Wilson 95% intervals)', ylabel='Accuracy (%)', ylim=(0, 100))
        ax.grid(axis='y', alpha=.2)
        save(fig, filename, EVAL_NOTE)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    keys = list(result['training']['rows'])
    total = np.array([result['training']['rows'][k]['total_generated_tokens'] for k in keys])
    discarded = np.array([result['training']['rows'][k]['discarded_rollout_tokens'] for k in keys])
    axes[0].bar(keys, total-discarded, label='Used-response tokens', color='#3474a5')
    axes[0].bar(keys, discarded, bottom=total-discarded, label='Discarded group tokens', color='#c5a875')
    axes[0].set(title='Total training generation cost', ylabel='Returned completion tokens')
    axes[0].legend(fontsize=8)
    axes[1].bar(keys, [result['training']['rows'][k]['training_seconds']/60 for k in keys], color=[COLORS[k] for k in keys])
    axes[1].set(title='Observed local training wall time', ylabel='Minutes')
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    save(fig, '08_generation_cost.png')

    files.append(make_summary_figure(result, destination))

    fig, grid = plt.subplots(2, 2, figsize=(10, 7))
    axes = grid.reshape(-1)
    for key, data in result['training_series'].items():
        p = data['updates']; x = [v['step'] for v in p]
        axes[0].plot(x, [v['policy_loss'] for v in p], color=COLORS[key], label=key)
        axes[1].plot(x, [v['kl'] for v in p], color=COLORS[key])
        axes[2].plot(x, [v['grad_norm'] for v in p], color=COLORS[key])
        axes[3].plot(x, [v['ratio_max'] for v in p], color=COLORS[key], alpha=.8)
        axes[3].plot(x, [v['ratio_min'] for v in p], color=COLORS[key], alpha=.4)
    axes[0].set(title='Policy loss (declared answer/token aggregation)', ylabel='Policy objective')
    axes[0].axhline(0., color='gray', linewidth=.7)
    axes[1].set(title='Reference KL estimator', ylabel='Token-weighted KL', ylim=(0, None))
    axes[2].set(title='Observed gradient norm', ylabel='Norm', ylim=(0, None))
    axes[3].set(title='Token ratio extrema by update', ylabel='Current / old probability', ylim=(0, None))
    axes[3].axhline(1., color='black', linestyle='--', linewidth=1)
    style(axes); axes[0].legend(ncol=3)
    save(fig, '10_kl_gradient_ratio.png')

    selected = ['E4_step100_vs_SFT', 'E4_step100_vs_E0_step100', 'E4_step30_vs_E0_step30']
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for index, name in enumerate(selected):
        c = result['comparisons'][name]; value = c['second_minus_first_accuracy']*100
        lo, hi = [x*100 for x in c['paired_bootstrap_95']]
        ax.errorbar(value, index, xerr=[[value-lo], [hi-value]], fmt='o', capsize=5, color='#2864a0')
    ax.axvline(0, color='gray', linestyle='--')
    ax.set_yticks(range(3), ['E4/100 - SFT', 'E4/100 - E0/100', 'E4/30 - E0/30'])
    ax.set(title='Question-paired accuracy differences', xlabel='Percentage points (unadjusted bootstrap 95% intervals)')
    ax.grid(axis='x', alpha=.2)
    save(fig, '11_paired_intervals.png', EVAL_NOTE)
    return files


def write_report(root, result, charts):
    out = root / 'reports/ablation_round1'
    metrics = result['metrics']; comparisons = result['comparisons']
    rows = model_table(result)
    write_csv(out / 'ablation_summary.csv', rows)
    write_csv(out / 'paired_comparisons.csv', [
        {'comparison': key, 'first': c['first_model'], 'second': c['second_model'],
         'delta_pp': 100*c['second_minus_first_accuracy'], 'ci_low_pp': 100*c['paired_bootstrap_95'][0],
         'ci_high_pp': 100*c['paired_bootstrap_95'][1], 'gains': c['second_only_correct'], 'losses': c['first_only_correct'],
         'mcnemar_p': c['mcnemar_exact_two_sided_p'], 'mcnemar_holm_p': c['mcnemar_holm_adjusted_p']}
        for key, c in comparisons.items()])
    lines = ['# StatReason-RL：GRPO/DAPO受控机制实验报告', '',
             '本文报告预算受限条件下的GRPO/DAPO受控机制实验。结论基于真实训练、统一判题和完整结果校验，不宣称复现论文规模或证明DAPO显著优于GRPO。', '',
             '## 模型、数据与方法', '',
             'Qwen3-0.6B，BF16/SDPA，RTX 5060 Laptop 8GB。LoRA r=8、alpha=16，q/k/v/o四类投影，共224个可训练张量、2,293,760参数。共用同一个SFT适配器（SFT运行300步，验证loss选择150步），RL池400题：GSM300、统计100。', '',
             '基于TRL 0.23.1集成Rollout、裁剪与损失，项目实现奖励、独立SFT参考、在线补采、长度整形、数学审计和统计分析；不是从零手写整个Trainer。每组独立初始化，G=4、旧概率复用2次、LR=1e-5、beta=0.04、种子42。E0显式loss_type=grpo；E1上裁剪0.28；E2在线筛选；E3 token损失；E4四机制组合。E4保留共同KL，区别于论文去KL的完整配方。', '',
             '统一评测固定12个检查点×400题，每题一次采样：Non-thinking、256 Token、温度0.7、top_p=1、top_k=0、batch=4。统计200题全部保留；GSM200题来自官方train的reserved划分，不是官方test成绩。原100题已降为开发集；本轮最终结果不再用于继续调参。', '',
             '400道独立记录包含相关模板与技能，因此不宣称完全未知结构泛化；一训练种子、一评测采样不能覆盖跨种子不确定性。结构难度分层为basic57、medium28、complex115，GSM难度不猜测。', '',
             '## 完整评测', '', '| 模型 | 严格正确 | GSM/200 | 统计/200 | 格式有效 | 截断 |', '|---|---:|---:|---:|---:|---:|']
    for key, m in metrics.items():
        lines.append(f"| {key} | {m['strict_correct_count']}/400 ({percentage(m['sampled_pass_at_1'])}) | {m['by_source']['openai/gsm8k']['strict_correct_count']} | {m['by_source']['statreason_synthetic']['strict_correct_count']} | {percentage(m['format_valid_rate'])} | {percentage(m['truncation_rate'])} |")
    lines += ['', '严格正确、数值正确和格式有效分别记录，完整分题型、分结构难度、Wilson区间与资源见results.json、ablation_summary.csv和逐题JSONL。不能把4800条回答当作4800道独立试题；统计单位为同一400题配对。未进行pass@8评测。', '',
              '## 配对统计', '', '| 比较 | 差值（百分点） | 未调整配对Bootstrap 95%区间 | McNemar p | Holm p |', '|---|---:|---:|---:|---:|']
    for key in ['SFT_vs_Base', 'E4_step100_vs_SFT', 'E4_step100_vs_E0_step100', 'E4_step30_vs_E0_step30']:
        c = comparisons[key]
        lines.append(f"| {key} | {c['second_minus_first_accuracy']*100:+.2f} | {interval(c['paired_bootstrap_95'])} | {c['mcnemar_exact_two_sided_p']:.5g} | {c['mcnemar_holm_adjusted_p']:.5g} |")
    lines += ['', f"Bootstrap按题目配对重采样{comparisons['E4_step100_vs_SFT']['bootstrap_resamples']}次，种子42，同时保存按来源分层的区间。Holm校正覆盖{result['comparison_family_size']}项探索性比较；未调整区间为正不等于整组多重比较校正后显著。", '',
              'E4相对SFT的点估计提升来自GSM题，统计题均3/200。E4相对Vanilla的区间跨0，不能宣称DAPO显著优于GRPO。共同30步比较同样没有清晰优势。单个最高分检查点不作为新一轮训练选择依据。', '',
              '## 机制行为与成本', '', '| 组别 | 实际步数 | 尝试/接受组 | 原始有效率 | 接受组有效率 | 总生成Token | 额外补采Token |', '|---|---:|---:|---:|---:|---:|---:|']
    for key, r in result['training']['rows'].items():
        lines.append(f"| {key} | {r['optimizer_steps']} | {r['attempts']}/{r['accepted_groups']} | {percentage(r['raw_effective_attempt_rate'])} | {percentage(r['accepted_effective_group_rate'])} | {r['total_generated_tokens']} | {r['extra_resampling_tokens']} |")
    tr = result['training']['rows']
    lines += ['', f"在线筛选使E4用于训练的原始混合奖励组比例达到100%，但总生成成本为E0的{tr['E4']['total_generated_tokens']/tr['E0']['total_generated_tokens']:.2f}倍。额外补采Token与丢弃Token集合重叠，不能相加当作总成本。E2连续9组无差异而停在30步，尾部2733 Token未用于新更新，仍计入成本。", '',
              '接受组100%有效是筛选定义保证的结果，并非全部候选题的正确率。E1上侧有效裁剪仅约0.0096%，E0约0.0217%；裁剪很少，不能宣称缓解熵坍塌。E3的token聚合及梯度累积归一化通过实际TRL方法和GPU损失独立核对。E4有53个接受回答受到长度惩罚，证明该机制参与真实训练；组合效果不能单独归因于长度奖励。没有额外E5，因此缺少独立长度消融。', '',
              '熵来自各组实际采样的题目和Token，动态筛选改变样本组成；均值差异不是固定探针分布上的探索能力差异。当前日志不足以证实策略熵坍塌。相同奖励组也可能有KL更新，不能将零优势等同于所有参数梯度为零。', '',
              '## 公平比较与预算限制', '',
              '五组共同30步比较，四组100步终点单列，E2/100标为缺失。按生成Token选取预算受限检查点，不使用准确率选择。', '',
              '| 成本比较对象 | 实际Token | E0/100预算 | 未用预算差 |', '|---|---:|---:|---:|']
    for info in result['training']['cost_comparisons'].values():
        lines.append(f"| {info['model']} | {info['actual_tokens']} | {info['budget']} | {info['residual_budget']} |")
    lines += ['', '因每10步保存，成本只是同一上限约束，不是实际Token完全相等；不能据此声称严格等成本优越。所有停止与失败记录保留。时间受电脑负载、预热和温度影响，Token成本更适合主要效率比较。', '',
              '## 正确性与错误案例', '',
              f"重新严格判题全部{result['prediction_count']}条评测回答，与保存结果一致；200道统计标准答案独立计算通过，最大绝对误差{result['gold_audit']['max_absolute_recompute_error']:.3g}。这不代表所有GSM答案都经独立人工证明。", '',
              '代表性错误包括：指标变量方差没有正确代入p(1-p)；相关样本均值标准误遗漏或错算协方差；将条件独立误当成无条件独立的贝叶斯题。对应原文保存在error_cases.jsonl。高格式有效率与极低统计正确率同时出现，说明本轮领域推理缺陷没有解决；不会通过放宽容差、删除难题或继续使用测试集调参修饰结果。', '',
              '## 图表', '',
              f"汇总图09的RL Token列以及ablation_summary.csv的training_generated_tokens_at_checkpoint列均统计截至各检查点的累计生成量；成本图08与上文总成本表统计完整运行。E2第30步检查点为{result['models']['E2_step30']['training_generated_tokens']:,} Token，停止前额外补采{tr['E2']['total_generated_tokens']-result['models']['E2_step30']['training_generated_tokens']:,} Token未进入新更新，完整运行合计{tr['E2']['total_generated_tokens']:,} Token。", '']
    for path in charts:
        lines += [f'![{path.stem}](figures/{path.name})', '']
    lines += ['图中直接显示原始更新序列，没有挑选有利区间；奖励每两次更新共用一组样本。准确率图从0开始，显示完整0–100%范围和Wilson区间。E2曲线在30步结束，不延长或补造至100步。', '',
              '## 复现与公开产物', '',
              '公开仓库提供代码、配置、依赖、测试、报告、CSV、结果JSON、图表和少量错误样例。完整原始回答、训练审计、冻结计划、处理后数据及模型/适配器检查点保存在本地，未随仓库分发。阅读公开结果无需模型；以下命令仅适用于已具备完整本地实验产物的环境。重新审计不能仅依赖结果JSON，具体前置条件见[运行指南](../../docs/training_guide.md)。', '', '```powershell',
              '.venv\\Scripts\\python.exe scripts/24_run_ablations.py --status',
              '.venv\\Scripts\\python.exe scripts/26_evaluate_ablations.py --execute',
              '.venv\\Scripts\\python.exe scripts/27_finalize_ablation_report.py', '```', '',
              '报告、CSV、PNG和本地校验清单均由保存的真实运行记录生成。公共结果表与完整运行的计算成本分别注明统计口径，未对失败样本或提前停止运行作删除或补造。', '',
              '本轮结论限定为预算受限的机制分析。后续可在新增独立开发/测试数据与额外预算下研究多种子、更强数学底座、严格等Token更新和独立长度消融。当前测试集不得再用作模型选择。']
    (out / 'final_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    if hasattr(__import__('sys').stdout, 'reconfigure'):
        __import__('sys').stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only', action='store_true', help='仅快速核对原始结果，不进行Bootstrap和绘图')
    parser.add_argument('--bootstrap-resamples', type=int, default=10000)
    args = parser.parse_args()
    print('正在核对冻结配置、4800条回答和真实训练日志；不启动模型。', flush=True)
    result = audit_and_analyze(PROJECT_ROOT, resamples=args.bootstrap_resamples, audit_only=args.audit_only)
    output = PROJECT_ROOT / 'reports/ablation_round1'
    if args.audit_only:
        write_json(output / 'final_integrity_audit.json', {k: result[k] for k in ['plan_signature', 'prediction_count', 'unique_question_count', 'gold_audit', 'audit_passed']})
        print(f"核对通过：{result['prediction_count']}条回答、{result['unique_question_count']}道题；200道统计标准答案复算通过。", flush=True)
        return 0
    print('配对统计完成，正在生成11张图和报告。', flush=True)
    charts = make_figures(result, output / 'figures')
    write_report(PROJECT_ROOT, result, charts)
    write_error_cases(PROJECT_ROOT, output)
    result.update(recorded_at_utc=utc_now(), acceptance_grade='B', selected_resume_version='B',
                  acceptance_scope='bounded controlled mechanism replication; E2 stopped at 30; single seed; no claim DAPO statistically beats GRPO')
    write_json(output / 'results.json', result)
    write_json(output / 'acceptance.json', {'grade': 'B', 'route': 'mechanism_analysis', 'audit_passed': True,
               'evaluated_checkpoints': len(result['models']), 'unique_questions': result['unique_question_count'],
               'actual_rl_optimizer_steps': sum(r['optimizer_steps'] for r in result['training']['rows'].values()),
               'figures_created': len(charts), 'selected_resume_version': 'B',
               'limitations': ['E2 has 30 not 100 steps', 'one seed', 'GSM8K train-reserved not official test',
                               'cost matches leave residual budget', 'no standalone E5 length-reward ablation']})
    manifest_files = charts + [output / n for n in ['final_report.md', 'results.json', 'acceptance.json', 'ablation_summary.csv', 'paired_comparisons.csv', 'error_cases.jsonl']]
    write_json(output / 'final_artifact_manifest.json', {p.relative_to(PROJECT_ROOT).as_posix(): file_sha256(p) for p in manifest_files})
    status = read_project_status(PROJECT_ROOT)
    status.update(acceptance={'grade': 'B', 'report': 'reports/ablation_round1/final_report.md', 'resume_version': 'B'},
                  next_phase='completed_bounded_mechanism_study', final_evaluation_completed=True,
                  next_actions=[], no_further_test_tuning=True)
    write_json(PROJECT_ROOT / 'reports/project_status.json', status)
    print('汇总完成：实验报告、统计结果和11张图表已保存。', flush=True)
    print(f'报告：{output / "final_report.md"}', flush=True)
    return 0


def read_project_status(root):
    return json.loads((root / 'reports/project_status.json').read_text(encoding='utf-8'))


def write_error_cases(root, output):
    """Export representative existing SFT errors without generating career materials."""
    from src.data.storage import read_jsonl
    from src.evaluation.storage import append_prediction

    chosen = {
        'stat-test-fbf3c099c73ae7f90612': '没有正确利用指示变量平方仍为自身；最终数值错误，推导不足以细分所有原因。',
        'stat-test-a3c0f37f5da79b93b45e': '读题与建模错误，相关样本均值的协方差项处理错误，兼有数值问题。',
        'stat-test-b742f81b1c4b1ab5524d': '中间矩与最终减法均有错误；不能归为单纯格式问题。',
        'stat-test-19b72a3ee065e8a467ce': '把后验概率写成事件交与事件并的比值，属于公式/建模错误。',
    }
    rows = read_jsonl(root / 'outputs/ablation_evaluation_round1/SFT/evaluation_predictions.jsonl')
    case_path = output / 'error_cases.jsonl'
    case_path.write_text('', encoding='utf-8')
    for row in rows:
        if row['id'] in chosen:
            append_prediction(case_path, {**row, 'qualitative_review': chosen[row['id']],
                                         'scope': 'representative_SFT_examples_not_exhaustive_semantic_counts'})


if __name__ == '__main__':
    raise SystemExit(main())
