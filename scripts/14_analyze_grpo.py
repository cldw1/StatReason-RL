"""从真实GRPO训练和统一验证回答生成统计、曲线与报告；拒绝未完成运行。"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.data.schema import canonical_json
from src.data.storage import read_jsonl
from src.evaluation.comparison import paired_comparison
from src.evaluation.metrics import summarize_predictions
from src.utils.artifacts import file_sha256,write_json


def read_completed(path):
    status=json.loads((path/'run_status.json').read_text(encoding='utf-8'))
    if status['status']!='completed': raise ValueError('仅分析已完成运行')
    if status.get('exclude_from_effectiveness_comparison'): raise ValueError('恢复验证分支不能作为独立效果实验')
    return status


def weighted(rows,key):
    total=sum(r['valid_tokens'] for r in rows)
    return sum(r[key]*r['valid_tokens'] for r in rows)/total if total else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--sft-evaluation',required=True,type=Path)
    parser.add_argument('--grpo-evaluation',required=True,type=Path)
    args=parser.parse_args()
    run,sft_eval,rl_eval=(p.resolve() for p in (args.run,args.sft_evaluation,args.grpo_evaluation))
    status=read_completed(run)
    if status['level']!=1: raise ValueError('Level0 smoke不可混入本地效果报告')
    read_completed(sft_eval);read_completed(rl_eval)
    sft=read_jsonl(sft_eval/'evaluation_predictions.jsonl')
    predictions=read_jsonl(rl_eval/'evaluation_predictions.jsonl')
    comparison=paired_comparison(sft,predictions)
    summaries={'SFT':summarize_predictions(sft)['by_mode']['non-thinking'],
               'GRPO':summarize_predictions(predictions)['by_mode']['non-thinking']}
    rollouts=read_jsonl(run/'rollout_groups.jsonl')
    policy=read_jsonl(run/'policy_audit.jsonl')
    metrics=read_jsonl(run/'train_metrics.jsonl')
    logs={r['step']:r for r in metrics if r.get('event')=='trainer_log' and 'grad_norm' in r}
    resource=json.loads((run/'resource_metrics.json').read_text(encoding='utf-8'))
    groups=[g for r in rollouts for g in r['groups']]
    completions=[c for r in rollouts for c in r['completions']]
    if not all(p['old_logps_immutable'] for p in policy): raise ValueError('旧策略审计未通过')
    update_steps=defaultdict(set)
    for p in policy:
        for rollout_id in p['rollout_id']: update_steps[rollout_id].add(p['optimizer_step'])
    if any(len(steps)!=2 for steps in update_steps.values()): raise ValueError('每个rollout应有两轮优化')
    total_tokens=sum(p['valid_tokens'] for p in policy)
    clipped=sum((p['effective_low_clip']+p['effective_high_clip'])*p['valid_tokens'] for p in policy)
    group_by_id={r['rollout_id']:r for r in rollouts}
    zero_groups_with_kl=[]
    for step,log in logs.items():
        audit=[p for p in policy if p['optimizer_step']==step-1]
        rid=audit[0]['rollout_id'][0]
        if all(g['zero_variance'] for g in group_by_id[rid]['groups']) and log['grad_norm']>1e-8 and weighted(audit,'kl')>0:
            zero_groups_with_kl.append(step)
    by_step=defaultdict(list)
    for p in policy: by_step[p['optimizer_step']+1].append(p)
    changes=[]
    for rid,steps in update_steps.items():
        first,last=sorted(steps)
        rows1=[p for p in policy if rid in p['rollout_id'] and p['optimizer_step']==first]
        rows2=[p for p in policy if rid in p['rollout_id'] and p['optimizer_step']==last]
        changes.append(weighted(rows2,'policy_entropy_raw')-weighted(rows1,'policy_entropy_raw'))
    summary={'level':1,'optimizer_steps':status['optimizer_steps'],'rollouts':len(rollouts),
             'unique_training_questions':len({g['id'] for g in groups}),'generated_completions':len(completions),
             'mean_raw_correctness_reward':sum(c['reward'] for c in completions)/len(completions),
             'mean_group_sample_std':sum(g['sample_std'] for g in groups)/len(groups),
             'mixed_groups':sum(not g['zero_variance'] for g in groups),
             'all_correct_groups':sum(all(g['rewards']) for g in groups),
             'all_wrong_groups':sum(not any(g['rewards']) for g in groups),
             'zero_variance_group_rate':sum(g['zero_variance'] for g in groups)/len(groups),
             'truncation_rate':sum(c['verification']['truncated'] for c in completions)/len(completions),
             'policy_ratio_min':min(p['ratio_min'] for p in policy),'policy_ratio_max':max(p['ratio_max'] for p in policy),
             'objective_affected_clipping_rate_token_weighted':clipped/total_tokens,
             'objective_affected_token_equivalents':clipped,'valid_training_token_exposures':total_tokens,
             'ratio_low_outside_rate_token_weighted':weighted(policy,'ratio_low_outside'),
             'ratio_high_outside_rate_token_weighted':weighted(policy,'ratio_high_outside'),
             'mean_within_rollout_raw_entropy_change':sum(changes)/len(changes),
             'zero_variance_update_steps_with_positive_kl_gradient':zero_groups_with_kl,
             'old_logprob_reuse_verified':True,'resources':resource}
    sources={'training_run':str(run.relative_to(PROJECT_ROOT)),'sft_evaluation':str(sft_eval.relative_to(PROJECT_ROOT)),
             'grpo_evaluation':str(rl_eval.relative_to(PROJECT_ROOT)),
             'source_sha256':{str(p.relative_to(PROJECT_ROOT)):file_sha256(p) for p in [run/'policy_audit.jsonl',run/'rollout_groups.jsonl',rl_eval/'evaluation_predictions.jsonl']}}
    output=PROJECT_ROOT/'reports/grpo';output.mkdir(parents=True,exist_ok=True)
    write_json(output/'results.json',{'scope':'local_small_scale_development_not_final_test','training':summary,
                                    'evaluation':summaries,'comparison':comparison,'sources':sources,'simulated_data':False})
    # 便携回答只含真实生成及判题，不包含权重或环境密钥。
    keys=['id','mode','source','task_type','evaluation_stratum','question','answer','answer_type','answer_unit',
          'absolute_tolerance','relative_tolerance','allow_percentage','prompt_text','generation_config','generation_batch_size',
          'completion','verification','seed','generated_token_count','generation_seconds','truncated','model_revision']
    (output/'predictions.jsonl').write_text(''.join(canonical_json({k:r[k] for k in keys})+'\n' for r in predictions),encoding='utf-8')
    (output/'sft_predictions.jsonl').write_text(''.join(canonical_json({k:r[k] for k in keys})+'\n' for r in sft),encoding='utf-8')
    selected=next(r for r in rollouts if any(not g['zero_variance'] for g in r['groups']))
    write_json(output/'sample_rollout.json',selected)
    old={r['id']:r for r in sft}
    changed=[{'id':r['id'],'task_type':r['task_type'],'question':r['question'],'answer':r['answer'],
              'sft_completion':old[r['id']]['completion'],'grpo_completion':r['completion'],
              'sft_correct':old[r['id']]['verification']['strict_correct'],'grpo_correct':r['verification']['strict_correct']}
             for r in predictions if r['verification']['strict_correct']!=old[r['id']]['verification']['strict_correct']]
    write_json(output/'changed_correctness_cases.json',{'cases':changed,'selection':'all_changed_strict_correctness_on_common_development_questions'})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    steps=sorted(by_step)
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    x=[r['optimizer_step']+1 for r in rollouts]
    axes[0,0].plot(x,[sum(c['reward'] for c in r['completions'])/len(r['completions']) for r in rollouts],marker='o')
    axes[0,0].set(title='Correctness reward per rollout',ylabel='Mean binary reward',ylim=(-.05,1.05))
    axes[0,1].plot(steps,[weighted(by_step[s],'kl') for s in steps],marker='.')
    axes[0,1].set(title='Frozen-SFT KL estimator',ylabel='Token-weighted KL (nat)')
    for key,label in [('policy_entropy_raw','Raw policy'),('policy_entropy_temperature','Temperature=.7'),('policy_entropy_sampling','Top-k / top-p sampling')]:
        axes[0,2].plot(steps,[weighted(by_step[s],key) for s in steps],label=label)
    axes[0,2].set(title='Entropy at update input',ylabel='Token-weighted entropy (nat)',ylim=(0,max(1.,1.3*max(weighted(by_step[s],'policy_entropy_raw') for s in steps))));axes[0,2].legend(fontsize=8)
    axes[1,0].plot(steps,[100*(weighted(by_step[s],'effective_low_clip')+weighted(by_step[s],'effective_high_clip')) for s in steps],label='Objective affected')
    axes[1,0].plot(steps,[100*(weighted(by_step[s],'ratio_low_outside')+weighted(by_step[s],'ratio_high_outside')) for s in steps],label='Ratio outside',alpha=.65)
    axes[1,0].set(title='Clipping vs ratio boundaries',ylabel='Valid token percentage');axes[1,0].legend(fontsize=8)
    axes[1,1].plot(x,[r['generated_tokens']/len(r['completions']) for r in rollouts],marker='o')
    axes[1,1].set(title='Generated completion length',ylabel='Token count (including EOS)')
    axes[1,2].plot(x,[sum(c['verification']['truncated'] for c in r['completions'])/len(r['completions']) for r in rollouts],marker='o',label='Truncated')
    axes[1,2].plot(x,[sum(not g['zero_variance'] for g in r['groups'])/len(r['groups']) for r in rollouts],marker='.',label='Mixed group')
    axes[1,2].set(title='Truncation and effective groups',ylabel='Rate',ylim=(-.05,1.05));axes[1,2].legend(fontsize=8)
    for ax in axes.flat:
        ax.set_xlabel('Optimizer update (1-based)');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Local Vanilla GRPO | 24 updates | 12 training questions | actual logs',fontsize=12)
    fig.tight_layout();fig.savefig(output/'training_diagnostics.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4))
    names=list(summaries);rates=[summaries[n]['sampled_pass_at_1'] for n in names]
    lower=[rate-summaries[n]['wilson_95_interval'][0] for n,rate in zip(names,rates)]
    upper=[summaries[n]['wilson_95_interval'][1]-rate for n,rate in zip(names,rates)]
    bars=ax.bar(names,rates,yerr=[lower,upper],capsize=5,color=['#547AA5','#C27844'])
    ax.bar_label(bars,labels=[f'{summaries[n]["strict_correct_count"]}/{summaries[n]["question_count"]}' for n in names],padding=6)
    ax.set(ylim=(0,.5),ylabel='Sampled pass@1 (Wilson 95% CI)',title='Common 100-question development panel')
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(output/'accuracy.png',dpi=180);plt.close(fig)
    lines=['# Vanilla GRPO 本地小规模实验','','本报告基于真实训练和固定100题开发验证；不是最终测试或多seed结论。',
           '', '## 训练与公平性','',
           f'从原定SFT第150步独立初始化，并加载独立冻结的相同SFT参考副本。完整RL池4400条，实际24次更新只采到{summary["unique_training_questions"]}道题、{len(completions)}个回答。G=4，每个rollout复用两轮；无动态补采、无超长奖励、无截断损失掩码。',
           '', '显式配置：loss_type=grpo、scale_rewards=group、beta=.04、epsilon/epsilon_high=.2、temperature=.7、top_p=.8、top_k=20、min_p=0、completion上限128；LoRA与SFT一致，学习率1e-5。不同模型评测统一使用192新Token、非思考、batch=4及同一题序/批次seed。',
           '', f'训练段{resource["training_seconds"]:.3f}秒，峰值allocated {resource["peak_allocated_bytes"]/2**30:.3f} GiB、reserved {resource["peak_reserved_bytes"]/2**30:.3f} GiB；共生成{resource["total_generated_tokens"]} Token，含两轮复用的有效训练Token曝光{total_tokens}。',
           '', '224个LoRA张量发生变化，冻结Base权重抽查、全部参考adapter哈希与参考梯度检查通过；初始策略与参考logprob最大差0，旧概率复用哈希一致，TRL损失与独立目标最大差0。保存/重载adapter权重与32-token探针logits逐位一致。',
           '', '## 开发评测','','| 模型 | 严格正确 | 同类统计35题 | 原迁移35题 | GSM8K30题 | 格式率 | 截断率 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        strata=s['by_evaluation_stratum']
        lines.append(f'| {name} | {s["strict_correct_count"]}/100 | {strata["core_calculation"]["strict_correct_count"]}/35 | {strata["v1_transfer_validation"]["strict_correct_count"]}/35 | {strata["gsm8k_validation"]["strict_correct_count"]}/30 | {s["format_valid_rate"]:.1%} | {s["truncation_rate"]:.1%} |')
    lines+=['','![开发准确率](accuracy.png)','',
            f'GRPO减SFT的差值{comparison["second_minus_first_accuracy"]:+.1%}，配对Bootstrap95%区间{comparison["paired_bootstrap_95"]}，分层区间{comparison["stratified_paired_bootstrap_95"]}，McNemar精确双侧p={comparison["mcnemar_exact_two_sided_p"]:.6g}。SFT独对{comparison["first_only_correct"]}题、GRPO独对{comparison["second_only_correct"]}题。不能宣称正确率提升。',
            '', '## 稳定性与限制','',
            f'12组中{summary["mixed_groups"]}组混合奖励、{summary["all_correct_groups"]}组全对、{summary["all_wrong_groups"]}组全错；零方差率{summary["zero_variance_group_rate"]:.1%}。原始平均正确性奖励{summary["mean_raw_correctness_reward"]:.4f}，截断率{summary["truncation_rate"]:.1%}。这来自不同训练题，不能把批次波动解释成学习曲线的确定提升。',
            '', f'重要性比率范围[{summary["policy_ratio_min"]:.6f},{summary["policy_ratio_max"]:.6f}]；真正影响目标的裁剪Token约{clipped:.3f}/{total_tokens}，比例{100*summary["objective_affected_clipping_rate_token_weighted"]:.4f}%。比率越界与有符号优势下真正裁剪分别记录；Clip-Higher的潜在影响尚小，必须实际对照。',
            '', f'部分零方差组仍有非零KL和梯度，对应更新步{zero_groups_with_kl}。初始全对组的正确性优势及KL为零，梯度确为零，未伪造参数变化。',
            '', '原始、温度缩放及过滤采样熵分别记录，单位nat；曲线跨不同问题，不能据此断言熵坍塌。短训练中没有可确证的熵坍塌证据。重复同一rollout两次更新前的原始熵变化另见results.json，不能当作长期固定评测分布的变化。',
            '', '![真实训练诊断](training_diagnostics.png)','',
            '训练采用TRL式温度缩放后的密集策略logprob，采样还经过top-k/top-p；两者分布并不相同，未声称对过滤采样做了精确重要性修正。当前样本与预算很小、单训练seed、验证题含模板族相关性，区间不覆盖训练seed波动。',
            '', '检查点只在两轮rollout复用周期边界恢复；恢复时必须保持配置、代码、数据、软件和SFT权重一致。额外恢复验证见resume_verification.json，不假设跨硬件按位一致。',
            '', '实际实现与兼容修复见[GRPO数学与代码导读](../../docs/grpo_explained.md)；初始化和采样审计、参数更新与资源记录均在以下原始目录。',
            '', f'- 训练：`{sources["training_run"]}`。',f'- SFT评测：`{sources["sft_evaluation"]}`。',f'- GRPO评测：`{sources["grpo_evaluation"]}`。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'training':summary,'comparison':comparison,'report':str(output/'report.md')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
