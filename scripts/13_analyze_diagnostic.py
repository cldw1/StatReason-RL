"""仅分析三个已完成真实运行；开发诊断数据不能用作最终测试结论。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.data.storage import read_jsonl
from src.evaluation.comparison import paired_comparison
from src.utils.artifacts import file_sha256, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run',required=True,type=Path)
    parser.add_argument('--old-sft-run',required=True,type=Path)
    parser.add_argument('--sft-run',required=True,type=Path)
    args=parser.parse_args()
    runs={'Base':args.base_run.resolve(),'SFT V1':args.old_sft_run.resolve(),'SFT revised':args.sft_run.resolve()}
    summaries,predictions,sources={},{},{}
    for name,run in runs.items():
        status=json.loads((run/'run_status.json').read_text(encoding='utf-8'))
        if status['status']!='completed': raise ValueError('拒绝将未完成评测作为完整对照')
        summary=json.loads((run/'evaluation_summary.json').read_text(encoding='utf-8'))
        if set(summary['by_mode'])!={'non-thinking'}: raise ValueError('本诊断比较固定非思考模式')
        summaries[name]=summary['by_mode']['non-thinking']
        predictions[name]=read_jsonl(run/'evaluation_predictions.jsonl')
        sources[name]={'run':str(run.relative_to(PROJECT_ROOT)),
                       'prediction_sha256':file_sha256(run/'evaluation_predictions.jsonl'),
                       'selection':json.loads((run/'selection.json').read_text(encoding='utf-8'))}
    if any(s['question_count']!=100 for s in summaries.values()): raise ValueError('此诊断要求固定100题全部完成')
    comparison={'revised_vs_v1':paired_comparison(predictions['SFT V1'],predictions['SFT revised']),
                'revised_vs_base':paired_comparison(predictions['Base'],predictions['SFT revised']),
                'v1_vs_base':paired_comparison(predictions['Base'],predictions['SFT V1'])}
    output=PROJECT_ROOT/'reports/diagnostic'
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'comparison.json',{'scope':'development_diagnostic_not_final_test','summaries':summaries,
                                        'paired_comparisons':comparison,'sources':sources,'simulated_data':False})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    strata=['core_calculation','v1_transfer_validation','gsm8k_validation']
    fig,ax=plt.subplots(figsize=(9,4.8))
    x=np.arange(len(strata))
    width=.24
    for i,(name,s) in enumerate(summaries.items()):
        cells=[s['by_evaluation_stratum'][key] for key in strata]
        means=np.array([cell['sampled_pass_at_1'] for cell in cells])
        errors=np.array([[max(0,m-cell['wilson_95_interval'][0]) for m,cell in zip(means,cells)],
                         [max(0,cell['wilson_95_interval'][1]-m) for m,cell in zip(means,cells)]])
        bars=ax.bar(x+(i-1)*width,means,width,label=name,yerr=errors,capsize=3)
        ax.bar_label(bars,labels=[f'{cell["strict_correct_count"]}/{cell["question_count"]}' for cell in cells],padding=5,fontsize=8)
    ax.set_xticks(x,['Core calculations (n=35)','V1 transfer panel (n=35)','GSM8K validation (n=30)'])
    ax.set_ylim(0,1.08)
    ax.set_ylabel('Sampled pass@1 (Wilson 95% CI)')
    ax.set_title('Development diagnosis | fixed non-thinking batch=4 | one generation per question')
    ax.legend()
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'accuracy_by_stratum.png',dpi=180)
    plt.close(fig)
    lines=['# SFT 基线修订开发诊断','','本报告仅基于固定的100道验证题。没有使用官方或自建测试集，也不能代表最终测试正确率。',
           '', '| 模型 | 全部100题 | 同类计算35题 | V1迁移题35题 | GSM8K 30题 | 格式有效率 | 截断率 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,s in summaries.items():
        values=[str(s['by_evaluation_stratum'][key]['strict_correct_count'])+'/'+str(s['by_evaluation_stratum'][key]['question_count']) for key in strata]
        lines.append(f'| {name} | {s["strict_correct_count"]}/100 | '+ ' | '.join(values)+f' | {s["format_valid_rate"]:.1%} | {s["truncation_rate"]:.1%} |')
    lines+=['','![实际开发诊断结果](accuracy_by_stratum.png)','','## 同题统计','','差值为修订模型减比较模型；重采样单位为同一道题，分层 Bootstrap 保持35/35/30配比。']
    for name,comp in comparison.items():
        lines.append(f'\n- {name}：差值 {comp["second_minus_first_accuracy"]:+.1%}；配对 Bootstrap 95% 区间 {comp["paired_bootstrap_95"]}；分层区间 {comp["stratified_paired_bootstrap_95"]}；McNemar 精确双侧 p={comp["mcnemar_exact_two_sided_p"]:.6g}。')
    lines+=['','## 解释范围','',
            'V2 SFT 同时改变统计/GSM8K配比（560/560）、数学技能覆盖（variant 0和2）及确定性参考步骤/分数答案。这是基线修订，不是三项因素的独立因果消融。V1 迁移题对 V2 已不是未见数学技能，只是保留原有输入模板；同类计算题也只是在输入表示上留出，未证明排除了所有语义近重复。',
            '', '验证面板、题序、提示词、batch=4、批次种子及采样参数在三组中固定；与先前 batch=1 的8题结果不直接合并。每题仅一个随机生成，因此这些区间不包含训练随机种子波动，也没有估计多次解码波动。McNemar 为探索性比较，未作多重检验校正。',
            '', 'SFT 检查点由预先固定的64题验证 Token loss 选择，不能把 Token loss 下降直接解释为数值答案正确率提升。原测试集继续封存。',
            '', '逐题回答、校验、耗时、显存和实际生成 Token 均见下列真实运行。']
    for name,source in sources.items(): lines+=['',f'- {name}：`{source["run"]}`。']
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'summaries':{name:{k:s[k] for k in ['strict_correct_count','format_valid_rate','truncation_rate']} for name,s in summaries.items()},
                      'comparisons':comparison,'report':str(output/'report.md')},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
