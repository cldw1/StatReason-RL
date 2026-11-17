"""同题配对 Bootstrap 与精确 McNemar 检验；统计单位始终为题目。"""
from __future__ import annotations

from collections import defaultdict
import numpy as np
from scipy.stats import binomtest


def paired_comparison(first: list[dict], second: list[dict], *, seed: int=42, resamples: int=10000) -> dict:
    if resamples < 100:
        raise ValueError('Bootstrap 次数至少为 100')
    def keyed(rows):
        result={}
        for row in rows:
            key=(row['mode'],row['id'])
            if key in result: raise ValueError('每题每模式只能一个回答')
            result[key]=row
        return result
    a,b=keyed(first),keyed(second)
    if not a or set(a)!=set(b) or len({k[0] for k in a})!=1:
        raise ValueError('配对要求相同非空题组，且只有一个生成模式')
    # 允许不同 adapter，但题面、容差、生成参数和批次种子必须一致。
    for key in a:
        for field in ('question','answer','absolute_tolerance','relative_tolerance','prompt_text','seed','generation_config','generation_batch_size'):
            if a[key].get(field)!=b[key].get(field): raise ValueError(f'配对协议不同: {field}')
    keys=sorted(a)
    x=np.array([a[k]['verification']['strict_correct'] for k in keys],dtype=int)
    y=np.array([b[k]['verification']['strict_correct'] for k in keys],dtype=int)
    delta=y-x
    rng=np.random.default_rng(seed)
    draws=delta[rng.integers(0,len(keys),size=(resamples,len(keys)))].mean(axis=1)
    strata=defaultdict(list)
    for index,key in enumerate(keys):
        s=a[key].get('evaluation_stratum','unspecified')
        if s!=b[key].get('evaluation_stratum','unspecified'): raise ValueError('分层标签不一致')
        strata[s].append(index)
    stratified=np.zeros(resamples)
    for indices in strata.values():
        values=delta[indices]
        stratified+=values[rng.integers(0,len(indices),size=(resamples,len(indices)))].sum(axis=1)/len(keys)
    losses=int(((x==1)&(y==0)).sum())
    gains=int(((x==0)&(y==1)).sum())
    return {'statistical_unit':'question_paired_within_one_mode','question_count':len(keys),
            'first_correct':int(x.sum()),'second_correct':int(y.sum()),'second_minus_first_accuracy':float(delta.mean()),
            'paired_bootstrap_95':np.quantile(draws,[.025,.975]).tolist(),
            'stratified_paired_bootstrap_95':np.quantile(stratified,[.025,.975]).tolist(),
            'bootstrap_resamples':resamples,'bootstrap_seed':seed,'first_only_correct':losses,'second_only_correct':gains,
            'mcnemar_exact_two_sided_p':float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.,
            'scope':'development_diagnostic_not_final_test_or_training_seed_variation',
            'multiplicity':'unadjusted_exploratory_comparisons',
            'stratum_counts':{s:len(indices) for s,indices in strata.items()}}
