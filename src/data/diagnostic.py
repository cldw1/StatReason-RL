"""V2 开发诊断：同类计算与 V1 迁移题分层；绝不读取测试题正文。"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import random

from .quality import audit_statistics
from .schema import canonical_json, validate_records
from .statistics import TASK_TYPES, _make_task
from .storage import read_jsonl, save_dataset
from src.utils.artifacts import file_sha256, write_json


def parameter_digest(task: str, parameters: dict) -> str:
    return hashlib.sha256((task + ":" + canonical_json(parameters)).encode()).hexdigest()[:20]


def worked_solution(task: str, p: dict) -> tuple[str, str | None]:
    """所有代入量由确定性运算生成；有理数保留精确分数，不能依赖 LLM 答案。"""
    v = p['variant']
    exact = None
    if task == 'conditional_probability':
        a, b = (p['joint_count'], p['b_count']) if v == 0 else (p['counts_ab_notab_anotb_neither'][0], sum(p['counts_ab_notab_anotb_neither'][:2]))
        exact = F(a, b)
        text = f'条件空间只包含 B。P(A|B)=P(A∩B)/P(B)={a}/{b}={exact}。'
    elif task == 'bayes':
        if v == 0:
            prior, la, lb = (F(str(p[k])) for k in ('prior', 'likelihood_a', 'likelihood_b'))
            joint, other = prior * la, (1-prior) * lb
            exact = joint / (joint+other)
            text = f'P(A∩E)={prior}×{la}={joint}；P(B∩E)=(1-{prior})×{lb}={other}。P(A|E)={joint}/({joint}+{other})={exact}。'
        else:
            a, b, c, d = (p[k] for k in ('odds_numerator','odds_denominator','likelihood_ratio_numerator','likelihood_ratio_denominator'))
            odds = F(a*c, b*d)
            exact = odds/(1+odds)
            text = f'先验赔率={a}/{b}，后验赔率=({a}/{b})×({c}/{d})={odds}。后验概率={odds}/(1+{odds})={exact}。'
    elif task == 'binomial_probability':
        n, k, prob = p['n'], p['k'], F(str(p['p']))
        point = lambda j: math.comb(n,j)*prob**j*(1-prob)**(n-j)
        if v == 0:
            exact = point(k)
            text = f'P(X={k})=C({n},{k})×{prob}^{k}×(1-{prob})^{n-k}。组合数={math.comb(n,k)}，两项幂分别为 {float(prob**k):.10f}、{float((1-prob)**(n-k)):.10f}；乘积={float(exact):.10f}。'
        else:
            cumulative = sum((point(j) for j in range(k)), F(0))
            exact = 1-cumulative
            text = f'用补事件：P(X≥{k})=1-P(X≤{k-1})。P(X≤{k-1})=Σ(j=0..{k-1}) C({n},j)({prob})^j(1-{prob})^({n}-j)={float(cumulative):.10f}。所求=1-{float(cumulative):.10f}={float(exact):.10f}。'
        # 较大的精确分母会增加无用 Token；最终答案保持原来八位小数规则。
        exact = None
    elif task in ('discrete_expectation', 'discrete_variance'):
        values, weights = p['values'], p['weights']
        total = sum(weights)
        first = sum(x*w for x,w in zip(values,weights,strict=True))
        second = sum(x*x*w for x,w in zip(values,weights,strict=True))
        mean = F(first,total)
        products = '+'.join(f'({x})×{w}' for x,w in zip(values,weights,strict=True))
        text = f'E(X)=({products})/{total}={first}/{total}={mean}。'
        if task == 'discrete_expectation':
            exact = mean
            if v == 2:
                a,b = p['coefficient'],p['intercept']
                exact = a*mean+b
                text += f'E(Y)={a}×({mean})+({b})={exact}。'
        else:
            variance = F(second,total)-mean**2
            text += f'E(X²)={second}/{total}。Var(X)={second}/{total}-({mean})²={variance}。'
            exact = variance
            if v == 2:
                a=p['coefficient']
                exact=a*a*variance
                text += f'Var(aX+b)=a²Var(X)，所以 Var(Y)={a*a}×({variance})={exact}。'
    elif task == 'standard_error':
        n,sigma=p['n'],p['sigma']
        a=p.get('coefficient',1)
        variance=a*a*sigma*sigma/n
        text=f'独立样本均值的方差=总体方差/n。变换后总体方差=({a})²×({sigma})²，均值方差={variance:.10f}；标准误=sqrt({variance:.10f})={math.sqrt(variance):.10f}。平移不改变标准差。'
    elif task == 'normal_mean_ci_endpoint':
        se=math.sqrt(p['sigma']**2/p['n']+(p['sigma2']**2/p['n2'] if v==2 else 0))
        center=p['mean']-p.get('mean2',0)
        margin=p['critical_value']*se
        text=f'已知总体方差，使用题给 z={p["critical_value"]:.10f}。中心={center:.10f}；标准误={se:.10f}；误差限=z×SE={margin:.10f}。双侧区间下限=中心-误差限={center-margin:.10f}。'
    else:
        raise ValueError(task)
    return text, str(exact) if exact is not None else None


def rewrite_question(task: str, p: dict, original: str, split: str) -> str:
    """划分的是输入表示模板，数学知识允许重合；并不声称所有语义近重复已消除。"""
    v=p['variant']
    if split == 'sft' and v == 0:
        return original
    if task == 'conditional_probability':
        if v==2: a,b=p['counts_ab_notab_anotb_neither'][:2]; b+=a
        else: a,b=p['joint_count'],p['b_count']
        total=b+max(11,b//2)
        return f'事件概率为 P(A∩B)={a}/{total}，P(B)={b}/{total}。求 P(A|B)。'
    if task == 'bayes':
        if v==0:
            prior=F(str(p['prior']))
            return f'A 与 B 互斥且完备，先验赔率 P(A):P(B)={prior.numerator}:{prior.denominator-prior.numerator}。P(E|A)={p["likelihood_a"]}，P(E|B)={p["likelihood_b"]}。求 P(A|E)。'
        a,b=p['odds_numerator'],p['odds_denominator']
        return f'P(H)={a}/{a+b}，H 与非H互斥且完备。观测 E 的似然比 P(E|H)/P(E|非H)={p["likelihood_ratio_numerator"]}/{p["likelihood_ratio_denominator"]}。求 P(H|E)。'
    if task == 'binomial_probability':
        return f'{p["n"]} 次相互独立的试验，每次只有成功或失败，失败概率为 {1-p["p"]:.1f}。令 X 为成功次数。求 P(X{"=" if v==0 else "≥"}{p["k"]})。'
    if task in ('discrete_expectation','discrete_variance'):
        description=f'一个有限总体的不同数值为 {p["values"]}，对应频数为 {p["weights"]}。等概率抽一个体，记其数值为 X。'
        if v==2: description+=f'令 Y=({p["intercept"]})+({p["coefficient"]})X。'
        return description+f'求{"期望" if task=="discrete_expectation" else "总体方差（不作样本无偏修正）"} {"X" if v==0 else "Y"} 的数值。'
    if task=='standard_error':
        return f'有 {p["n"]} 个独立同分布观测，总体标准差为 {p["sigma"]}。'+(f'先将每个观测变换为 Y=({p["intercept"]})+({p["coefficient"]})X，再取平均。' if v==2 else '直接取所有观测的平均。')+'求该平均值的标准误。'
    if task=='normal_mean_ci_endpoint':
        prefix=f'来自正态总体的独立样本，第一组样本量 {p["n"]}，观测总和 {float(F(str(p["mean"]))*p["n"]):.1f}，已知总体方差 {p["sigma"]**2:.2f}。'
        if v==2:
            prefix+=f'第二组与第一组独立，样本量 {p["n2"]}，均值 {p["mean2"]}，已知总体方差 {p["sigma2"]**2:.2f}。求第一组减第二组的总体均值差'
        else: prefix+='求总体均值'
        return prefix+f'的双侧 {p["confidence"]:.0%} 置信区间下限，使用 z={p["critical_value"]:.10f}。'
    raise ValueError(task)


def generate_diagnostic_statistics(v1: Path, seed: int = 20261116, per_training_variant: int = 40,
                                  per_validation_task: int = 5, *, notation_revision: int = 2) -> tuple[list[dict],dict]:
    if notation_revision not in (1,2): raise ValueError('未知教师文本修订号')
    split_ids=json.loads((v1/'split_ids.json').read_text(encoding='utf-8'))
    manifest=json.loads((v1/'manifest.json').read_text(encoding='utf-8'))
    if file_sha256(v1/'split_ids.json')!=manifest['file_sha256']['split_ids.json']:
        raise ValueError('旧数据划分摘要哈希不符')
    forbidden={item.rsplit('-',1)[-1] for ids in split_ids.values() for item in ids}
    used=set(forbidden)
    rows=[]
    for split,variants,count in [('sft',(0,2),per_training_variant),('validation',(0,),per_validation_task)]:
        for task in TASK_TYPES:
            for variant in variants:
                rng=random.Random(f'diagnostic-v2:{seed}:{split}:{task}:{variant}')
                for _ in range(count):
                    for attempt in range(10000):
                        p,q,_,answer,unit=_make_task(task,variant,rng)
                        digest=parameter_digest(task,p)
                        if digest not in used and (unit!='probability' or 1e-5<answer<1-1e-5): break
                    else: raise RuntimeError('唯一参数空间不足')
                    used.add(digest)
                    text,exact=worked_solution(task,p)
                    if notation_revision==1 and task=='standard_error':
                        # 仅复现已冻结旧评测版本；该文本不能用来启动新的训练。
                        a,sigma=p.get('coefficient',1),p['sigma']
                        text=text.replace(f'({a})²×({sigma})²',f'{a}²×{sigma}²')
                    final=exact if exact is not None else f'{answer:.8f}'.rstrip('0').rstrip('.')
                    q=rewrite_question(task,p,q,split)+' 最终答案至少保留8位小数，精确分数也可。'
                    if unit=='probability': q+='概率可写为0到1的小数，或带%符号的百分数。'
                    rows.append({'id':f'stat-v2-{split}-{digest}','source':'statreason_synthetic','task_type':task,
                                 'template_family':f'{task}/v2_{split}_representation_{variant}',
                                 'question':q,'parameters':p,'solution':text+f'\nFINAL_ANSWER: {final}',
                                 'answer':format(answer,'.15g'),'answer_type':'scalar','answer_unit':unit,
                                 'absolute_tolerance':1e-8 if unit=='probability' else 1e-6,'relative_tolerance':1e-8,
                                 'allow_percentage':unit=='probability','split':split,
                                 'evaluation_stratum':'core_calculation' if split=='validation' else 'training'})
    # 只复用旧 validation / RL 正文，测试集仅使用已保存 ID 的不可逆摘要排重。
    old_validation=_checked_split(v1,'validation')
    rng=random.Random(seed)
    for task in TASK_TYPES:
        pool=sorted((r for r in old_validation if r['task_type']==task),key=lambda r:r['id'])
        rows.extend({**r,'evaluation_stratum':'v1_transfer_validation'} for r in rng.sample(pool,per_validation_task))
    rows.extend(_checked_split(v1,'rl'))
    validate_records(rows)
    audit=audit_statistics(rows)
    audit.update(v1_test_body_accessed=False, v1_parameter_digest_exclusions=len(forbidden),
                 new_sft_parameter_overlap_with_v1=0, split_policy='held_out_input_representations_with_shared_mathematical_skills',
                 semantic_near_duplicate_exclusion='not_claimed; representation families and exact parameters audited')
    return rows,audit


def _checked_split(directory: Path, split: str) -> list[dict]:
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    path=directory/f'{split}.jsonl'
    if file_sha256(path)!=manifest['file_sha256'][path.name]: raise ValueError('源文件哈希不符')
    return read_jsonl(path)


def prepare_diagnostic_release(project: Path, *, notation_revision: int = 2) -> dict:
    v1=project/'data/processed/statreason-v1'
    version='statreason-diagnostic-v2_1' if notation_revision==2 else 'statreason-diagnostic-v2'
    v2=project/'data/processed'/version
    rows,audit=generate_diagnostic_statistics(v1/'statistics_full',notation_revision=notation_revision)
    stats=save_dataset(v2/'statistics_full',rows,{'version':version,'seed':20261116,
                       'training_allowed':notation_revision==2,
                       'purpose':'development_diagnosis_not_final_test','audit':audit,
                       'teacher_change':'deterministic_substitution_steps_and_exact_rational_answers',
                       'confounding':'coverage_mixture_and_teacher_changed_together_not_single_factor_ablation'})
    gsm_directory=v1/'gsm8k'
    train=_checked_split(gsm_directory,'sft')
    rng=random.Random(20261116)
    chosen=rng.sample(sorted(train,key=lambda r:r['id']),560)
    validation=[{**r,'evaluation_stratum':'gsm8k_validation'} for r in _checked_split(gsm_directory,'validation')]
    gsm=save_dataset(v2/'gsm8k',chosen+validation+_checked_split(gsm_directory,'rl'),
                     {'source_release':'statreason-v1','sft_subset_seed':20261116,'official_test_accessed':False})
    summary={'statistics':stats['statistics'],'gsm8k':gsm['statistics'],'audit':audit,
             'sft_mixture':{'statistics':560,'gsm8k':560},'development_panel':{'core':35,'v1_transfer':35,'gsm8k':30},
             'v1_preserved':True,'official_test_accessed':False}
    write_json(v2/'release_summary.json',summary)
    return summary
