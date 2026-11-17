"""重算历史判题、审计 SFT 标签/快照，并保存逐题人工语义诊断证据。"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.data.quality import audit_statistics
from src.data.storage import read_jsonl
from src.evaluation.answer_diagnostics import diagnose_answer
from src.evaluation.storage import append_prediction
from src.models.lora import validate_adapter
from src.models.prompts import build_messages
from src.models.snapshot import resolve_snapshot
from src.rewards.numeric import verify_completion
from src.training.sft_data import prepare_sft_data
from src.utils.artifacts import file_sha256, utc_now, write_json
from src.utils.config import load_config


RUNS = {'Base': 'base_development_20261116T080351Z_09285dd4',
        'SFT': 'sft_development_20261116T080656Z_fbd523ce',
        'GRPO': 'post_training_development_20261116T132326Z_c049b297',
        'SFT_revised': 'sft_development_20261116T082147Z_eb9eee43'}
SFT_RUN = 'sft_local_epoch1_20261115T173458Z_7e5291a1'

# Explicit semantic review of the frozen SFT prediction file, indexed in saved order.
# Do not infer these labels for other models. Every uncertain case stays separate.
FORMULA = {
    0: '将3位clarinet演奏者读成4位，且遗漏鼓手和指挥。',
    3: '将似然比除以先验概率，未按后验赔率再转换为概率。',
    6: '题目给失败概率0.2，模型当作成功概率0.2。',
    7: '频数[3,8,2]被忽略，改为三个取值等权平均。',
    11: '下限用加号，且混淆方差与标准差。',
    12: '标准误遗漏仿射系数绝对值3。',
    13: '均值差下限错误地将两组各自下限相加。',
    15: '条件空间B的计数计算错误。',
    18: '均值差置信区间错误引入未知μ，且分母用n而非平方根。',
    20: '将总体标准差加到仿射截距，最终SE遗漏σ。',
    21: '期望中将概率之和充当E(X)。',
    22: '均值差下限公式错误，分母用n。',
    23: '未按5个人各4个甜甜圈建模。',
    27: '忽略频数[1,1,7]，错误使用三个取值等权。',
    29: '期望的数值与频数配对错误，且遗漏总频数归一化。',
    30: '题目先验0.2却代入0.25；公式形式正确但已错误读取参数。',
    36: '条件概率使用错误分子分母，且把计数当概率。',
    37: '先验0.82及补概率被替换为0.69和0.68。',
    39: '把观测总和当均值，把总体方差直接用于误差项。',
    41: '应为(4+2)×4，错误改为加3再翻倍。',
    42: '题目6辆van被读成10辆。',
    43: '忽略频数，用三个取值等权计算方差。',
    44: '总重量遗漏每组10次重复。',
    46: '后验概率未结合先验赔率与似然比。',
    47: '将两个标准误直接相加，没有平方和开根。',
    48: '购买数量与单价组合错误，未按半数高价和全部低价计算。',
    51: '贝叶斯计算除以条件概率，并凭空设置P(E)=0.9。',
    57: '将奖学金错误减半，且工资余额没有减去父母支付。',
    59: '将观测总和当均值，标准误项结构也错误。',
    61: '写样本均值方差而非Y均值标准误，遗漏系数2。',
    62: '带频数方差的两个加权和均未乘频数。',
    63: '应为75+36/4，错误只计算36/4。',
    65: '加权和75未除总频数10。',
    66: 'Y的总体标准差仍写为11.7，遗漏系数绝对值2。',
    74: 'B计数用20+87，正确为21+20。',
    75: '后验只使用似然比，遗漏先验赔率7:4。',
    85: '重复加税，并从已付费用中减去找零。',
    90: '忽略频数[2,2,9]，用三个值等权。',
    91: 'P(E)未乘先验权重，最终只返回条件似然。',
    92: 'Y样本均值标准误遗漏仿射系数且只写方差形式。',
    93: '按每只鸡的数量错算，忽略鸡蛋按打计价。',
    95: '两组下限相减，未使用均值差的合并标准误。',
    97: '1减累积概率丢失括号，展开后只有第一项取负。',
    98: 'Y平方展开中X²系数应为4，模型写2。',
    99: '只计算每晚增加2小时，遗漏31晚。',
}
NUMERIC = {
    5: '完整二项点概率公式及参数正确，但乘积数值错误。',
    8: '明确写6.9/√145，最终数值与该表达式不符。',
    25: '明确写C(20,3)×0.2³×0.8¹⁷，但给大于1的概率。',
    26: '正确上尾补集表达式，但最终值错误。',
    32: '明确写17.9/√95，最终数字错误。',
    35: '上尾补集结构正确，P(X≤2)的数值错误。',
    49: '正确化简到95/105，随后小数值错误。',
    50: '正确二项上尾求和，最终数值错误。',
    54: '正确化简到106/182，小数值错误。',
    55: '正确写1-P(0)-P(1)-P(2)，但返回0。',
    69: '最后化简到98/122，随后小数值错误；连续斜杠记法有歧义。',
    70: '正确内部尺寸及24×24×12/1728表达式，最终误算成12。',
    71: '成功概率和二项PMF结构正确，但返回0。',
    79: '正确写C(9,5)×0.4⁵×0.6⁴，最终乘积错误。',
    81: '正确写15.5/√179，最终近似数错误。',
    83: '正确写2.6/√91，最终数值错误；有额外无关均值假设。',
    87: '正确写4300/100×3，误算为12.9而非129。',
    94: '正确写7.8/√168，最终数值错误。',
}
INSUFFICIENT = {
    4: '仿射期望恒等式正确，未展示E(X)计算，无法区分缺算与心算错误。',
    17: '只写通用期望公式，未展示频数代入。',
    24: '文字说明加权平均正确，没有具体代入。',
    38: '仿射恒等式正确，离散期望仅列未求值表达式。',
    56: '方差恒等式正确，但Y的取值与概率未完成代入。',
    60: '仿射期望恒等式正确，没有计算E(X)。',
    67: '方差恒等式正确，没有计算Y的两个矩。',
    68: '正确Var(Y)=16Var(X)，但直接返回16，未计算Var(X)。',
    88: '正确E(Y)=3E(X)-2，没有计算E(X)。',
}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    root = PROJECT_ROOT
    destination = root / 'reports/capability_round1'
    destination.mkdir(parents=True, exist_ok=True)
    results, reviewed = {}, []
    prediction_sets = {}
    for name, directory in RUNS.items():
        path = root / 'outputs/evaluations' / directory / 'evaluation_predictions.jsonl'
        rows = read_jsonl(path)
        prediction_sets[name] = rows
        mismatch = []
        for row in rows:
            fresh = verify_completion(row['completion'], row, truncated=row['truncated']).to_dict()
            if fresh != row['verification']:
                mismatch.append(row['id'])
        results[name] = {'prediction_file': str(path.relative_to(root)), 'sha256': file_sha256(path),
            'count': len(rows), 'strict_correct': sum(r['verification']['strict_correct'] for r in rows),
            'numeric_correct_diagnostic_same_tolerance': sum(diagnose_answer(r['completion'], r, truncated=r['truncated'])['numeric_correct_at_same_tolerance'] for r in rows),
            'format_valid': sum(r['verification']['format_valid'] for r in rows),
            'truncated': sum(r['truncated'] for r in rows), 'recomputed_verification_mismatches': mismatch,
            'reasons': dict(Counter(r['verification']['reason'] for r in rows)),
            'by_source': {source: {'n': sum(r['source'] == source for r in rows),
                'correct': sum(r['source'] == source and r['verification']['strict_correct'] for r in rows)} for source in {r['source'] for r in rows}}}
        if mismatch:
            raise ValueError('历史判题重算不一致')
    rows = prediction_sets['SFT']
    if results['SFT']['sha256'] != 'd38ee71b62eed273a4f1d07db7f0285506422f58942ea0c46658d1310c33d443':
        raise ValueError('手工语义分类只适用于已审阅的冻结预测文件')
    reference = prediction_sets['Base']
    assert [r['id'] for r in rows] == [r['id'] for r in reference]
    for index, row in enumerate(rows):
        diag = diagnose_answer(row['completion'], row, truncated=row['truncated'])
        if row['verification']['strict_correct']:
            category, evidence = 'correct', '严格最终答案通过原验证器。'
        elif row['truncated']:
            category, evidence = 'reasoning_truncated', '真实生成无EOS且达到长度/时间边界；不猜测未生成的结论。'
        elif diag['numeric_correct_at_same_tolerance']:
            category, evidence = 'correct_number_invalid_format', '唯一末尾答案同容差数值正确，严格格式未通过。'
        elif not row['verification']['parse_success']:
            category, evidence = 'answer_parse_failure', '最终标记未独占一行或最终值不是合法标量。'
        elif index in FORMULA:
            category, evidence = 'formula_or_problem_modeling_error', FORMULA[index]
        elif index in NUMERIC:
            category, evidence = 'correct_formula_numeric_error', NUMERIC[index]
        elif index in INSUFFICIENT:
            category, evidence = 'insufficient_work_to_disambiguate', INSUFFICIENT[index]
        else:
            raise ValueError(f'存在未审阅错误样本: {index}')
        reviewed.append({'index': index, 'id': row['id'], 'task_type': row['task_type'], 'source': row['source'],
            'primary_category': category, 'evidence': evidence, 'question': row['question'],
            'reference_answer': row['answer'], 'completion': row['completion'],
            'verification': row['verification'], 'diagnostics': diag,
            'review_method': 'explicit_semantic_review_of_this_frozen_prediction_file_not_keyword_classification'})
    review_path = destination / 'sft_error_review.jsonl'
    review_path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in reviewed), encoding='utf-8')
    counts = Counter(r['primary_category'] for r in reviewed)
    for key in ('formula_or_problem_modeling_error', 'correct_formula_numeric_error', 'correct_number_invalid_format',
                'answer_parse_failure', 'reasoning_truncated', 'insufficient_work_to_disambiguate'):
        counts.setdefault(key, 0)
    results['SFT_semantic_review'] = {'counts': dict(counts), 'denominator_all': len(rows),
        'denominator_errors': sum(not r['verification']['strict_correct'] for r in rows),
        'standard_answer_error_observed': 0,
        'standard_answer_error_scope': 'statistics independently recomputed; no exhaustive independent GSM8K proof',
        'question_ambiguity_note': 'gsm8k-train-01855 pet question relies on implicit no-overlap interpretation; retained as-is',
        'semantic_scope': 'SFT only; Base/GRPO have objective parser diagnostics, not transferred semantic labels'}
    numeric_errors = [abs(Decimal(r['verification']['parsed_answer']) - Decimal(r['answer'])) for r in rows
                      if r['source'] == 'statreason_synthetic' and r['verification']['parsed_answer'] is not None]
    results['SFT_statistics_precision_diagnostic'] = {
        'parsed_answers': len(numeric_errors),
        'absolute_error_at_most_1e_4': sum(e <= Decimal('0.0001') for e in numeric_errors),
        'absolute_error_at_most_1e_3': sum(e <= Decimal('0.001') for e in numeric_errors),
        'scope': 'absolute_error_bins_only_not_accuracy_or_changed_reward',
        'correction_to_previous_unverified_summary': '1e-3 bin has 4 answers, not 3; strict statistics remains 3/70'}
    legacy_path = root / 'outputs/evaluations/base_development_20261115T171125Z_4addb5e7/evaluation_predictions.jsonl'
    legacy = read_jsonl(legacy_path)
    results['legacy_192_token_audit'] = {
        'file_sha256': file_sha256(legacy_path),
        'by_mode': {mode: {'count': sum(r['mode'] == mode for r in legacy),
                          'truncated': sum(r['mode'] == mode and r['truncated'] for r in legacy),
                          'strict_correct': sum(r['mode'] == mode and r['verification']['strict_correct'] for r in legacy),
                          'max_new_tokens': sorted({r['generation_config']['max_new_tokens'] for r in legacy if r['mode'] == mode})}
                    for mode in ('thinking', 'non-thinking')},
        'scope': 'old_smoke_only_not_model_ability_estimate'}
    # Real tokenizer audit recomputes the complete encoded dataset and its fingerprint.
    train_root = root / 'outputs/runs' / SFT_RUN
    config = load_config(train_root / 'resolved_config.yaml')
    snapshot = resolve_snapshot(config['model'])
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot.path), local_files_only=True, trust_remote_code=False, padding_side='left')
    _, _, selection, audits = prepare_sft_data(tokenizer, root / config['paths']['dataset_release'], config['sft'], config['reproducibility']['seed'])
    old_selection = read_json(train_root / 'data_selection.json')
    assert selection == old_selection
    saved_audits = read_jsonl(train_root / 'label_audit.jsonl')
    assert audits == saved_audits
    prompt_mismatches = []
    for name, subset in prediction_sets.items():
        for row in subset:
            rendered = tokenizer.apply_chat_template(build_messages(row), tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if rendered != row['prompt_text']:
                prompt_mismatches.append([name, row['id']])
    if prompt_mismatches:
        raise ValueError('发现训练和评测提示不一致')
    adapter_info = validate_adapter(train_root / 'adapter', snapshot.repository, snapshot.revision)
    for name in ('SFT',):
        identity = read_json(root / 'outputs/evaluations' / RUNS[name] / 'adapter_identity.json')
        if identity != adapter_info:
            raise ValueError('预测使用的SFT权重与保留adapter不一致')
    training_statistics = []
    for split in ('sft', 'rl', 'validation'):
        training_statistics.extend(read_jsonl(root / 'data/processed/statreason-v1/statistics_full' / f'{split}.jsonl'))
    stats_audit = audit_statistics(training_statistics)
    eval_stats_audit = audit_statistics([r for r in read_jsonl(root / 'data/processed/statreason-diagnostic-v2/statistics_full/validation.jsonl')])
    updates = read_json(train_root / 'parameter_updates.json')
    results['implementation_audit'] = {'audited_at_utc': utc_now(), 'sft_selection_recomputed_exact_match': True,
        'label_audits_recomputed_exact_match': True, 'audited_label_rows': len(audits),
        'prompt_mask_failures': sum(not a['prompt_fully_masked'] for a in audits),
        'overlong_excluded': len(selection['sft']['excluded_overlong_ids']),
        'sft_mixture': selection['sft']['selected_by_source'],
        'supervised_tokens': selection['sft']['total_selected_supervised_tokens'],
        'prompt_template_mismatches': prompt_mismatches, 'retained_adapter_identity': adapter_info,
        'changed_lora_tensors': updates['changed_tensor_count'], 'all_non_lora_parameters_frozen': updates['all_non_lora_parameters_frozen'],
        'historical_reload_verification': read_json(train_root / 'reload_verification.json'),
        'v1_non_test_statistics_audit': stats_audit, 'old_development_statistics_audit': eval_stats_audit,
        'limitation': 'parameter-update/reload evidence is historical; hashes, encoding and labels rechecked now'}
    grpo = root / 'outputs/runs/grpo_local_24_20261116T094046Z_0f8144b9'
    rollouts = read_jsonl(grpo / 'rollout_groups.jsonl')
    groups = [g for rollout in rollouts for g in rollout['groups']]
    ids = [g['id'] for g in groups]
    pool = read_jsonl(root / 'data/processed/statreason-v1/gsm8k/rl.jsonl') + read_jsonl(root / 'data/processed/statreason-v1/statistics_full/rl.jsonl')
    by_id = {r['id']: r for r in pool}
    results['GRPO_rollout_audit'] = {'groups': len(groups), 'mixed_groups': sum(not g['zero_variance'] for g in groups),
        'all_correct': sum(all(r == 1 for r in g['rewards']) for g in groups),
        'all_wrong': sum(all(r == 0 for r in g['rewards']) for g in groups),
        'actual_question_source_counts': dict(Counter(by_id[i]['source'] for i in ids)),
        'actual_generated_completions': sum(len(g['rewards']) for g in groups)}
    write_json(destination / 'existing_diagnosis.json', results)
    lines = ['# 历史实验错误诊断', '', '来自逐题文件重算；旧100题已用于调参，统一标为开发集。', '',
             '| 模型 | 严格正确 | 同容差数值正确（诊断） | 格式有效 | 截断 |', '|---|---:|---:|---:|---:|']
    for name in RUNS:
        result = results[name]
        lines.append(f"| {name} | {result['strict_correct']}/100 | {result['numeric_correct_diagnostic_same_tolerance']}/100 | {result['format_valid']}% | {result['truncated']}% |")
    lines += ['', '以下语义分类仅适用于逐条审阅的原SFT模型，不将标签套用到Base或GRPO。类别互斥；优先记录截断和解析失败。', '',
              '| SFT分类 | 数量 | 占100题 | 占81个错误 |', '|---|---:|---:|---:|']
    labels = {'formula_or_problem_modeling_error': '公式、建模或读题代入错误', 'correct_formula_numeric_error': '公式正确但数值计算错误',
              'correct_number_invalid_format': '数值正确但格式错误', 'answer_parse_failure': '答案解析失败',
              'reasoning_truncated': '推理截断', 'insufficient_work_to_disambiguate': '过程不足，无法细分'}
    for key, label in labels.items():
        count = counts[key]
        lines.append(f'| {label} | {count} | {count:.1f}% | {count / 81:.1%} |')
    lines += ['', '统计训练/RL/验证题700条、旧开发统计70条均独立数值复算通过，未发现标准数值答案错误；没有据此宣称GSM8K所有题均无歧义。',
              'SFT的1200条训练样本及64条验证样本重新编码，与保存的label audit及指纹完全相同；提示词屏蔽、EOS监督、超长排除、400条评测的Chat Template检查均通过。',
              '1200条训练中统计200、GSM8K1000；224个LoRA张量的历史更新证据及重载证据均通过，当前adapter哈希与实际SFT评测加载身份一致。',
              '', '严格判题保持原样；boxed或行内标记只作诊断，不扫描过程找正确数字，不放宽容差。',
              '过程不足单列，不把只有通用公式的回答推断为已经正确代入。解析失败样例也可能同时有公式错误，主分类不会重复计数。',
              '', '完整逐题证据：[sft_error_review.jsonl](sft_error_review.jsonl)。机器可读审计：[existing_diagnosis.json](existing_diagnosis.json)。',
              '', '## 代表案例', '']
    for index in (3, 8, 6, 7, 72, 2, 68):
        row = reviewed[index]
        lines += [f"### {row['id']}", '', row['question'], '', f"标准答案：{row['reference_answer']}；主分类：{row['primary_category']}。{row['evidence']}", '',
                  '```text', row['completion'], '```', '']
    (destination / 'existing_diagnosis.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'scores': {k: results[k]['strict_correct'] for k in RUNS}, 'SFT_errors': counts,
                      'GRPO': results['GRPO_rollout_audit'], 'audit': 'passed'}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
