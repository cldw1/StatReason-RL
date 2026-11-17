"""开发诊断版本的泄漏、独立复算、教师解答和分层指标测试。"""
from collections import Counter
import json
from pathlib import Path

import pytest

from src.data.diagnostic import generate_diagnostic_statistics, parameter_digest, worked_solution
from src.data.schema import content_fingerprint, validate_records
from src.data.statistics import TASK_TYPES
from src.evaluation.metrics import summarize_predictions
from src.rewards.numeric import verify_completion

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def diagnostic_rows():
    return generate_diagnostic_statistics(ROOT/'data/processed/statreason-v1/statistics_full')


def test_diagnostic_reproducible_and_independently_verified(diagnostic_rows):
    rows,audit=diagnostic_rows
    again,_=generate_diagnostic_statistics(ROOT/'data/processed/statreason-v1/statistics_full')
    assert content_fingerprint(rows)==content_fingerprint(again)
    assert Counter(r['split'] for r in rows)=={'sft':560,'rl':400,'validation':70}
    assert audit['independent_recompute_failures']==0
    assert audit['v1_test_body_accessed'] is False
    assert validate_records(rows)['cross_split_family_overlap']==0


def test_no_v1_parameters_reused_in_new_sft(diagnostic_rows):
    ids=json.loads((ROOT/'data/processed/statreason-v1/statistics_full/split_ids.json').read_text())
    forbidden={i.rsplit('-',1)[-1] for group in ids.values() for i in group}
    for r in diagnostic_rows[0]:
        if r['split']=='sft':
            assert parameter_digest(r['task_type'],r['parameters']) not in forbidden


@pytest.mark.parametrize('task',TASK_TYPES)
def test_each_task_has_both_training_skills_and_held_out_validation(task,diagnostic_rows):
    rows=[r for r in diagnostic_rows[0] if r['task_type']==task]
    assert Counter(r['parameters']['variant'] for r in rows if r['split']=='sft')=={0:40,2:40}
    assert Counter(r['evaluation_stratum'] for r in rows if r['split']=='validation')=={'core_calculation':5,'v1_transfer_validation':5}
    for r in rows:
        assert verify_completion(r['solution'],r).strict_correct


def test_teacher_affine_variance_is_squared_not_linear():
    p={'variant':2,'values':[0,2,4],'weights':[1,2,1],'coefficient':-3,'intercept':7}
    text,final=worked_solution('discrete_variance',p)
    assert final=='18'
    # 平移不影响方差，负系数必须平方；无需限定教师文本的具体写法。
    assert worked_solution('discrete_variance',{**p,'intercept':-100})[1]=='18'


def test_negative_standard_error_coefficient_is_parenthesized():
    text,_=worked_solution('standard_error',{'variant':2,'n':25,'sigma':2.,'coefficient':-3,'intercept':0})
    assert '(-3)²' in text and '=-3²' not in text


def test_strata_metrics_preserve_question_units():
    def row(i,stratum,correct):
        return {'id':i,'mode':'non-thinking','evaluation_stratum':stratum,'task_type':'x','source':'x',
                'verification':{'strict_correct':correct,'format_valid':True,'numeric_correct':correct,'parse_success':True},
                'truncated':False,'generated_token_count':3,'generation_seconds':.1}
    summary=summarize_predictions([row('a','core',True),row('b','transfer',False)])['by_mode']['non-thinking']
    assert summary['question_count']==2
    assert summary['by_evaluation_stratum']['core']['sampled_pass_at_1']==1
    assert summary['by_evaluation_stratum']['transfer']['sampled_pass_at_1']==0
