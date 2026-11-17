"""批量 completion 边界、配对统计和零方差处理的构造样例，不是模型成绩。"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.evaluation.comparison import paired_comparison
from src.models.inference import generate_completions
from src.training.group_rewards import relative_advantages
from src.utils.config import load_config

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('rewards',[[0,0,0,0],[1,1,1,1]])
def test_zero_variance_has_zero_relative_signal(rewards):
    result=relative_advantages(rewards)
    assert result['zero_variance'] and result['advantages']==[0]*4


def test_group_std_matches_bessel_corrected_reference():
    result=relative_advantages([0,0,1,1])
    assert result['sample_std']==pytest.approx(np.sqrt(1/3))
    assert sum(result['advantages'])==pytest.approx(0)
    assert result['advantages'][0]<0<result['advantages'][2]


@pytest.mark.parametrize('rewards',[[1],[0,float('nan')],[[0,1],[1,0]]])
def test_invalid_groups_fail(rewards):
    with pytest.raises(ValueError): relative_advantages(rewards)


def test_invalid_advantage_epsilon_fails():
    for epsilon in (0,-1,float('nan'),float('inf')):
        with pytest.raises(ValueError): relative_advantages([0,1],epsilon)


def test_paired_bootstrap_and_mcnemar_question_unit():
    a=[{'id':str(i),'mode':'non-thinking','evaluation_stratum':'core' if i<4 else 'transfer',
        'verification':{'strict_correct':i<2}} for i in range(8)]
    b=deepcopy(a)
    for row in b: row['verification']['strict_correct']=True
    result=paired_comparison(a,b,resamples=1000)
    assert result['second_minus_first_accuracy']==.75
    assert result['second_only_correct']==6 and result['first_only_correct']==0
    assert result['mcnemar_exact_two_sided_p']==pytest.approx(.03125)
    assert paired_comparison(a,b,resamples=1000)==result
    with pytest.raises(ValueError): paired_comparison(a,a+a,resamples=1000)
    with pytest.raises(ValueError): paired_comparison(a,b[:-1],resamples=1000)
    b[0]['seed']=99
    with pytest.raises(ValueError,match='seed'): paired_comparison(a,b,resamples=1000)


def test_batch_boundaries_left_padding_and_accounting(monkeypatch):
    import torch
    from transformers import AutoTokenizer, GenerationConfig
    from src.models.snapshot import resolve_snapshot
    config=load_config(ROOT/'configs/evaluation_diagnostic.yaml')
    snapshot=resolve_snapshot(config['model'])
    tokenizer=AutoTokenizer.from_pretrained(snapshot.path,local_files_only=True,padding_side='left')
    eos=tokenizer.eos_token_id
    for name in ('synchronize','reset_peak_memory_stats'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**kw:None)
    for name in ('max_memory_allocated','max_memory_reserved'):
        monkeypatch.setattr(torch.cuda,name,lambda *a,**kw:0)
    def generate(input_ids,attention_mask,generation_config):
        assert attention_mask[0].sum()<attention_mask[1].sum()
        assert (input_ids[0][attention_mask[0]==0]==tokenizer.pad_token_id).all()
        tail=torch.tensor([[10,eos,tokenizer.pad_token_id,tokenizer.pad_token_id],[11,12,13,eos]])
        return torch.cat([input_ids,tail],dim=1)
    model=SimpleNamespace(device=torch.device('cpu'),config=SimpleNamespace(model_type='qwen3'),
                          generation_config=GenerationConfig(eos_token_id=eos),generate=generate)
    rows=generate_completions(model,tokenizer,[{'id':'a','question':'2+2?'},{'id':'b','question':'Compute the sum of the first ten positive integers.'}],
                              config['evaluation'],enable_thinking=False,seed=42)
    assert [r['generated_token_count'] for r in rows]==[2,4]
    assert all(r['eos_seen'] and not r['truncated'] for r in rows)
    assert tokenizer.pad_token not in rows[0]['prompt_text']
    assert all(r['seed_scope']=='fixed_batch' for r in rows)
    assert sum(r['generation_seconds'] for r in rows)==pytest.approx(rows[0]['generation_batch_seconds'])
