"""GRPO数学和数据边界测试；构造张量不作模型实验结果。"""
from copy import deepcopy
import math
from pathlib import Path

import pytest
import torch

from src.training.grpo import validate_vanilla,resolve_rl_fingerprint
from src.training.grpo_math import distribution_entropy,grpo_objective
from src.training.rl_data import prepare_rl_data
from src.utils.config import load_config

ROOT=Path(__file__).resolve().parents[1]


def test_frozen_rl_fingerprint_mismatch_is_not_overwritten():
    c={'data':{'fingerprint':'frozen'}}
    with pytest.raises(ValueError): resolve_rl_fingerprint(c,'changed')
    assert c['data']['fingerprint']=='frozen'
    resolve_rl_fingerprint(c,'frozen')
    smoke={'data':{'fingerprint':None}}
    resolve_rl_fingerprint(smoke,'actual')
    assert smoke['data']['fingerprint']=='actual'


def test_sample_mean_normalization_and_padding_gradient():
    current=torch.zeros(2,3,requires_grad=True)
    old=torch.zeros_like(current,requires_grad=True)
    ref=torch.zeros_like(current,requires_grad=True)
    advantage=torch.tensor([1.,-1.],requires_grad=True)
    mask=torch.tensor([[1.,0.,0.],[1.,1.,1.]])
    loss,_=grpo_objective(current,old,ref,advantage,mask,beta=0.)
    assert loss.item()==0.
    loss.backward()
    torch.testing.assert_close(current.grad,torch.tensor([[-.5,0,0],[1/6,1/6,1/6]]))
    assert old.grad is None and ref.grad is None and advantage.grad is None


@pytest.mark.parametrize('ratio,adv,effective,expected',[(1.5,1.,True,-1.2),(.5,-1.,True,.8),(.5,1.,False,-.5),(1.5,-1.,False,1.5)])
def test_clipping_depends_on_advantage_sign(ratio,adv,effective,expected):
    cur=torch.tensor([[math.log(ratio)]],requires_grad=True)
    zero=torch.zeros_like(cur)
    loss,d=grpo_objective(cur,zero,zero,torch.tensor([adv]),torch.ones_like(cur),beta=0.)
    assert loss.item()==pytest.approx(expected)
    assert bool(d['effective_low_clip']|d['effective_high_clip']) is effective
    loss.backward()
    if effective: assert cur.grad.item()==0.
    else: assert cur.grad.item()!=0.


def test_zero_advantages_can_have_kl_gradient():
    cur=torch.tensor([[-1.]],requires_grad=True)
    ref=torch.tensor([[-1.5]])
    loss,d=grpo_objective(cur,cur.detach(),ref,torch.zeros(1),torch.ones_like(cur))
    assert d['policy_loss']==0 and loss>0
    loss.backward()
    assert cur.grad.item()!=0


def test_masked_entropy_and_temperature():
    assert distribution_entropy(torch.zeros(2,4)).tolist()==pytest.approx([math.log(4)]*2)
    assert distribution_entropy(torch.tensor([[0.,-float('inf')]])).item()==0
    logits=torch.tensor([[0.,1.,2.]])
    assert distribution_entropy(logits/.7)<distribution_entropy(logits)
    with pytest.raises(ValueError): distribution_entropy(torch.full((1,3),-float('inf')))


def test_vanilla_guard_rejects_default_dapo_and_unaligned_cycle():
    c=load_config(ROOT/'configs/grpo.yaml')
    validate_vanilla(c)
    for change in [{'loss_type':'dapo'},{'save_steps':1},{'max_steps':3},{'sync_ref_model':True}]:
        invalid=deepcopy(c);invalid['trainer'].update(change)
        with pytest.raises(ValueError): validate_vanilla(invalid)


def test_real_rl_split_prompt_template_and_no_answer_leakage():
    from transformers import AutoTokenizer
    from src.models.snapshot import resolve_snapshot
    c=load_config(ROOT/'configs/grpo.yaml')
    tokenizer=AutoTokenizer.from_pretrained(resolve_snapshot(c['model']).path,local_files_only=True,padding_side='left')
    rows,records,s=prepare_rl_data(tokenizer,ROOT/c['paths']['dataset_release'],c['rl_data'],1024,42)
    assert s['selected_count']==16
    assert records[rows[0]['record_id']]['task_type']=='conditional_probability'
    assert records[rows[1]['record_id']]['task_type']=='gsm8k_math'
    assert all(r['split']=='rl' for r in records.values())
    assert all('<think>\n\n</think>' in r['prompt'] and r['prompt_token_count']<=1024 for r in rows)
    rows2,_,s2=prepare_rl_data(tokenizer,ROOT/c['paths']['dataset_release'],c['rl_data'],1024,42)
    assert rows==rows2 and s==s2
    with pytest.raises(ValueError): prepare_rl_data(tokenizer,ROOT/c['paths']['dataset_release'],c['rl_data'],1,42)


def test_trl_checkpoint_restore_regression_real_tiny_lora_cpu():
    from types import SimpleNamespace
    from transformers import Qwen3Config,Qwen3ForCausalLM
    from peft import LoraConfig,get_peft_model
    from trl.models.utils import unwrap_model_for_generation
    from src.training.grpo_trainer import AuditedGRPOTrainer
    tiny=Qwen3ForCausalLM(Qwen3Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                                    num_attention_heads=2,num_key_value_heads=1,head_dim=8))
    model=get_peft_model(tiny,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
    model.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    accelerator=SimpleNamespace(unwrap_model=lambda m:m,state=SimpleNamespace(deepspeed_plugin=None))
    with unwrap_model_for_generation(model,accelerator):
        assert not model.is_gradient_checkpointing
    # 本地0.23.1确实把False恢复成了默认True。
    functions=[m._gradient_checkpointing_func for m in model.modules() if hasattr(m,'_gradient_checkpointing_func')]
    assert any(f.keywords.get('use_reentrant',True) for f in functions)
    audited=object.__new__(AuditedGRPOTrainer)
    audited.model=model;audited.accelerator=accelerator
    audited.args=SimpleNamespace(gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False})
    audited.restore_checkpointing_configuration()
    assert audited.audit_checkpointing_configuration['use_reentrant'] is False
    loss=model(torch.tensor([[1,3,5,7]]),use_cache=False).logits.float().square().mean()
    assert loss.requires_grad
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_' in n)
    assert all(p.grad is None for n,p in model.named_parameters() if 'lora_' not in n)
