"""Mechanism behavior, actual pinned TRL loss, and bounded online selection."""
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from trl import GRPOTrainer

from src.data.storage import read_jsonl
from src.training.dapo_mechanisms import (DynamicSamplingBudgetExhausted, MechanismGRPOTrainer,
                                         SamplingBudget, overlong_penalty, raw_group_status,
                                         validate_ablation, freeze_generation_defaults)
from src.training.grpo_math import grpo_objective
from src.training.grpo_trainer import AuditedGRPOTrainer
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_transformers_model_default_fallback_cannot_silently_restore_top_p_filter():
    from transformers import GenerationConfig, Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    tiny = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    tiny.generation_config = GenerationConfig(do_sample=True, temperature=.6, top_p=.95, top_k=20,
                                               transformers_version='4.51.0')
    protocol = GenerationConfig(do_sample=True, temperature=.7, top_p=1., top_k=0, min_p=0.,
                                 max_new_tokens=128, repetition_penalty=1.)
    before, _ = tiny._prepare_generation_config(protocol)
    assert before.top_p == .95  # Reproduce the actual silent fallback.
    policy = get_peft_model(tiny, LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj'], task_type='CAUSAL_LM'))
    audit = freeze_generation_defaults(policy, protocol)
    after, _ = policy.get_base_model()._prepare_generation_config(protocol)
    assert audit['equal'] and after.top_p == 1. and after.top_k == 0


@pytest.mark.parametrize('advantage', [1., -1.])
def test_clip_higher_changes_only_positive_advantage_gradient_at_125_percent(advantage):
    gradients, losses = [], []
    for high in [.2, .28]:
        current = torch.tensor([[math.log(1.25)]], requires_grad=True)
        zero = torch.zeros_like(current)
        loss, _ = grpo_objective(current, zero, zero, torch.tensor([advantage]), torch.ones_like(zero),
                                 beta=0., epsilon_high=high)
        loss.backward()
        gradients.append(current.grad.item())
        losses.append(loss.item())
    if advantage > 0:
        assert gradients == pytest.approx([0., -1.25])
        assert losses == pytest.approx([-1.2, -1.25])
    else:
        assert gradients == pytest.approx([1.25, 1.25])


def test_loss_normalization_weights_short_and_long_answers_differently():
    mask = torch.tensor([[1., 0., 0.], [1., 1., 1.]])
    advantage = torch.tensor([1., -1.])
    observed = {}
    for kind in ['grpo', 'dapo']:
        current = torch.zeros(2, 3, requires_grad=True)
        loss, _ = grpo_objective(current, current.detach(), current.detach(), advantage, mask,
                                 beta=0., loss_type=kind)
        loss.backward()
        observed[kind] = (loss.item(), current.grad)
    assert observed['grpo'][0] == 0.
    assert observed['dapo'][0] == .5
    torch.testing.assert_close(observed['grpo'][1], torch.tensor([[-.5, 0, 0], [1/6, 1/6, 1/6]]))
    torch.testing.assert_close(observed['dapo'][1], torch.tensor([[-.25, 0, 0], [.25, .25, .25]]))


@pytest.mark.parametrize('kind', ['grpo', 'dapo'])
def test_installed_trl_accumulation_equals_unsplit_objective_and_gradient(kind):
    # Test the actual installed TRL method, with no model or GPU required.
    # Unequal lengths exercise padding and a global, not per-microbatch, token denominator.
    current = torch.tensor([[.12, -.2, .3, .5], [-.3, .1, -.4, .2], [.1, -.3, -.1, .4]], requires_grad=True)
    old = torch.zeros_like(current)
    ref = torch.full_like(current, -.1)
    advantages = torch.tensor([1., -1., .5])
    mask = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 0, 0]])
    whole, _ = grpo_objective(current, old, ref, advantages, mask, loss_type=kind, epsilon_high=.28)
    whole_grad = torch.autograd.grad(whole, current)[0]
    micro_current = current.detach().clone().requires_grad_()
    trainer = object.__new__(GRPOTrainer)
    trainer.top_entropy_quantile = 1.
    trainer.beta = .04
    trainer.epsilon_low, trainer.epsilon_high = .2, .28
    trainer.loss_type, trainer.importance_sampling_level = kind, 'token'
    trainer.use_vllm, trainer.current_gradient_accumulation_steps = False, 3
    trainer.args = SimpleNamespace(delta=None)
    trainer.accelerator = SimpleNamespace(num_processes=1, gather=lambda t: t)
    trainer._metrics = {'train': defaultdict(list)}
    model = SimpleNamespace(training=True)
    trainer.model = model
    losses = []
    for index in range(3):
        row = slice(index, index+1)
        trainer._get_per_token_logps_and_entropies = lambda *a, row=row, **kw: (micro_current[row], torch.zeros(1, 4))
        inputs = {'prompt_ids': torch.ones(1, 2, dtype=torch.long), 'prompt_mask': torch.ones(1, 2),
                  'completion_ids': torch.ones(1, 4, dtype=torch.long), 'completion_mask': mask[row],
                  'advantages': advantages[row], 'old_per_token_logps': old[row],
                  'ref_per_token_logps': ref[row], 'num_items_in_batch': mask.sum()}
        losses.append(GRPOTrainer._compute_loss(trainer, model, inputs))
    accumulated = sum(losses)
    accumulated.backward()
    torch.testing.assert_close(accumulated.detach(), whole.detach())
    torch.testing.assert_close(micro_current.grad, whole_grad)
    assert torch.all(micro_current.grad[mask == 0] == 0)


@pytest.mark.parametrize('length,expected', [(0, 0.), (96, 0.), (112, -.5), (128, -1.), (200, -1.)])
def test_overlong_penalty_boundaries(length, expected):
    assert overlong_penalty(length, 96, 128) == expected


def test_dynamic_uses_binary_accuracy_not_length_reward_variation():
    assert not raw_group_status([0., 0., 0., 0.])['mixed']
    assert raw_group_status([1., 0., 0., 0.])['mixed']
    with pytest.raises(ValueError):
        raw_group_status([0., -.5, -.2, -1.])


def test_budget_reserves_entire_next_group_and_honors_attempt_limit():
    budget = SamplingBudget(4, 1200, 512)
    budget.record(400)
    budget.record(300)
    assert not budget.can_attempt()  # Only 500 left: never begin a 512-token group.
    with pytest.raises(ValueError):
        budget.record(1)
    cap = SamplingBudget(1, 9000, 512)
    cap.record(10)
    assert not cap.can_attempt()


@pytest.mark.parametrize('name', ['E0', 'E1', 'E2', 'E3', 'E4'])
def test_five_configs_explicitly_match_declared_mechanisms(name):
    config = load_config(ROOT / f'configs/ablations/{name}_smoke.yaml')
    validate_ablation(config)
    bad = deepcopy(config)
    bad['trainer']['top_p'] = .8
    with pytest.raises(ValueError):
        validate_ablation(bad)
    bad = deepcopy(config)
    bad['mechanisms']['dynamic_sampling']['on_budget_exhausted'] = 'zero_loss_update'
    with pytest.raises(ValueError):
        validate_ablation(bad)


def online_fixture(monkeypatch, tmp_path, rewards, rounds):
    trainer = object.__new__(MechanismGRPOTrainer)
    trainer.num_generations, trainer.max_completion_length = 4, 128
    trainer.mechanisms = {'dynamic_sampling': {'enabled': True, 'max_resample_rounds': rounds,
                                              'max_generated_tokens_per_batch': 4096}}
    trainer.state = SimpleNamespace(global_step=7)
    trainer.audit_run = tmp_path
    trainer.audit_rollouts = trainer.sampling_attempts = trainer.sampling_cursor = 0
    trainer.num_iterations = 2
    trainer.sampling_pool = [{'record_id': 'replacement'}]
    sequence = iter(rewards)

    def generate(self, inputs):
        self.audit_rollouts += 1
        self.audit_reward_rows = [{'completion_ids': [1, 2, 3], 'accuracy_reward': value, 'reward': value}
                                 for value in next(sequence)]
        return {'completion_mask': torch.ones(4, 3), 'old_per_token_logps': torch.full((4, 3), float(self.audit_rollouts))}

    monkeypatch.setattr(AuditedGRPOTrainer, '_generate_and_score_completions', generate)
    return trainer


def test_online_replaces_zero_variance_group_and_preserves_accepted_old_probabilities(monkeypatch, tmp_path):
    trainer = online_fixture(monkeypatch, tmp_path, [[0., 0., 0., 0.], [1., 0., 1., 0.]], rounds=1)
    output = trainer._generate_and_score_completions([{'record_id': 'initial'}] * 4)
    rows = read_jsonl(tmp_path / 'sampling_audit.jsonl')
    assert [r['accepted_for_training'] for r in rows] == [False, True]
    assert [r['question_id'] for r in rows] == ['initial', 'replacement']
    assert output['num_items_in_batch'] == 12
    assert torch.all(output['old_per_token_logps'] == 2.)
    assert trainer.state.global_step == 7


def test_budget_exhaustion_raises_before_any_optimizer_call(monkeypatch, tmp_path):
    trainer = online_fixture(monkeypatch, tmp_path, [[1., 1., 1., 1.]], rounds=0)
    with pytest.raises(DynamicSamplingBudgetExhausted):
        trainer._generate_and_score_completions([{'record_id': 'initial'}] * 4)
        pytest.fail('an unfilled group must never reach the optimizer')
    assert trainer.state.global_step == 7
    assert read_jsonl(tmp_path / 'sampling_audit.jsonl')[0]['budget_exhausted_without_update']
