"""Bounded, single-GPU DAPO mechanism adapters around the pinned TRL trainer."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
import math
import random

from src.data.storage import read_jsonl
from src.evaluation.storage import append_prediction
from src.utils.config import validate_config
from .grpo_trainer import AuditedGRPOTrainer


class DynamicSamplingBudgetExhausted(RuntimeError):
    """No optimizer call is permitted for an unfilled dynamic sampling batch."""


def overlong_penalty(length: int, soft_length: int, maximum: int, coefficient: float = 1.) -> float:
    if not 0 < soft_length < maximum or length < 0 or not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError('invalid overlong penalty settings')
    return -coefficient * min(1., max(0., (length - soft_length) / (maximum - soft_length)))


def raw_group_status(rewards):
    if len(rewards) < 2 or any(r not in (0., 1.) for r in rewards):
        raise ValueError('dynamic selection uses raw binary correctness only')
    return {'mixed': min(rewards) < max(rewards), 'all_correct': all(r == 1 for r in rewards),
            'all_wrong': all(r == 0 for r in rewards)}


def freeze_generation_defaults(policy, generation_config):
    """Prevent Transformers >=4.50 from replacing global-default values.

    TRL passes a GenerationConfig rather than explicit generate(top_p=...)
    kwargs. Transformers may otherwise silently turn top_p=1 back into the
    checkpoint's 0.95. Freeze both PEFT and underlying model defaults, then
    resolve the actual HF configuration and verify it before any rollout.
    """
    policy.generation_config = deepcopy(generation_config)
    base = policy.get_base_model() if hasattr(policy, 'get_base_model') else policy
    base.generation_config = deepcopy(generation_config)
    actual, remaining = base._prepare_generation_config(generation_config)
    fields = ['do_sample', 'temperature', 'top_p', 'top_k', 'min_p', 'repetition_penalty', 'max_new_tokens']
    expected = {k: getattr(generation_config, k) for k in fields}
    observed = {k: getattr(actual, k) for k in fields}
    if observed != expected or remaining.get('top_p') is not None:
        raise RuntimeError('effective HF generation configuration differs from frozen protocol')
    return {'expected': expected, 'effective': observed, 'equal': True,
            'model_defaults_frozen_to_trl_generation_config': True}


@dataclass
class SamplingBudget:
    max_attempts: int
    max_tokens: int
    worst_case_attempt_tokens: int
    attempts: int = 0
    tokens: int = 0

    def can_attempt(self):
        return self.attempts < self.max_attempts and self.tokens + self.worst_case_attempt_tokens <= self.max_tokens

    def record(self, tokens):
        if not self.can_attempt() or not 0 <= tokens <= self.worst_case_attempt_tokens:
            raise ValueError('rollout exceeds reserved generation budget')
        self.attempts += 1
        self.tokens += tokens


ABLATIONS = {
    'E0': ('grpo', .2, False, False),
    'E1': ('grpo', .28, False, False),
    'E2': ('grpo', .2, True, False),
    'E3': ('dapo', .2, False, False),
    'E4': ('dapo', .28, True, True),
    'E5': ('grpo', .2, False, True),
}


def validate_ablation(config):
    validate_config(config)
    t = config['trainer']
    m = config['mechanisms']
    key = config['experiment']['ablation_id']
    actual = (t['loss_type'], t['epsilon_high'], m['dynamic_sampling']['enabled'], m['overlong']['enabled'])
    if key not in ABLATIONS or actual != ABLATIONS[key] or t['epsilon'] != .2:
        raise ValueError('mechanisms do not match the declared ablation')
    if t['top_p'] != 1. or t['top_k'] != 0 or t['min_p'] != 0. or t['repetition_penalty'] != 1.:
        raise ValueError('new protocol requires temperature-only sampling to match policy log probabilities')
    if t['steps_per_generation'] != t['gradient_accumulation_steps'] or t['per_device_train_batch_size'] != 1:
        raise ValueError('single optimizer batch must equal one generation batch')
    if t['gradient_accumulation_steps'] != t['num_generations']:
        raise ValueError('bounded implementation currently supports exactly one question group per optimizer batch')
    if t['max_steps'] % t['num_iterations'] or t['save_steps'] % t['num_iterations']:
        raise ValueError('stop and save only at rollout reuse boundaries')
    if t['mask_truncated_completions']:
        raise ValueError('truncation masks are held disabled to isolate length reward shaping')
    if t['scale_rewards'] != 'group' or t['importance_sampling_level'] != 'token' or t['delta'] is not None:
        raise ValueError('uncontrolled advantage or ratio change')
    if t['beta'] != .04 or t['sync_ref_model'] or t['top_entropy_quantile'] != 1.:
        raise ValueError('all ablations hold KL and entropy settings fixed')
    if config['initialization']['reference_policy'] != 'frozen_copy_of_same_sft_adapter':
        raise ValueError('reference must be an independent frozen copy of the same SFT')
    if t['generation_kwargs'] != {'do_sample': True, 'max_time': 60, 'num_beams': 1, 'num_return_sequences': 1}:
        raise ValueError('unsupported hidden generation override')
    if m['dynamic_sampling']['selection_reward'] != 'raw_binary_correctness':
        raise ValueError('selection must not use the shaped reward')
    if m['dynamic_sampling']['on_budget_exhausted'] != 'stop_training_before_update_and_log':
        raise ValueError('budget exhaustion must not perform an Adam momentum-only optimizer step')
    if m['dynamic_sampling']['max_generated_tokens_per_batch'] < t['num_generations'] * t['max_completion_length']:
        raise ValueError('budget cannot reserve even one complete group')


class MechanismGRPOTrainer(AuditedGRPOTrainer):
    """TRL clipping/token loss, custom bounded online sampling and length rewards.

    Scope is intentionally one question group per update, one CUDA process.
    Exhaustion stops before any update for that unfilled batch, including Adam
    momentum and weight decay. It is never simulated by a zero scalar loss.
    """

    def __init__(self, *, experiment_config, **kwargs):
        self.mechanisms = experiment_config['mechanisms']
        self.protocol = experiment_config
        self.sampling_pool = sorted(kwargs['train_dataset'], key=lambda r: r['record_id'])
        random.Random(experiment_config['reproducibility']['data_seed'] + 7919).shuffle(self.sampling_pool)
        path = kwargs['run'] / 'sampling_audit.jsonl'
        previous = read_jsonl(path) if path.exists() else []
        self.sampling_cursor = previous[-1]['next_candidate_cursor'] if previous else 0
        self.sampling_attempts = len(previous)
        super().__init__(**kwargs)
        self.generation_protocol_audit = freeze_generation_defaults(
            self.accelerator.unwrap_model(self.model), self.generation_config)

    def strict_numeric_reward(self, completions, completion_ids, record_id, **kwargs):
        binary = super().strict_numeric_reward(completions, completion_ids, record_id, **kwargs)
        settings = self.mechanisms['overlong']
        for row, raw in zip(self.audit_reward_rows, binary, strict=True):
            penalty = (overlong_penalty(len(row['completion_ids']), settings['soft_length'],
                                       self.args.max_completion_length, settings['penalty_coefficient'])
                       if settings['enabled'] else 0.)
            row.update(accuracy_reward=raw, format_reward=0., length_penalty=penalty, reward=raw + penalty,
                       length_includes_eos=True, truncation_loss_mask_enabled=False)
        return [r['reward'] for r in self.audit_reward_rows]

    def _generate_and_score_completions(self, inputs):
        import torch
        if len(inputs) != self.num_generations or len({r['record_id'] for r in inputs}) != 1:
            raise ValueError('expected exactly one complete repeated question group')
        settings = self.mechanisms['dynamic_sampling']
        budget = SamplingBudget(1 + settings['max_resample_rounds'] if settings['enabled'] else 1,
                                settings['max_generated_tokens_per_batch'],
                                self.num_generations * self.max_completion_length)
        initial_id = inputs[0]['record_id']
        while budget.can_attempt():
            if budget.attempts:
                candidate = self.sampling_pool[self.sampling_cursor % len(self.sampling_pool)]
                self.sampling_cursor += 1
                inputs = [dict(candidate) for _ in range(self.num_generations)]
            output = super()._generate_and_score_completions(inputs)
            generated = sum(len(row['completion_ids']) for row in self.audit_reward_rows)
            raw = [row['accuracy_reward'] for row in self.audit_reward_rows]
            group = raw_group_status(raw)
            accepted = not settings['enabled'] or group['mixed']
            attempt_index = budget.attempts
            budget.record(generated)
            self.sampling_attempts += 1
            # TRL computes this before its optional truncation mask. Explicitly
            # use actual participating tokens, shared across all microbatches.
            output['num_items_in_batch'] = output['completion_mask'].sum().clamp(min=1)
            append_prediction(self.audit_run / 'sampling_audit.jsonl', {
                'sampling_attempt_id': self.sampling_attempts, 'rollout_id': self.audit_rollouts,
                'optimizer_step_before_batch': self.state.global_step, 'initial_question_id': initial_id,
                'question_id': inputs[0]['record_id'], 'attempt_index_in_batch': attempt_index,
                'raw_rewards': raw, 'shaped_rewards': [row['reward'] for row in self.audit_reward_rows],
                **group, 'accepted_for_training': accepted, 'dynamic_enabled': settings['enabled'],
                'generated_tokens': generated, 'batch_generated_tokens_so_far': budget.tokens,
                'next_candidate_cursor': self.sampling_cursor,
                'budget_exhausted_without_update': not accepted and not budget.can_attempt(),
                'sampling_is_online': True, 'selection_uses': 'raw_binary_correctness',
                'reused_optimizer_steps_planned': self.num_iterations if accepted else 0,
            })
            if accepted:
                return output
        raise DynamicSamplingBudgetExhausted(
            f'No mixed raw reward group after {budget.attempts} attempts / {budget.tokens} generated tokens; '
            'stopped before optimizer update for this batch.')
