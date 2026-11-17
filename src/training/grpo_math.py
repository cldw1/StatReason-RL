"""独立可测试的 GRPO 目标与诊断，当前只验证单卡、Token 重要性比率。"""
from __future__ import annotations

import torch


def distribution_entropy(logits: torch.Tensor) -> torch.Tensor:
    """允许过滤后的 -inf，避免 0×(-inf) 产生 NaN。单位为 nat。"""
    values=logits.float()
    if torch.isnan(values).any() or torch.isposinf(values).any() or not torch.isfinite(values).any(dim=-1).all():
        raise ValueError('每个分布至少一个有限 logit，不能有 NaN/+Inf')
    probabilities=values.softmax(-1)
    safe=torch.where(torch.isfinite(values),values,torch.zeros_like(values))
    return values.logsumexp(-1)-(probabilities*safe).sum(-1)


def grpo_objective(current, old, reference, advantages, mask, *, beta=.04, epsilon=.2, epsilon_high=.2,
                   loss_type='grpo', token_normalizer=None):
    """每回答先按有效Token平均，再按回答平均；不包含梯度累积的外层缩放。"""
    if current.ndim!=2 or current.shape!=old.shape or current.shape!=reference.shape or current.shape!=mask.shape:
        raise ValueError('logprob 与 mask 必须形状一致且为二维')
    if advantages.shape!=(current.shape[0],) or beta<0 or epsilon<=0 or epsilon_high<=0:
        raise ValueError('优势形状、KL系数或裁剪阈值错误')
    if not all(torch.isfinite(x).all() for x in (current,old,reference,advantages)):
        raise FloatingPointError('非有限logprob/advantage')
    weights=mask.to(current.dtype)
    ratio=(current-old.detach()).exp()
    clipped=ratio.clamp(1-epsilon,1+epsilon_high)
    a=advantages.detach().unsqueeze(1)
    policy=-torch.minimum(ratio*a,clipped*a)
    difference=reference.detach()-current
    kl=difference.exp()-difference-1
    lengths=weights.sum(-1).clamp(min=1)
    if loss_type == 'grpo':
        aggregate=lambda tensor: ((tensor*weights).sum(-1)/lengths).mean()
    elif loss_type == 'dapo':
        # Each microbatch shares the whole optimizer-step token denominator.
        # This branch must not divide again by gradient accumulation steps.
        denominator = weights.sum() if token_normalizer is None else torch.as_tensor(token_normalizer, device=current.device)
        if not torch.isfinite(denominator) or denominator <= 0:
            raise ValueError('token normalizer must be positive and finite')
        aggregate=lambda tensor: (tensor*weights).sum()/denominator
    else:
        raise ValueError('unsupported loss aggregation')
    loss=aggregate(policy+beta*kl)
    return loss, {'ratio':ratio,'policy':policy,'kl':kl,'policy_loss':aggregate(policy),
                  'sample_normalized_kl':aggregate(kl),
                  'low_outside':ratio<1-epsilon,'high_outside':ratio>1+epsilon_high,
                  'effective_low_clip':(ratio<1-epsilon)&(a<0),
                  'effective_high_clip':(ratio>1+epsilon_high)&(a>0)}


def masked_mean(values, mask) -> float:
    return float((values.float()*mask).sum()/mask.sum().clamp(min=1))
