"""二元奖励的组内相对优势；这里只提供奖励诊断，不代表已完成 GRPO 更新。"""
import numpy as np


def relative_advantages(rewards, epsilon: float=1e-4) -> dict:
    values=np.asarray(rewards,dtype=float)
    if values.ndim!=1 or len(values)<2 or not np.isfinite(values).all() or not np.isfinite(epsilon) or epsilon<=0:
        raise ValueError('至少两个有限奖励，epsilon 必须为正')
    mean=float(values.mean())
    # 与已固定的 TRL 0.23.1 组内标准差语义对齐（Bessel 修正）。
    std=float(values.std(ddof=1))
    return {'mean':mean,'sample_std':std,'zero_variance':std==0.,
            'advantages':((values-mean)/(std+epsilon)).tolist(), 'epsilon':epsilon,'std_correction':1}
