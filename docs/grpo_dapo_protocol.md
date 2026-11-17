# GRPO/DAPO受控机制实验协议

目标是以真实训练、内部指标与统一评测判定机制行为。性能没有显著提高时，仍可形成机制分析；只有代码或4步检查不能达到B级。

## 实现归属与环境

使用PyTorch 2.8.0+cu128、Transformers 4.57.1、TRL 0.23.1、PEFT 0.17.1、Accelerate 1.10.1。Rollout、策略裁剪、优化器和两种损失聚合基于TRL集成。项目实现严格奖励、冻结SFT参考策略、独立损失与梯度校验、在线补采、长度奖励与逐Token审计。

主模型复用Qwen3-0.6B原SFT，导出适配器SHA256为`1ba65fe7e4537b6c5a18684adf4a6223e6fce8a4347414cba568e72062070cfa`。该模型来自300步SFT中验证loss选择的150步检查点。每组从该适配器独立加载，不串行续训各实验。三模型开发对照保留，暂不新增完整SFT周期。

## 五组实验

| 组别 | 上裁剪ε | 损失 | 在线筛选补采 | 长度奖励 |
|---|---:|---|---|---|
| E0 | 0.20 | 回答归一化GRPO | 关闭 | 关闭 |
| E1 | 0.28 | 回答归一化GRPO | 关闭 | 关闭 |
| E2 | 0.20 | 回答归一化GRPO | 开启 | 关闭 |
| E3 | 0.20 | 全局Token归一化 | 关闭 | 关闭 |
| E4 | 0.28 | 全局Token归一化 | 开启 | 开启 |

E5仅长度奖励为可选扩展。五组固定下裁剪ε=0.20、KL系数β=0.04、相同SFT参考、LoRA、优化器、学习率和训练数据。论文原始DAPO配方去掉KL；本项目E4保留共同KL以控制变量，应称“四机制组合的受控DAPO适配”，不宣称完整复刻论文大规模配方和结果。

## 奖励与优势

严格奖励只接受唯一、独占末行的`FINAL_ANSWER: 数值`，不搜索推理中的数字。容差沿用每题声明，概率语义允许时支持百分数，其他合法表达包括整数、小数、科学计数法和分数。重复答案、非法表达、NaN/Inf、截断回答均不能获得严格正确奖励。boxed只供诊断，不混入训练奖励。

组内优势使用TRL 0.23.1的样本标准差，保留其稳定项：

$$A_i=\frac{r_i-\bar r}{s_r+10^{-4}},\qquad s_r^2=\frac{1}{G-1}\sum_i(r_i-\bar r)^2.$$

E0至E3的$r_i$为严格正确性0/1；E4为该奖励加长度惩罚。格式奖励固定为0，未进行格式奖励消融。零方差组的策略优势为0，但β>0时仍可能存在KL梯度，不能将其一律称为完全没有参数更新。

## 策略概率与裁剪

新实验固定Non-thinking官方模板，温度0.7，`top_p=1, top_k=0, min_p=0, repetition_penalty=1`。温度缩放后的完整softmax既用于采样，也用于当前、旧策略和参考策略log probability，避免额外过滤分布不匹配。单独记录原始模型熵、温度熵和实际采样熵，单位nat。

Transformers 4.57.1可能将GenerationConfig中等于全局默认值的字段回退为模型默认值，即使TRL显式传入top_p=1也可能回退到0.95。`freeze_generation_defaults`同时冻结PEFT及底层模型生成默认值，并调用实际HF配置解析检查。每个新运行保存`effective_generation_audit.json`。已观察到此问题的早期4步检查完整保留，以`protocol_audit.json`排除出公平比较。

$$\rho_{it}=\exp(\log\pi_\theta(y_{it})-\log\pi_{old}(y_{it})),\quad
\ell_{it}=-\min(\rho_{it} A_i,\operatorname{clip}(\rho_{it},1-\epsilon_{low},1+\epsilon_{high})A_i)+\beta k_{it}.$$

$$k_{it}=\exp(\log\pi_{ref}-\log\pi_\theta)-(\log\pi_{ref}-\log\pi_\theta)-1.$$

每组Rollout固定旧概率并复用两次优化器更新。第一个更新前比例通常接近1；第二次更新才能观察偏离。记录越界比例和真正影响损失的裁剪比例，按优势符号区分。没有实际触发上裁剪时不能声称Clip-Higher改善探索。

## 损失归一化与梯度累积

令$m_{it}$为Completion有效Token掩码。回答归一化与Token归一化分别为：

$$L_{GRPO}=\frac1B\sum_i\frac{\sum_t m_{it}\ell_{it}}{\sum_t m_{it}},\qquad
L_{token}=\frac{\sum_{i,t}m_{it}\ell_{it}}{\sum_{i,t}m_{it}}.$$

单卡当前实现每个优化器批次恰含一道题的G个回答，batch size=1，gradient accumulation=G，steps per generation=G。Token损失的分母为该整个优化器批次的有效Completion Token数，所有微批共用该分母，不能再次除以累积步数。TRL自己的`compute_loss_func`设置关闭父Trainer额外缩放；Accel­erate维持TRL配置的缩放行为。

测试直接调用已安装TRL的`_compute_loss`，比较不同长度、Padding和三次微批累积相对整批公式的损失及梯度。实际GPU每个微批同时独立复算并记录误差。当前五组均不掩掉截断回答，因此整批分母与实际参与Token数一致。

## 在线Dynamic Sampling

`MechanismGRPOTrainer`先生成完整G回答组，以**原始二元正确性**判断是否同时有对有错。只有E2、E4拒绝零方差原始组。被拒后从同一RL训练池的确定性打乱顺序取下一题补采，不能读取开发或最终测试答案筛选训练数据。E4不会将长度惩罚造成的奖励差异误当作原始正确性学习信号。

每次补采预留完整G×最大长度的Token上界，同时限制补采次数，永不开始无法覆盖最坏长度的下一组。预算耗尽而未填入有效组时，抛出专门停止信号，在优化器更新前终止该运行；不会以零损失调用Adam，因为动量仍可能改变参数。此前已完成的更新及检查点保留，状态记为`sampling_budget_stopped`，不能报告成计划步数已完成。

`sampling_audit.jsonl`记录每次尝试的原题、新题、原始奖励、整形奖励、接受状态、下一候选游标和Token消耗，关联`rollout_id`；`policy_audit.jsonl`用同一ID记录实际参与的微批与优化器步。额外补采Token与被丢弃Rollout Token分开记录，生成总成本包括两者。

这属于训练中的在线筛选。当前基于来源配额的训练池构造只是离线分层抽样，并未按模型正确率筛选，不属于Dynamic Sampling。

## Overlong Reward Shaping

$$p(L)=-c\min\left(1,\max\left(0,\frac{L-L_{soft}}{L_{max}-L_{soft}}\right)\right),\qquad r=r_{accuracy}+p(L).$$

检查配置使用$L_{soft}=96,L_{max}=128,c=1$；长度包含EOS，与TRL Completion长度口径一致。原始奖励、长度惩罚及最终奖励分别保存。

**超长惩罚与截断损失掩码是不同机制。** 五组统一`mask_truncated_completions=false`，E4只增加长度奖励，不偷偷更改Completion mask。截断回答严格正确性为0，但参与策略和KL损失。

## 本地检查与正式预算边界

`configs/ablations/E0_smoke.yaml`至`E4_smoke.yaml`为每组4步、G=4的实现检查，16条RL池包括GSM12条、统计4条。它们不是正式实验，也不能用于声称准确率提升、训练坍塌或B级验收。

```powershell
.venv\Scripts\python.exe -m pytest tests/test_dapo_mechanisms.py tests/test_grpo.py -q
.venv\Scripts\python.exe scripts/21_run_ablation_smokes.py --execute
# 单独运行某组：
.venv\Scripts\python.exe scripts/07_train_grpo.py --config configs/ablations/E3_smoke.yaml
```

旧Vanilla检查点已验证20至24步精确恢复。E4组合也已实测从第2步恢复到第4步，224个最终适配器张量、生成Token、奖励、旧/参考概率和在线采样决策均与连续运行一致，额外生成1177 Token。两者均限于同机固定依赖，不推断跨硬件逐位一致。恢复入口拒绝检查点之后存在未提交补采记录的运行，以免错用RNG和候选游标。代码、版本、奖励、数据、SFT哈希变化也会拒绝直接恢复旧运行；原代码快照保留。

正式预算应根据本轮实测训练与补采吞吐确定，再冻结来源比例、题量、步数和评测参数。较大GPU占用需用户确认。正式评测前保持测试封存，报告题目级配对Bootstrap、Wilson区间和分层结果。单随机种子只能作为预算受限实验，不能推断跨种子稳定改进。

参考：[TRL 0.23.1](https://huggingface.co/docs/trl/v0.23.1/grpo_trainer)、[固定版本源码](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_trainer.py)、[DAPO论文](https://arxiv.org/abs/2503.14476)、[DAPO官方仓库](https://github.com/BytedTsinghua-SIA/DAPO)、[Open-R1](https://github.com/huggingface/open-r1)、[Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)。
