# GRPO：公式、代码与本地实验证据

本项目在已有 Qwen3-0.6B 的 SFT LoRA 上继续强化学习，不是从零训练模型。实际后端为固定 TRL 0.23.1、Transformers 4.57.1、PEFT 0.17.1；`AuditedGRPOTrainer`保留TRL策略损失，同时增加参考策略、概率、奖励和恢复审计。

## 建议阅读顺序

| 文件/函数 | 阅读目标 |
|---|---|
| `configs/grpo.yaml`、`grpo_local.yaml` | 核对全部关键算法开关、SFT哈希、预算和数据范围 |
| `scripts/07_train_grpo.py` → `training/grpo.py::run_grpo_cli` | 模型/参考加载、配置身份、Trainer执行、导出与重载 |
| `training/rl_data.py::prepare_rl_data` | 只加载RL划分、生成完整Chat Template、排除过长题面 |
| `training/grpo_trainer.py::_generate_and_score_completions` | 生成G个回答、获取固定旧概率与参考概率、保存奖励组 |
| `strict_numeric_reward` → `rewards/numeric.py::verify_completion` | 仅判模型completion，检查EOS截断，输出0/1奖励 |
| `training/group_rewards.py::relative_advantages` | 独立复算组内均值、样本标准差与优势 |
| `_get_per_token_logps_and_entropies`、`_compute_loss` | 当前概率、三种熵、TRL目标及独立数学对照 |
| `training/grpo_math.py::grpo_objective` | 用小张量理解裁剪、KL和样本级归一化 |
| `scripts/14_analyze_grpo.py`、`15_verify_grpo_resume.py` | 真实曲线、同题统计和续训等价性验证 |

以上`training/`、`rewards/`路径均相对项目`src/`。配置与训练逻辑分开，SFT共同初始化位于`data/metadata/sft_checkpoint.json`。

## 题目与rollout

每道题的prompt只包含系统提示和题面，标准答案留给奖励函数。通过`apply_chat_template(..., enable_thinking=False)`预先生成纯文本prompt；TRL不会再次给它套一层对话模板。长度超过1024 Token的完整记录整体排除，不通过左截断保留半道题。当前完整4400条RL池均符合长度上限，最长311 Token。

同题生成G=4个回答。二元奖励为：完整结束、最终答案格式正确且数值通过原有容差，奖励1；否则0。生成长度上限128 Token，截断样本严格奖励0。`mask_truncated_completions=False`表示它们仍进入损失；这是独立于奖励处理的开关。

TRL按第一个EOS构造completion mask，包含EOS、排除后续padding。当前训练EOS为Tokenizer的`im_end`；固定开发评测沿用已有的统一停止配置与192 Token上限，全部模型相同。

## 组内优势

对同题奖励$R_1,\ldots,R_G$，当前实现与本地TRL源码对齐，采用样本标准差：

$$\bar R=\frac1G\sum_i R_i,\quad s_R=\sqrt{\frac{\sum_i(R_i-\bar R)^2}{G-1}},\quad A_i=\frac{R_i-\bar R}{s_R+10^{-4}}.$$

例如真实smoke组`[1,1,0,0]`的均值为0.5，样本标准差约0.57735，优势约`[0.865875,0.865875,-0.865875,-0.865875]`。`relative_advantages`独立计算后，与TRL返回优势逐项比较。

全对或全错时，优势为0。此时正确性策略梯度缺少组内相对信号；不能据此断言整个loss无梯度，因为策略若已偏离参考，KL项仍可推动参数变化。本地24步实验确实记录到这种情况。初始第一组全对且策略等于参考时，优势与KL均为0，梯度范数也确实为0。

## 当前策略、旧策略和参考策略

| 策略 | 用途 | 是否变化 |
|---|---|---|
| 当前策略 | 计算正在优化的Token概率 | 每次优化器更新改变LoRA |
| 旧策略 | rollout生成时的概率，用作比率分母 | 缓冲为脱离梯度的张量，固定两轮 |
| 参考策略 | 限制偏离共同SFT | 独立冻结SFT模型，整个run不更新 |

这里的概率目标沿用TRL 0.23.1语义，使用温度缩放后的**密集**分布：$\pi_T(\cdot|h)=\operatorname{softmax}(z_\theta(h)/T)$，$T=0.7$。采样另施加top-k=20和top-p=0.8，所以实际采样分布与目标密集分布有区别。本项目没有声称对过滤采样做了精确重要性修正。

重要性比率为$r_{i,t}=\exp(u_{\theta,i,t}-u_{\mathrm{old},i,t})$。当前、旧、参考的密集logprob都用同一温度和FP32概率计算。每行旧logprob的有效Token哈希随rollout保存，并在每次训练时核对，避免第二轮把分母替换成当前概率。

TRL的PEFT默认参考路径会禁用adapter，回到Base。本项目已经SFT，故显式加载另一个相同SFT副本，`requires_grad_(False)`并设为eval；它没有与策略共享参数存储。参考全部LoRA和抽样Base权重在训练前后核对哈希，所有参考梯度必须为None。首个正常rollout中策略/参考logprob最大差为0。

## 损失、裁剪和梯度累积

令$d=u_{\mathrm{ref}}-u_\theta$，KL估计项为$k=\exp(d)-d-1$。Vanilla取$\epsilon_{low}=\epsilon_{high}=0.2$、$\beta=0.04$：

$$\ell_{i,t}=-\min\{r_{i,t}A_i,\operatorname{clip}(r_{i,t},0.8,1.2)A_i\}+\beta k_{i,t}.$$

样本级GRPO先在每个回答内部按有效Token平均，再在回答之间平均：

$$L=\frac1G\sum_{i=1}^G\frac{\sum_t m_{i,t}\ell_{i,t}}{\max(1,\sum_t m_{i,t})}.$$

当前单卡microbatch=1，梯度累积4，四个回答一起构成一轮优化。TRL内部已处理该累积缩放，不能再人为除以4。`grpo_objective`返回未缩放目标，在每个microbatch与`TRL_loss×current_gradient_accumulation_steps`比较；实测最大差0。

`num_iterations=2`，因此同一个rollout在旧概率不变的前提下执行两轮优化，共8次microbatch前反传。只做一次更新时，初始比率通常为1，裁剪可能没有实际作用。

正优势只在比率过大时产生上侧目标裁剪，负优势只在比率过小时产生下侧目标裁剪。比率越界不一定改变`min`目标。例如零优势Token的比率越界，仍不能算作目标实际裁剪。日志分别记录两类比例，以及正/负优势Token数。

真实4步smoke的第一步loss为0，但梯度范数约1.944。这是组内优势数值平均抵消，不代表各Token概率的梯度抵消；224个LoRA张量确实发生了变化。

## 三种熵与统计口径

`distribution_entropy`按条件Token分布计算$H(p)=-\sum_v p_v\log p_v$，单位nat。对采样过滤后的`-inf`单独处理，避免$0\times(-\infty)$变成NaN。

- `policy_entropy_raw`：原始logits、T=1、完整词表。
- `policy_entropy_temperature`：logits除以0.7、完整词表，对应TRL的entropy含义。
- `policy_entropy_sampling`：温度之后，再按HF相同顺序执行top-k和top-p。

这些都是访问到的回答历史条件下的熵，不是对所有可能prompt精确积分。TRL日志对microbatch均值再平均；报告图从`policy_audit.jsonl`按有效Token数重新加权，并明确口径。跨不同题目的熵波动不能直接解释成训练导致的熵变化；同一个rollout在两轮中的变化另存。

## 版本兼容问题与真实修复

初次目标模型GRPO在反向传播时报`loss does not require grad`，没有完成优化器更新。查阅本机源码发现：TRL 0.23.1的`unwrap_model_for_generation`在生成后调用不带kwargs的`gradient_checkpointing_enable()`，把配置中的非重入模式变回默认重入模式；对冻结embedding的LoRA路径会丢失梯度图。

`restore_checkpointing_configuration`在初始化、当前模型概率计算及rollout结束后，显式恢复`use_reentrant=False`，并核对29个相关模块的实际设置。策略侧开启input-gradient hook，保持参数本身冻结。该问题由真实微型Qwen3+LoRA CPU回归测试复现并验证修复，不靠升级依赖或忽略警告。

## 保存、恢复与实测范围

运行目录保留完整配置、依赖、数据与SFT指纹、TRL源码哈希、rollout、policy审计、训练指标、adapter与优化器检查点。新建run保存关键训练源码快照。声明的数据指纹与实际记录不一致时拒绝训练，不静默修改冻结配置。

`max_steps`和`save_steps`必须是`num_iterations`的整数倍；本地每4步保存一次，检查点都落在完整rollout复用周期边界。中间周期的buffer不作为可恢复状态。

已经从完整24步实验的第20步检查点，在独立目录实际补跑到24步：额外生成529 Token；Token、奖励、优势、旧/参考概率相同；224个最终LoRA张量逐位相等、最大差0。结论只适用于本机、固定依赖与完整周期边界，不推广为跨硬件按位一致。

恢复严格检查代码、配置、依赖、数据和SFT哈希。已保存run绑定其训练时源码；若当前代码继续开发，应从匹配源码版本恢复，或先用当前代码建立新run再验证，不能绕过身份检查。

## 本地结果与阅读材料

24步实验：12题、48回答、3777生成Token，平均奖励0.6042，8/12组零方差；真正改变目标的裁剪约3/7554 Token曝光。固定100题中SFT与GRPO均19/100，各有3题独对；差值Bootstrap95%区间[-5,+5]个百分点。没有观察到可确证的能力提升或熵坍塌。

数据和图见`reports/grpo/report.md`、`results.json`、`sample_rollout.json`、`changed_correctness_cases.json`与`resume_verification.json`。当前还没有DAPO正式对照结果。

官方参考：[TRL 0.23.1 GRPO文档](https://huggingface.co/docs/trl/v0.23.1/en/grpo_trainer)、[对应Trainer源码](https://github.com/huggingface/trl/blob/v0.23.1/trl/trainer/grpo_trainer.py)、[生成解包工具源码](https://github.com/huggingface/trl/blob/v0.23.1/trl/models/utils.py)、[DeepSeekMath论文](https://arxiv.org/abs/2402.03300)。本项目以已安装源码和实际运行结果验收API。
