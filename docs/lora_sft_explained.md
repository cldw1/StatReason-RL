# 当前项目的 LoRA 和 SFT

## 从实际调用看训练

`scripts/05_train_sft.py` 调用 `src/training/sft.py:run_sft_cli()`。函数从固定 Qwen 快照加载 Tokenizer，处理并审计标签，加载 BF16 Base，注入 LoRA，交给 Transformers Trainer 进行真实反传和优化器更新，最后保存 adapter 并在重新加载的 Base 上核对权重和 logits。

SFT使用Transformers Trainer和PEFT，自行构造并审计助手标签掩码。GRPO/DAPO已基于TRL 0.23.1完成集成与五组真实训练；两种阶段分别使用监督交叉熵与强化学习目标。

## LoRA 如何改变模型

原始线性层为 `y=Wx`。LoRA 冻结 W，并学习较小的两个矩阵 A、B：

\[
y=Wx+\frac{\alpha}{r}BAx,
\qquad A\in\mathbb{R}^{r\times d_{in}},\quad
B\in\mathbb{R}^{d_{out}\times r}.
\]

本项目 r=8、alpha=16，所以缩放系数为 2，dropout=0。只向 q_proj、k_proj、v_proj、o_proj 注入 LoRA，不训练原始模型权重，不使用量化。因此这是 BF16 Base 上的普通 LoRA，不是 QLoRA。

当前 Qwen3-0.6B 实际有 28 层，hidden_size=1024、16 个 query head、8 个 KV head、head_dim=128。注意 q_proj 输出为 2048 维，不能直接把所有投影都当成 1024×1024。

| 模块 | 输入维度 | 输出维度 | 每层 LoRA 参数 |
|---|---:|---:|---:|
| q_proj | 1024 | 2048 | 8×(1024+2048)=24576 |
| k_proj | 1024 | 1024 | 16384 |
| v_proj | 1024 | 1024 | 16384 |
| o_proj | 2048 | 1024 | 24576 |

每层 81920，28 层合计 2293760 个可训练参数。代码通过 `parameter_counts.json` 实际计数，而不是仅引用理论值。原始 Base 为 596049920 参数，加入 adapter 后总参数为 598343680。

PEFT 通常将 LoRA 参数保留为 FP32，Base 是 BF16，Trainer 使用 BF16 混合精度计算。A 随机初始化、B 初始为零时，初始 LoRA 增量为零；第一步 B 可以获得非零梯度，A 的梯度可能为零，随后 B 非零后 A 也能更新。因此不能要求第一步所有矩阵均有非零变化。

冻结 Base 不等于完全不需要反传。梯度仍需通过中间激活传到各层 adapter，所以训练显存高于只加载 BF16 权重。梯度检查点以重算部分前向换取更低的激活保存开销。

## SFT 到底监督什么

SFT 让模型在已知前文时提高标准助手回答中下一个 Token 的概率。令监督标记 m=1 表示助手回答及其 EOS，m=0 表示 system、user、角色边界或 padding，则训练目标为这些受监督位置的交叉熵。

\[
\mathcal L=-\frac{\sum_{i,t}m_{i,t+1}\log p_\theta(x_{i,t+1}\mid x_{i,\le t})}
{\sum_{i,t}m_{i,t+1}}.
\]

代码将不监督的位置 labels 设为 -100，模型内部的因果语言模型损失负责右移对齐。例如 labels 在第 j 个位置是答案 Token，则对应的是第 j-1 个位置的 logits。不要把这理解成模型在看到该 Token 后再复制它。

`encode_sft_record()` 先用与推理完全相同的 `build_messages()` 和 chat template 得到 prompt。再添加标准 assistant 消息取得完整序列，要求完整序列的前缀逐 Token 等于 prompt；不相等时直接报错，不猜测分隔位置。

Non-thinking 的空 think 块属于生成前缀，整个前缀都被屏蔽。只有真实助手解答和其终止 EOS 参与监督。EOS 后的聊天分隔换行不参与训练。`label_audit.jsonl` 保存每题的前缀长度、监督长度、掩码检查和实际受监督解码文本。

## 长样本与 padding

GSM8K 解答中的 `<<公式=结果>>` 是数据集的计算注记。本项目去掉这些注记，保留原始自然语言推导和最终数值答案，处理前后都执行严格判题。

超过 max_length 时整题排除并记录 ID，绝不截掉最终答案还把残缺内容用于 SFT。4 步 smoke 的长度上限是 384；完整本地实验使用 768，实际选中的 1200 题最大长度为 527，因此没有排除训练题。

`AssistantCollator` 对 input_ids 右侧 padding、attention_mask 补 0、labels 补 -100，并按 8 的倍数对齐。即便 pad_token_id 恰好与 EOS 一样，也只屏蔽真正的 padding，不能把回答末尾 EOS 一起屏蔽。

本机 Trainer 的 `_get_num_items_in_batch()` 按非 -100 标签计算累积批次的有效 Token。由于每个样本开头全部属于已屏蔽的 prompt，计数与实际受监督的移位标签相符。实现沿用该版本 Trainer/模型的损失缩放，不额外手动除一次梯度累积步数。

## 配置与实际样本

`configs/sft.yaml` 是 Level 0 smoke：32 条按八个任务类别均衡选择的训练题、16 条独立验证题、4 个优化器步骤、每设备 batch=1、累积 4。它只用于检查闭环，不能当作正式训练的能力结果。

`configs/sft_local.yaml` 是 Level 1 本地实验：全部 1200 条 SFT 题（统计 200、GSM8K 1000），验证损失使用固定的 64 题均衡子集（统计 56、GSM8K 8）。300 步、有效 batch=4 对应一遍完整数据；正式配置使用 cosine 学习率和 20 步 warmup。每 50 步验证并保存检查点，依据验证损失选择最佳 adapter，绝不使用测试集选模型。

RL 数据和官方测试集不会被 SFT 入口读取。训练数据与编码标签各自有指纹。数据、Tokenizer 前处理、配置或依赖改变时，严格恢复要求新建运行，避免把不同实验混成一个连续记录。

## 如何证明训练真的发生

`capture_trainable()` 保存训练前的 adapter 参数副本。`parameter_update_evidence()` 核对训练后每个可训练张量的初始/最终哈希、最大绝对变化和 L2 变化，并检查非 LoRA 参数全部冻结。另抽查第一层 Base 投影权重哈希未变。

4 步真实 smoke 中 224 个 adapter 张量均有变化，最大绝对变化约 0.00040115；损失和梯度范数有限。验证子集的交叉熵从约 2.8660 降到 2.1935，这只说明该损失下降，不能据此宣称数值正确率提高。

保存后重新加载同一固定 Base 和 adapter，逐参数进行精确比较，并比较同一个 32 Token 输入的最后位置 logits。smoke 已通过，logits 最大绝对差为 0。该检查独立于“文件存在”检查。

## 训练日志怎么读

`trainer_log` 行来自 Trainer 的真实回调，包含 loss、grad_norm、learning_rate、epoch 或验证损失。`optimizer_step` 行记录每次实际更新的耗时和 GPU 峰值；不要把两类行当作两次更新。

训练 loss 是近期不同训练批次的平均值，可能上下波动。验证 loss 来自同一固定子集，但它是 teacher forcing 下的 Token 预测损失，不是独立生成的答案正确率。最后若回载了较早的最佳检查点，生成评测和重载证明针对的是该检查点，计算成本则涵盖全部实际执行步骤。

`resource_metrics.json` 中 allocated 是 PyTorch 已分配峰值，reserved 是缓存保留峰值，不等于整张显卡的系统占用。smoke 的峰值 allocated 约 1.68 GiB、reserved 约 2.65 GiB，不能直接当成所有长度下的正式训练峰值。

## 运行、恢复与评测

```powershell
.\.venv\Scripts\python.exe scripts\05_train_sft.py --prepare-only
.\.venv\Scripts\python.exe scripts\05_train_sft.py
.\.venv\Scripts\python.exe scripts\05_train_sft.py --config configs\sft_local.yaml
.\.venv\Scripts\python.exe scripts\05_train_sft.py --config configs\sft_local.yaml --resume <已有SFT运行目录>
.\.venv\Scripts\python.exe scripts\06_evaluate_sft.py --adapter <运行目录中的adapter> --mode non-thinking
.\.venv\Scripts\python.exe scripts\10_analyze_results.py --sft-run <已完成的正式SFT运行目录>
```

尖括号为待替换路径。恢复使用完整 Trainer 检查点，包含 optimizer、scheduler、TrainerState 和 RNG；仅加载 adapter 是权重续训，不等价于严格恢复。

早期开发入口使用192 Token上限及单题seed，只用于历史开发比较。最终统一评测由`scripts/26_evaluate_ablations.py`执行：Base、SFT与RL检查点使用同一400题、Non-thinking、256 Token、temperature=0.7、top_p=1、top_k=0、batch=4，每题一次采样。数值正确、格式有效与截断分别报告，不通过放宽判题补偿。详见[最终报告](../reports/ablation_round1/final_report.md)。

## 推荐代码阅读顺序

先读 `configs/sft.yaml` 与 `sft_local.yaml`，再读 `sft_data.py:encode_sft_record()`、`AssistantCollator`、`lora.py:attach_lora()`、`sft.py:run_sft_cli()`、`callbacks.py`，最后看 `parameter_updates.json`、`reload_verification.json` 和 `06_evaluate_sft.py` 的实际输出。

GRPO 中还会出现 rollout、旧策略概率、组内优势和裁剪等内容，不能用当前 SFT 的交叉熵训练冒充强化学习。
