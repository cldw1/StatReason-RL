# StatReason-RL

基于 **Qwen3-0.6B** 的数学推理可验证奖励强化学习项目：在 **RTX 5060 Laptop 8GB** 上完成 LoRA SFT、GRPO 与 DAPO-style 受控实验。

项目基于TRL、Transformers和PEFT，在RTX 5060 Laptop 8GB上完成SFT、受控GRPO/DAPO训练与统一评测。重点研究有效奖励组、裁剪行为、长度归一化和采样成本。训练器基于TRL集成，数据、验证、在线补采、长度整形及审计由项目实现。

## 背景与目标

小模型数学推理训练可能出现同题回答奖励全部相同、有效策略优势稀缺的问题；在线补采、裁剪与长度处理又会改变样本分布和计算成本。本项目固定 SFT 初始化与训练池，通过五组对照区分算法机制、能力变化和采样开销，并保留未改善及提前停止的结果。

**主要结果：** 同一400题上，SFT、Vanilla GRPO、四机制组合的严格正确率分别为 **16%、19%、21%**。组合方案较 SFT 高 **5 pts**、较 GRPO 高 **2 pts**（点估计）；与 GRPO 的差异未获统计显著性支持。GSM题来自官方 train 的预留划分。

## 系统与实现

| 模块 | 代码与职责 |
|---|---|
| 数据 | src/data：GSM8K处理、统计题生成、独立数值校验、分组隔离 |
| 奖励 | src/rewards/numeric.py：唯一最终标量、容差、格式与截断防护 |
| SFT | src/training/sft*.py：助手标签mask、LoRA、检查点验证 |
| GRPO审计 | grpo_trainer.py、grpo_math.py：旧/参考概率、损失与Token指标 |
| DAPO适配 | dapo_mechanisms.py：在线补采、长度奖励、非对称裁剪与Token损失接入 |
| 评测统计 | src/evaluation：冻结协议、逐题输出、配对统计与图表 |

固定SFT → G=4 Rollout → 严格奖励 → 可选在线筛选/补采 → 相对优势 → TRL裁剪目标与KL → LoRA更新 → 冻结评测。每批旧概率复用两次，参考策略为独立冻结的同一SFT副本。

- **GRPO：** 同题生成4个回答，组内标准化优势，固定旧策略概率复用两次更新，并以独立冻结的SFT模型提供KL参考。
- **Clip-Higher：** 将上侧裁剪epsilon从0.20调至0.28，下侧保持0.20。
- **Dynamic Sampling：** 依据原始二元正确性在线筛除无差异组，在次数与Token预算内补采；预算耗尽则在下一次优化器更新前停止。
- **Token损失与长度奖励：** 使用优化器批次全局有效Token分母，核对梯度累积缩放；对超过96 Token的回答施加线性惩罚，训练长度上限128 Token。

LoRA配置为r=8、alpha=16，作用于q/k/v/o投影，共2,293,760个可训练参数。使用BF16、SDPA、梯度检查点和微批梯度累积适配单卡；未采用量化、vLLM或分布式训练。

## 数据和协议

原v1统计题900条、28个模板族，按模板族划分SFT/RL/validation/test，标准答案由确定程序计算并独立校验。GSM8K官方train共7473条，固定版本与来源指纹。主SFT使用GSM1000条、统计200条，训练300步后导出验证loss选中的150步。RL共同池400题：GSM300、统计100。

最终评测400题：封存统计200题全部保留，另含GSM8K train的reserved题200道，**不是官方GSM8K test成绩**。相关数学结构出现在历史扩展开发中，不宣称完全未知结构泛化。原100题只作开发使用。最终评测后停止调参。

所有模型使用同一官方Non-thinking模板、256 Token、temperature=0.7、top_p=1、top_k=0、batch=4，每题采样一次。严格正确率、数值正确率、格式有效率、截断与成本分开记录。没有pass@8结果。

## 实验设计

| 组别 | 变化 | 实际更新步数 |
|---|---|---:|
| E0 | Vanilla GRPO，回答归一化，对称裁剪 | 100 |
| E1 | 仅上裁剪epsilon 0.20→0.28 | 100 |
| E2 | 仅在线Dynamic Sampling | 30，达到预定补采上限停止 |
| E3 | 仅全局有效Token归一化 | 100 |
| E4 | 四机制组合，包括超长奖励 | 100 |

各组固定初始SFT、RL池、LoRA、优化器、学习率1e-5与KL系数0.04；E4保留共同KL，是受控四机制适配，不是论文去KL的大规模配方。五组共同30步公平比较，四个100步终点另列。生成成本受限比较按Token而非准确率选择保存检查点，明确离散预算差。

## 真实结果

| 模型 | 全部400题 | GSM train-reserved/200 | 统计/200 |
|---|---:|---:|---:|
| Base | 9/400 (2.25%) | 8 | 1 |
| SFT | 64/400 (16.00%) | 61 | 3 |
| E0_step100 | 76/400 (19.00%) | 75 | 1 |
| E1_step100 | 80/400 (20.00%) | 78 | 2 |
| E2_step30 | 78/400 (19.50%) | 75 | 3 |
| E3_step100 | 81/400 (20.25%) | 75 | 6 |
| E4_step100 | 84/400 (21.00%) | 81 | 3 |

E4相对Vanilla点估计 **+2.00 pts**，题目级配对Bootstrap 95%区间[-1.25,5.25] pts跨0，不能宣称DAPO显著优于GRPO。E4相对SFT点估计 **+5.00 pts**，未调整区间[1.75,8.50] pts；14项探索性比较的Holm校正后p=0.0717，单种子不支持跨seed稳健结论。

E4接受训练的有效组比例100%，E0为38%，但总生成Token为44,832与16,157，约2.77倍。E1上侧有效裁剪仅约0.0096%，E4有53个接受回答触发长度惩罚。统计正确率仍低，实验结论采用机制分析路线，不包装成全面数学能力提升。

![真实训练奖励](reports/ablation_round1/figures/01_reward.png)
![有效组与成本](reports/ablation_round1/figures/03_effective_groups.png)
![GSM留出准确率](reports/ablation_round1/figures/06_gsm8k_accuracy.png)
![配对区间](reports/ablation_round1/figures/11_paired_intervals.png)

完整12检查点表、训练与成本曲线、Wilson区间、配对Bootstrap、McNemar与Holm校正见[实验报告](reports/ablation_round1/final_report.md)。完整原始回答保存在本地outputs/ablation_evaluation_round1，训练日志与代码快照保存在本地outputs/runs，未随公开仓库分发。

成本口径：汇总图与CSV的检查点Token列只累计至对应检查点；完整运行成本还包含后续尝试。E2第30步为17,876 Token，加上停止前未进入新更新的2,733 Token补采，合计20,609 Token。

## 项目结构

| 路径 | 内容 |
|---|---|
| `configs/` | 模型、数据、SFT、GRPO及E0–E4实验配置 |
| `src/data/`、`src/rewards/` | 数据生成/划分、独立答案校验与严格数值奖励 |
| `src/models/`、`src/training/` | 模型加载、LoRA、SFT、GRPO和DAPO-style机制 |
| `src/evaluation/` | 冻结评测、逐题保存、诊断与配对统计 |
| `scripts/` | 数据准备、训练、评测与报告入口 |
| `tests/` | 判题、损失、采样预算与评测协议检查 |
| `docs/` | 架构、算法协议和运行指南 |
| `reports/ablation_round1/` | 真实结果JSON/CSV、实验报告、11张图及错误样例 |

## 安装与数据准备

环境：Windows、Python 3.13、PyTorch 2.8.0+cu128、Transformers 4.57.1、TRL 0.23.1、PEFT 0.17.1、Accelerate 1.10.1。依赖锁文件为requirements.lock.windows-py313-cu128.txt；安装与基础训练见docs/training_guide.md。模型权重和虚拟环境不提交。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/00_validate_configs.py
.\.venv\Scripts\python.exe scripts/03_prepare_data.py --statistics full --gsm8k
```

模型固定为`Qwen/Qwen3-0.6B`的指定revision，见[模型配置](configs/model.yaml)。默认缓存为`D:/AIModels/HuggingFace`，新机器可按实际磁盘修改`model.hf_home`。GSM8K只获取官方train；统计题按模板族划分，不以模型答题成败筛选数据。

## 训练与评测

以下是已有模型缓存及数据时的SFT入口；新机器缺少缓存时，先按[运行指南](docs/training_guide.md)下载模型。

```powershell
.\.venv\Scripts\python.exe scripts/05_train_sft.py --config configs/sft_local.yaml
```

SFT完成后，从新生成的`adapter_metadata.json`读取真实适配器路径与哈希，更新`configs/grpo.yaml`的共同初始化；数据指纹也需与实际准备版本一致。配置保留原实验身份，不能将缺失的历史检查点视为已下载，也不能伪造哈希绕过检查。

**产物范围：** 公开仓库提供代码、配置、报告、CSV、结果JSON、图表和少量错误样例。完整处理后数据、评测JSONL、冻结计划、训练日志与模型/适配器检查点保留本地。克隆仓库即可阅读结果；执行完整审计、重新评测或重建报告，需要先准备对应原始产物，不能仅凭公开结果JSON重建审计证据。以下命令面向原实验的完整本地环境；新机器准备步骤见[运行指南](docs/training_guide.md)。

```powershell
.venv\Scripts\python.exe scripts/07_train_grpo.py --config configs/ablations/E0_100.yaml
.venv\Scripts\python.exe scripts/24_run_ablations.py --status
.venv\Scripts\python.exe scripts/26_evaluate_ablations.py --execute
.venv\Scripts\python.exe scripts/27_finalize_ablation_report.py
```

训练日志自动保存，批量运行入口提供中文进度；评测可中断后用同一命令续跑。已有完整运行会保留，配置、代码、数据或权重改变时应建立新实验版本。完整运行与迁移说明见[运行指南](docs/training_guide.md)。

## 验证与局限

覆盖严格奖励边界、不同长度和Padding损失、实际TRL梯度累积、补采预算与停止、生成配置回退、冻结测试ID及配对统计。所有正式运行有224个LoRA张量变化与重载证据；Vanilla与E4短程恢复通过同机逐位一致性校验。

局限包括单训练种子、每题单样本、较弱统计迁移、E2提前停止、成本匹配存在离散剩余、缺少独立E5长度消融，以及共享模板相关性。没有证实熵坍塌，也没有证明Clip-Higher单独改善探索。项目达到预算受限机制分析目标后停止扩展，不用最终测试集继续优化。

参考：[TRL 0.23.1](https://huggingface.co/docs/trl/v0.23.1/grpo_trainer)、[Open-R1](https://github.com/huggingface/open-r1)、[Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)、[DAPO官方项目](https://dapo-sia.github.io/)、[论文](https://arxiv.org/abs/2503.14476)。
