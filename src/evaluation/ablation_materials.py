"""Render evidence-bound README, resume variants and interview notes."""
from __future__ import annotations

from src.data.storage import read_jsonl
from src.evaluation.storage import append_prediction
from src.utils.artifacts import write_json


def write_project_materials(root, result):
    metrics = result['metrics']
    tr = result['training']['rows']
    c = result['comparisons']['E4_step100_vs_E0_step100']
    s = result['comparisons']['E4_step100_vs_SFT']
    docs = root / 'docs'
    out = root / 'reports/ablation_round1'
    resume = f'''# StatReason-RL简历素材

本轮正式采用版本B。所有成绩来自同一400题、每题一次采样，GSM部分为官方train的reserved题；一训练种子，不声称跨种子稳定性。版本A仅提供性能事实的候选表述，不应用于宣称DAPO显著优于GRPO。

## 简历可直接使用：机制分析版

**Qwen数学推理强化学习：GRPO/DAPO机制复现与训练诊断**

技术栈：PyTorch、Transformers、TRL、PEFT/LoRA、RLVR、SciPy。

项目介绍：面向数学推理任务，在8GB单卡上搭建可验证奖励后训练流程，通过受控消融分析GRPO/DAPO的训练行为与采样成本。

- 基于TRL/PEFT完成Qwen3-0.6B的SFT与GRPO训练，接入Clip-Higher、在线Dynamic Sampling、Token级损失及超长奖励整形，完成五组独立运行、{sum(r['optimizer_steps'] for r in tr.values())}次参数更新。
- 实现严格数值奖励、旧策略概率复用和独立冻结参考策略审计，核验224个LoRA张量更新及检查点重载，完成Vanilla与组合方案的同机短程恢复一致性验证。
- 量化有效训练信号与生成成本：组合方案进入优化的混合奖励组占比为100%，Vanilla为38%；对应生成Token成本为基线的{tr['E4']['total_generated_tokens']/tr['E0']['total_generated_tokens']:.2f}倍，记录补采预算耗尽的提前停止案例。
- 在固定400题上保存12个检查点的4800条回答，结合Wilson区间、题目级配对Bootstrap与McNemar检验分析结果；组合方案与Vanilla准确率分别为21%和19%，差异未获得显著性支持。

口径说明：100%是筛选后进入优化的混合奖励组比例，不是回答正确率；GSM题来自官方训练集预留划分。E2完成30步，其余四组各100步，五组共同30步另作公平比较。详细结果及单种子限制见下文和最终报告。

## 版本A：性能事实候选（本轮不作为首选）

项目名称：Qwen数学推理后训练与GRPO/DAPO实验

技术栈：PyTorch、Transformers、TRL、PEFT/LoRA、RLVR、NumPy/SciPy。

一句话：在8GB显卡上完成小模型可验证奖励后训练，比较数学能力变化与训练机制。

- 构建GSM8K与确定程序校验的统计题流程，实现严格最终答案验证与数据划分、来源指纹及模板族审计。
- 基于TRL/PEFT完成SFT、Vanilla GRPO与四机制DAPO适配，对LoRA更新、旧策略概率、损失归一化、保存重载和恢复进行验证。
- 在固定400题上，SFT为64/400，E4为84/400，点估计提升5.00个百分点；未调整配对Bootstrap 95%区间为[{s['paired_bootstrap_95'][0]*100:.2f},{s['paired_bootstrap_95'][1]*100:.2f}]个百分点。GSM子集由61/200变为81/200，统计子集均3/200。
- 同步报告E4相对Vanilla仅增加2.00个百分点，其95%区间[{c['paired_bootstrap_95'][0]*100:.2f},{c['paired_bootstrap_95'][1]*100:.2f}]跨0；不把该点估计写成DAPO相对GRPO的显著提升。

难点与解决：小模型统计迁移能力差、采样组零方差较多。优先核对验证器、SFT标签和参数更新，再使用对照实验区分能力不足与工程错误；保留未改善的统计结果。

## 版本B：机制分析型（本轮采用）

项目名称：Qwen数学推理强化学习：GRPO/DAPO机制复现与训练诊断

技术栈：Python、PyTorch、Transformers、TRL、PEFT/LoRA、GRPO、RLVR、SciPy、Matplotlib。

一句话：面向小模型数学推理，基于可验证数值奖励复现GRPO/DAPO关键机制，并在8GB消费级显卡上开展真实受控训练与统计分析。

- 基于TRL/PEFT集成Vanilla GRPO与Clip-Higher、在线Dynamic Sampling、Token级损失、超长奖励整形；固定SFT初始化、400题RL池与核心配置，完成五组独立运行及430次实际更新。
- 实现严格数值奖励与逐Token训练审计，核对旧策略概率复用、独立冻结SFT参考、LoRA参数变化及检查点重载；原Vanilla与E4短程恢复均通过同机逐位一致性验证。
- 量化在线筛选收益与成本：E4用于训练的混合奖励组率为100%，对照E0为38%；但生成成本为44,832对16,157 Token。保留E2在30步因连续9组无差异停止的真实案例，分析补采上限与无效组问题。
- 在同一400题上保存12个检查点共4800条回答，使用Wilson区间、题目级配对Bootstrap与McNemar检验分析结果；E4准确率21.0%，Vanilla为19.0%，差异区间跨0，结论重点为机制行为与计算成本。

项目难点与解决方案：

1. 有效组稀缺：在线筛选和补采设次数、Token上限，建立尝试—接受—实际优化步的对应记录；到上限停止，不用零损失假装跳过Adam更新。
2. 训练概率与采样不匹配：统一温度策略采样，修复Transformers将top_p=1回退到模型默认值的问题，保存实际生效配置。
3. 长度归一化与梯度累积：直接对照安装版TRL方法及整批公式检验损失和梯度，防止Token分母或累积缩放错误。
4. 正确率提升不确定：固定30步公平比较，单列可用100步结果及有预算差的成本比较，不以更换测试集或反复调参改善展示。

## 使用边界

不能写“从零手写完整GRPOTrainer”“DAPO显著优于GRPO”“统计推理能力明显改善”“复现熵坍塌”“减少采样46%”或“官方GSM8K test达到40.5%”。本轮没有pass@8结果，没有独立E5长度奖励消融，只有一个训练种子。
'''
    (docs / 'resume_materials.md').write_text(resume, encoding='utf-8')
    guide = r'''# StatReason-RL面试与学习指南

所有内容对应当前真实工程。最终结果与图表见reports/ablation_round1/final_report.md；本轮采用预算受限机制分析路线。

## 架构与代码位置

```mermaid
flowchart TD
 A[固定SFT检查点和RL训练池] --> B[G=4生成回答]
 B --> C[严格奖励与可选长度惩罚]
 C --> D{原始二元奖励有差异？}
 D -->|启用动态筛选且无差异：预算内补采| B
 D -->|有差异或关闭筛选| E[相对优势与TRL裁剪损失]
 E --> F[LoRA更新；旧概率复用两次]
 F --> B
 F --> G[冻结检查点统一评测]
```

| 内容 | 代码 |
|---|---|
| 统计题与独立标准答案校验 | src/data/statistics.py、quality.py |
| 严格最终答案解析 | src/rewards/numeric.py |
| SFT标签mask、样本选择 | src/training/sft_data.py、sft.py |
| RL-only训练池 | src/training/rl_data.py |
| 当前/旧/参考概率与逐Token审计 | src/training/grpo_trainer.py |
| 独立目标公式 | src/training/grpo_math.py |
| 在线补采、长度惩罚、生成配置冻结 | src/training/dapo_mechanisms.py |
| 统一训练与检查点 | src/training/grpo.py；scripts/07_train_grpo.py |
| 安静的本地训练入口 | scripts/24_run_ablations.py |
| 冻结评测与恢复 | src/evaluation/ablation_protocol.py；scripts/26_evaluate_ablations.py |
| 统计、图表与交付 | src/evaluation/ablation_analysis.py；scripts/27_finalize_ablation_report.py |

## 核心公式

组内优势：$A_i=(r_i-\bar r)/(s_r+10^{-4})$。这里使用样本标准差，匹配TRL 0.23.1。

重要性比率：$\rho_{it}=\exp(\log\pi_\theta-\log\pi_{old})$。

Token损失：$\ell_{it}=-\min(\rho_{it}A_i,\operatorname{clip}(\rho_{it},1-\epsilon_l,1+\epsilon_h)A_i)+\beta k_{it}$。

参考KL估计：$k_{it}=e^{d_{it}}-d_{it}-1$，$d_{it}=\log\pi_{ref}-\log\pi_\theta$。

回答归一化：$B^{-1}\sum_i(\sum_t m_{it}\ell_{it})/(\sum_t m_{it})$。

Token归一化：$(\sum_{it}m_{it}\ell_{it})/(\sum_{it}m_{it})$。微批共用整优化器批次的有效Token分母，不能再多除一次累积步数。

长度惩罚：$-c\min(1,\max(0,(L-L_{soft})/(L_{max}-L_{soft})))$。本轮soft=96、max=128，长度含EOS；截断损失mask在各组统一关闭。

## 约2分钟项目介绍

我做的是Qwen小模型数学推理强化学习项目，重点是可验证奖励和GRPO/DAPO机制分析。项目在RTX 5060 Laptop 8GB显卡上运行，使用Qwen3-0.6B和LoRA，训练器基于TRL集成。我负责数值奖励、数据划分、在线动态采样、长度奖励，以及概率和参数更新的审计，没有把框架功能说成从零实现。

数据包括GSM8K和程序生成、独立计算校验的统计题。早期发现统计题正确率低，我先排查了SFT标签、Chat Template、答案解析和检查点加载，确认主要问题在能力和迁移，随后把目标收敛为受控机制实验。

我固定同一SFT和400题RL池，独立运行五组。四组完成100步，Dynamic Sampling单独组在30步遇到连续9个无奖励差异组而停止，我保留了这次结果。组合组用于优化的有效组比例达到100%，但总生成Token是Vanilla的约2.77倍，说明筛选有效不等于计算更省。

最终在同一400题上评测12个检查点。组合组答对84题，Vanilla答对76题，但配对区间跨0，所以我没有宣称DAPO显著更好。项目展示的是实现正确性、真实机制行为、计算成本和统计解释。统计题表现仍弱、单种子以及成本匹配不完全相等，都是我会明确说明的局限。

## 20个技术问题及参考回答

### 1. 为什么选这个项目？

它补充我RAG项目之外的大模型后训练能力，也能结合统计背景研究实验公平性和不确定性。目标是解释真实训练行为，不以漂亮分数作为唯一验收标准。

### 2. 哪些是自己实现的，哪些来自框架？

模型来自Qwen，LoRA来自PEFT，Rollout、裁剪和优化流程主要基于TRL。项目实现严格奖励、RL数据边界、独立参考策略接入、在线补采、长度奖励、逐Token审计、损失公式校验与实验统计。不能说整个Trainer从零手写。

### 3. GRPO为什么不需要价值模型？

它对同一道题采样多个回答，用组内奖励均值作基线，用相对奖励构造优势，因此不需要单独学习状态价值。代价是同组奖励没有差异时，正确性策略梯度信号会消失。

### 4. 同组回答全部正确或全部错误会怎样？

组内中心化后优势为0。本项目beta=0.04，仍可能有KL梯度，不能说整个更新一定为零。E0保留这些组，E2/E4按原始二元正确性过滤并补采。

### 5. 旧策略概率和参考策略概率有什么区别？

旧概率来自生成本批回答的策略，在该批两次更新期间固定，用于重要性比率。参考概率来自独立冻结的初始SFT副本，用于KL约束。二者不是同一个概念。

### 6. 为什么使用独立SFT参考，而不是关闭LoRA？

关闭适配器会回到原始Base，不是SFT初始化。项目单独加载同一SFT并冻结，检查无共享参数、无梯度、哈希不变，避免错误的KL目标。

### 7. 如何证明模型真的更新了？

保存训练前后LoRA张量哈希、delta范数、梯度范数，224个张量实际变化；重载后参数和探针输出一致。还实测Vanilla从20恢复24步、E4从2恢复4步，与连续训练逐位一致，范围限于同机固定环境。

### 8. Clip-Higher具体改变什么？

把上裁剪epsilon从0.20变成0.28，下侧保持0.20。对正优势、比率1.25的Token，Vanilla会截断梯度，新上界允许继续更新；负优势在这个上侧区域的行为不同。测试专门核对优势符号与梯度。

### 9. 你观察到它缓解熵坍塌了吗？

没有足够证据。E0上侧有效裁剪约0.0217%，E1约0.0096%，触发非常少。只有两次旧策略复用且学习率较低，比例通常接近1。不能从配置开启或一条熵曲线推断缓解坍塌。

### 10. 在线Dynamic Sampling怎么实现？

先生成G=4完整回答组，按原始0/1奖励判断是否混合。无效时从同一RL池取下一个候选，在次数和Token预算内补采。每次尝试记录题目、奖励、成本、接受状态，并通过rollout_id与实际优化微批对应。

### 11. 为什么E2在30步停了？

最后连续9个候选组全对或全错，达到既定补采上限。总共62次尝试，仅15组进入两次更新，完成30步。它不是显存崩溃，也不是没有任何更新；这是有效组稀缺与预算上限的真实结果。

### 12. 为什么不用零损失跳过无效更新？

Adam有动量，即使当前梯度为零仍可能改变参数。当前适配选择在未填入有效组时停止于优化器调用之前，保留已完成结果，避免把无信号更新计为有效训练。

### 13. 动态采样更省计算吗？

本轮没有。E4训练接受组的有效率100%，但需要139次候选尝试获得50组，总生成44,832 Token；Vanilla为16,157。它提高每次优化使用有效组的比例，却付出了额外采样成本。

### 14. Token级损失为什么可能影响长度？

回答平均让每个回答总权重相近，短回答的单Token权重更大；全局Token平均让单Token权重更一致，长回答总权重更高。归一化确实改变了梯度，但本轮结果不足以证明它单独稳定改善长度或正确率。

### 15. 梯度累积的分母怎么验证？

当前batch size=1、累积4次，整步恰含一道题的4个回答。Token损失每个微批共用整步Token数；直接调用安装版TRL方法，与不同长度、Padding的整批公式比较损失和梯度。GPU微批还独立复算目标，误差记录通过。

### 16. 超长惩罚与截断掩码有什么区别？

前者改变奖励，进而改变优势；后者让部分Token不参与损失。各组截断掩码统一关闭，E4增加soft=96到max=128的线性惩罚，53个接受回答实际触发。没有独立E5，因此不能把组合收益归于这一个机制。

### 17. 奖励如何防止取巧？

只解析唯一的末行FINAL_ANSWER标量，支持受限合法数值、合理容差和语义允许的百分数；拒绝冲突标记、NaN/Inf、非法表达式与截断，不执行模型表达式，也不从推理中任意找正确数字。boxed仅做诊断。

### 18. 为什么后来统一top_p=1、top_k=0？

为了让温度缩放后的完整策略分布同时用于采样和log probability，减少过滤分布不匹配。还发现HF会把GenerationConfig的默认值回退成模型top_p=0.95，专门冻结并核对实际生效配置，保留受影响运行但不用于公平比较。

### 19. 怎样设计公平评测并解释结果？

五组从同一SFT独立开始，模型、RL池、优化器、模板和判题一致。E2只有30步，所以五组统一比较第30步，四个100步终点另列。成本比较只按Token选检查点，报告未用预算差。400题按题目配对Bootstrap，不能把12个模型的4800回答当作独立题目；还报告Wilson、McNemar和多重比较校正。

### 20. 最终效果不好，项目还有价值吗？

有，但必须有正确实现和真实机制证据。E4相对SFT有点估计改善，但相对Vanilla的差异区间跨0；统计题仍只有3/200，单种子也限制结论。因此采用机制分析型展示，强调真实更新、筛选成本、罕见裁剪与长度奖励触发，明确未解决的领域能力问题，不继续用最终测试调参。
'''
    (docs / 'interview_guide.md').write_text(guide, encoding='utf-8')
    header = '''# StatReason-RL

面向数学推理的Qwen小模型可验证奖励强化学习与GRPO/DAPO机制分析。

项目基于TRL、Transformers和PEFT，在RTX 5060 Laptop 8GB上完成SFT、受控GRPO/DAPO训练与统一评测。重点研究有效奖励组、裁剪行为、长度归一化和采样成本。训练器基于TRL集成，数据、验证、在线补采、长度整形及审计由项目实现。

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
'''
    lines = [header.rstrip()]
    for name in ['Base', 'SFT', 'E0_step100', 'E1_step100', 'E2_step30', 'E3_step100', 'E4_step100']:
        m = metrics[name]
        lines.append(f"| {name} | {m['strict_correct_count']}/400 ({m['sampled_pass_at_1']*100:.2f}%) | {m['by_source']['openai/gsm8k']['strict_correct_count']} | {m['by_source']['statreason_synthetic']['strict_correct_count']} |")
    lines += ['', f"E4相对Vanilla点估计+2.00个百分点，题目级配对Bootstrap 95%区间[{c['paired_bootstrap_95'][0]*100:.2f},{c['paired_bootstrap_95'][1]*100:.2f}]跨0，不能宣称DAPO显著优于GRPO。E4相对SFT点估计+5.00个百分点，未调整区间[{s['paired_bootstrap_95'][0]*100:.2f},{s['paired_bootstrap_95'][1]*100:.2f}]；同时报告多重比较校正，单种子不支持跨seed稳健结论。", '',
              'E4接受训练的有效组比例100%，E0为38%，但总生成Token为44,832与16,157，约2.77倍。E1上侧有效裁剪仅约0.0096%，E4有53个接受回答触发长度惩罚。统计正确率仍低，实验结论采用机制分析路线，不包装成全面数学能力提升。', '',
              '![真实训练奖励](reports/ablation_round1/figures/01_reward.png)',
              '![有效组与成本](reports/ablation_round1/figures/03_effective_groups.png)',
              '![GSM留出准确率](reports/ablation_round1/figures/06_gsm8k_accuracy.png)',
              '![配对区间](reports/ablation_round1/figures/11_paired_intervals.png)', '',
              '完整12检查点表、训练与成本曲线、Wilson区间、配对Bootstrap、McNemar与Holm校正见[实验报告](reports/ablation_round1/final_report.md)。完整原始回答保存在本地outputs/ablation_evaluation_round1，训练日志与代码快照保存在本地outputs/runs，未随公开仓库分发。', '',
              f"成本口径：汇总图与CSV的检查点Token列只累计至对应检查点；完整运行成本还包含后续尝试。E2第30步为{result['models']['E2_step30']['training_generated_tokens']:,} Token，加上停止前未进入新更新的{tr['E2']['total_generated_tokens']-result['models']['E2_step30']['training_generated_tokens']:,} Token补采，合计{tr['E2']['total_generated_tokens']:,} Token。", '',
              '## 复现', '',
              '环境：Windows、Python 3.13、PyTorch 2.8.0+cu128、Transformers 4.57.1、TRL 0.23.1、PEFT 0.17.1、Accelerate 1.10.1。依赖锁文件为requirements.lock.windows-py313-cu128.txt；安装与基础训练见docs/training_guide.md。模型权重和虚拟环境不提交。', '',
              '**产物范围：** 公开仓库提供代码、配置、报告、CSV、结果JSON、图表和少量错误样例。完整处理后数据、评测JSONL、冻结计划、训练日志与模型/适配器检查点保留本地。克隆仓库即可阅读结果；执行完整审计、重新评测或重建报告，需要先准备对应原始产物，不能仅凭公开结果JSON重建审计证据。以下命令面向原实验的完整本地环境；新机器准备步骤见[运行指南](docs/training_guide.md)。', '',
              '```powershell',
              '.venv\\Scripts\\python.exe scripts/07_train_grpo.py --config configs/ablations/E0_100.yaml',
              '.venv\\Scripts\\python.exe scripts/24_run_ablations.py --status',
              '.venv\\Scripts\\python.exe scripts/26_evaluate_ablations.py --execute',
              '.venv\\Scripts\\python.exe scripts/27_finalize_ablation_report.py', '```', '',
              '训练日志自动保存，批量运行入口提供中文进度；评测可中断后用同一命令续跑。已有完整运行会保留，配置、代码、数据或权重改变时应建立新实验版本。完整运行与迁移说明见[运行指南](docs/training_guide.md)。', '',
              '## 验证与局限', '',
              '覆盖严格奖励边界、不同长度和Padding损失、实际TRL梯度累积、补采预算与停止、生成配置回退、冻结测试ID及配对统计。所有正式运行有224个LoRA张量变化与重载证据；Vanilla与E4短程恢复通过同机逐位一致性校验。', '',
              '局限包括单训练种子、每题单样本、较弱统计迁移、E2提前停止、成本匹配存在离散剩余、缺少独立E5长度消融，以及共享模板相关性。没有证实熵坍塌，也没有证明Clip-Higher单独改善探索。项目达到预算受限机制分析目标后停止扩展，不用最终测试集继续优化。', '',
              '参考：[TRL 0.23.1](https://huggingface.co/docs/trl/v0.23.1/grpo_trainer)、[Open-R1](https://github.com/huggingface/open-r1)、[Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math)、[DAPO官方项目](https://dapo-sia.github.io/)、[论文](https://arxiv.org/abs/2503.14476)。']
    previous = root / 'README.md'
    backup = out / 'README_before_finalization.md'
    if previous.exists() and not backup.exists():
        backup.write_text(previous.read_text(encoding='utf-8'), encoding='utf-8')
    previous.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    # Human-reviewed examples are diagnostic illustrations, not counts for all errors.
    chosen = {
        'stat-test-fbf3c099c73ae7f90612': '没有正确利用指示变量平方仍为自身；最终数值错误，推导不足以细分所有原因。',
        'stat-test-a3c0f37f5da79b93b45e': '读题与建模错误，相关样本均值的协方差项处理错误，兼有数值问题。',
        'stat-test-b742f81b1c4b1ab5524d': '中间矩与最终减法均有错误；不能归为单纯格式问题。',
        'stat-test-19b72a3ee065e8a467ce': '把后验概率写成事件交与事件并的比值，属于公式/建模错误。',
    }
    rows = read_jsonl(root / 'outputs/ablation_evaluation_round1/SFT/evaluation_predictions.jsonl')
    case_path = out / 'error_cases.jsonl'
    case_path.write_text('', encoding='utf-8')
    for row in rows:
        if row['id'] in chosen:
            append_prediction(case_path, {**row, 'qualitative_review': chosen[row['id']], 'scope': 'representative_SFT_examples_not_exhaustive_semantic_counts'})
    write_json(out / 'materials_identity.json', {'selected_resume_version': 'B', 'score_source': 'reports/ablation_round1/results.json',
                                                'interview_question_count': 20, 'E4_vs_E0_delta_pp': c['second_minus_first_accuracy']*100,
                                                'E4_vs_E0_bootstrap_95_pp': [x*100 for x in c['paired_bootstrap_95']]})
