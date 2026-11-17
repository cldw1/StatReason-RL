# 架构与代码阅读顺序

正式实验已完成E0–E4五组独立运行：四组各100步，E2在30步达到补采上限，共430次RL参数更新。统一评测覆盖12个检查点、同一400题。当前结论以[最终报告](../reports/ablation_round1/final_report.md)为准；早期8题冒烟和100题开发实验单独保留。

## 当前真实调用

环境：`scripts/01_check_env.py` → `environment.py` → CUDA/BF16/SDPA/AdamW 小张量检查、包/API 审计 → `artifacts.py` 写入运行记录。

配置：`scripts/00_validate_configs.py` → `config.py:load_config()` → `deep_merge()` → `validate_config()`。消融配置继承 GRPO，GRPO 再继承模型配置。正式实验缺少模型 revision、数据指纹或共同 SFT 指纹时拒绝运行。

数据：`scripts/03_prepare_data.py` → `statistics.py:generate_statistics()` 或 `gsm8k.py:prepare_gsm8k()` → `schema.py:validate_records()` → `quality.py:audit_statistics()`（统计题）→ `storage.py:save_dataset()`。

判题：`numeric.py:score_completions()` → `completion_text()` → `verify_completion()` → `parse_numeric()`。`scripts/03_audit_rewards.py` 保存构造攻击样例的实际判题结果。

## 模块状态

| 模块 | 当前职责与状态 |
|---|---|
| src/utils | 已实现配置继承、严格 JSON、SHA256、run_id、环境检查 |
| src/data/statistics.py | 已实现七题型、28 个正式模板族和确定性答案 |
| src/data/quality.py | 已实现独立复算和训练样本抽查材料 |
| src/data/gsm8k.py | 已实现固定版本的官方train获取、答案解析、来源指纹与题组隔离 |
| src/data/schema.py | 已实现字段、去重、划分和模板族校验 |
| src/data/storage.py | 已实现 JSONL、内容指纹、逐文件哈希和版本保护 |
| src/rewards/numeric.py | 已实现严格标量 completion 判题与独立诊断字段 |
| src/models | 已实现固定快照、统一提示、加载生成、LoRA和独立冻结SFT参考 |
| src/training | 已实现SFT掩码、GRPO概率与参数审计，集成Clip-Higher、在线补采、Token损失与超长奖励；五组均完成真实参数更新 |
| src/evaluation | 已实现冻结评测、逐题保存恢复、Wilson、配对Bootstrap/McNemar/Holm及11张实验图表 |

建议阅读顺序：`configs/data.yaml` → `scripts/03_prepare_data.py` → `statistics.py` → `quality.py` → `schema.py` → `storage.py` → `numeric.py` → 两个数据/判题测试文件。公式与判题细节见 `docs/data_and_verifier.md`。

模型生成入口会明确切除 prompt Token 后再调用判题；不提供返回成功却未执行真实训练的占位入口。

当前模型调用：`04_evaluate_base.py` → `runner.run_cli()` → `panel.select_development_panel()` → `snapshot.resolve_snapshot()` → `inference.load_qwen()` → `generate_completion()` → `extract_completion()` → `numeric.verify_completion()` → JSONL/CSV 与 `metrics.summarize_predictions()`。

模型代码导读见[模型与评测](model_and_evaluation.md)。真实目标模型已完成Base、SFT及RL检查点的统一评测；随机微型模型功能测试独立保存，不作为能力结果。

SFT 调用：`05_train_sft.py` → `sft.run_sft_cli()` → `sft_data.prepare_sft_data()`/`encode_sft_record()` → `lora.attach_lora()` → Transformers Trainer → `LocalTrainingCallback` → 参数更新证据、adapter 保存与重载校验。SFT 生成评测由 `06_evaluate_sft.py` 复用相同开发题组入口，先校验 adapter 与 Base 版本的绑定。数学与标签细节见 `docs/lora_sft_explained.md`。

GRPO调用：`07_train_grpo.py` → `grpo.run_grpo_cli()` → `rl_data.prepare_rl_data()` → 两个独立SFT实例 → `AuditedGRPOTrainer` → `_generate_and_score_completions()` → 固定旧/参考概率和组内优势 → TRL损失/独立数学核对 → 梯度累积与LoRA更新 → adapter重载。`14_analyze_grpo.py`分析真实训练与统一评测，`15_verify_grpo_resume.py`在周期边界验证续训。详见`docs/grpo_explained.md`。

DAPO适配在`src/training/dapo_mechanisms.py`及Trainer接入层执行。`24_run_ablations.py`管理批量训练入口，`25_analyze_ablation_training.py`审计训练并准备比较计划，`26_evaluate_ablations.py`执行冻结评测，`27_finalize_ablation_report.py`审计原始产物并生成报告。公开仓库提供代码和结果汇总；完整原始运行目录与检查点保留本地，复算前置条件见[运行指南](training_guide.md)。
