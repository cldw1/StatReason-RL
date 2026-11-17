# 模型加载、Tokenizer 与开发评测

## 当前入口

`scripts/02_model_inference.py` 默认只测试一题，用于模型加载与模板探测。`scripts/04_evaluate_base.py` 默认使用八道固定验证题：七类统计题各一道，再加一道 GSM8K。二者调用 `src/evaluation/runner.py:run_cli()`，不会读取官方测试集。

首次下载并运行两种模式：

```powershell
.\.venv\Scripts\python.exe scripts\04_evaluate_base.py --download-model --mode both
```

这条命令在本人的终端执行，因为当前受限执行环境不能连接本机代理。它下载公开模型、不调用付费 API，不启动云 GPU。下载前查询官方文件大小，选定文件总大小超过 2 GiB 或大小不完整时停止。之后生成最多 8题×2模式×192 Token；单条生成还有 60 秒的软时间上限，不承诺未经实测的完成时间。

只下载不生成：

```powershell
.\.venv\Scripts\python.exe scripts\04_evaluate_base.py --download-model --prepare-only
```

缓存已有完整模型时，无需下载参数：

```powershell
.\.venv\Scripts\python.exe scripts\04_evaluate_base.py --mode both
```

## 快照与版本

`src/models/snapshot.py:resolve_snapshot()` 先检查指定 commit、本地 refs/main 或唯一完整快照。多个版本且没有明确引用时拒绝自行挑选。下载只允许指定的 Qwen3 仓库与必要配置、Tokenizer 和 safetensors 文件，不开启 trust_remote_code，不下载不安全的权重 pickle。

下载先将 main 解析为完整 commit，再下载该 commit。完成后记录权重、Tokenizer、chat template 和配置文件的真实 SHA256，保存到运行目录及 `data/metadata/model_snapshot.json`。模型名相同但权重文件不同会改变实验标识。

`load_qwen()` 仅从已解析的本地快照加载，显式 local_files_only=True。使用 `dtype=torch.bfloat16`、SDPA、单张 CUDA GPU 和 eval 模式，不自动 offload 到 CPU 后继续报告 GPU 速度。

本项目 GPU 测试观察到：先建 FP32 Qwen 再整体 `.to(dtype=BF16)` 会把非持久 RoPE inv_freq buffer 也转成 BF16，而重新加载时该 buffer 为 FP32，输出可能不同。代码使用 Hugging Face 的 dtype 初始化/加载路径，随后只移动 device，保留 FP32 RoPE 频率。该结论来自当前版本微型架构实测，不替代目标预训练权重的完整检查。

## Tokenizer 和 Chat Template

Tokenizer 将文字映射成 Token ID；chat template 按模型约定把 system/user/assistant 角色和生成起点串联起来。不能随意把训练时的一段纯文本格式与评测时的聊天格式混用。

`prompts.py:build_messages()` 只读取 record.question，不将 answer 或 solution 放进提示。系统提示要求简短计算过程和唯一最后答案行。`generate_completion()` 调用 `apply_chat_template(..., add_generation_prompt=True, enable_thinking=...)`。

控制器的 mode 参数逐次决定 enable_thinking；每条预测保存真实布尔值。两种模式在开发对比中使用相同温度 0.7、top_p 0.8、top_k 20 和长度上限，仅改变模板的思考开关。因此它是一个受短预算约束的开发对照，不是各模式独立调优后的官方基准复现。

官方模型卡给出了各模式的建议采样设置。后续根据验证集实测选择并冻结统一评测配置，不能把两模式八题结果作为稳定能力结论。

## 只判生成的回答

decoder-only 模型 generate 返回的序列通常为 prompt Token 加新生成 Token。代码首先核对返回前缀，再由 `extract_completion()` 按实际 prompt_length 切片。保存原始新 Token、含特殊 Token 的 raw_completion 和去特殊 Token 的 completion，验证器只接收最后这一项。

如果在 prompt 中出现了正确答案标记，而模型没有生成有效答案，奖励仍是零。测试直接检查了这一调用关系。

EOS 与截断分开判断：扫描新 Token 中的第一个 EOS 并去掉其后的填充；EOS 恰好位于长度上限时正常结束。没有 EOS 且达到长度上限，标记 max_new_tokens 截断；因时间等原因更早停止且无 EOS，标记 stopped_without_eos 并严格判错。生成 Token 成本包含 EOS，另保存不含 EOS 的 text_token_count。

## 题组、指标和复现

`panel.py` 只读取两个数据版本的 validation.jsonl，并核对文件哈希。固定 seed 选择题目，记录 panel_fingerprint、全部选中 ID 与源数据指纹。不能通过此入口传入 test 路径。

`example_seed()` 从总 seed、题目 ID 和 sample_index 派生单题 seed，使处理顺序变化或恢复已完成记录时不改变该题的初始随机状态。它不承诺不同设备、驱动或内核下按位相同。

每题每模式只生成一次。随机采样指标命名 sampled_pass_at_1；若配置改为贪心，则单列 greedy_accuracy。两种模式分别聚合，拒绝把重复回答误算成独立题目。报告格式有效率、数值正确率、严格正确率、解析失败率、截断率、Token、耗时和 Wilson 区间。八题区间只描述此小题组，不能用于宣称一般测试能力或训练 seed 波动。

## 保存与中断恢复

每次运行记录：resolved_config.yaml、environment.json、selection.json、model_snapshot.json、evaluation_identity.json、model_load.json、evaluation_predictions.jsonl/CSV、evaluation_summary.json、resource_metrics.json 和 run_status.json。仅生成阶段成功后才有对应预测；缺权重时只保存阻塞状态，不创建假准确率。

每完成一题追加 JSONL 并 flush/fsync；CSV 和汇总从这些记录导出。恢复示例：

```powershell
.\.venv\Scripts\python.exe scripts\04_evaluate_base.py --mode both --resume <已有评测run目录>
```

尖括号为待替换值。恢复核对模型、数据、提示、依赖和评测配置的实验标识；发生变化时要求新建运行。可修复最后一条不完整 JSON/UTF-8 记录，并先保留原始中断备份，禁止跳过中间损坏记录。已完成的题不会重新生成。

resource_metrics 的 session_wall_seconds 仅是当前进程时间；已记录生成 Token/耗时包括恢复前的有效记录。中断时尚未返回的生成成本无法精确观测，明确标记未知，不能偷偷按零成本解释。

## 当前验证边界

当前完整测试 127 项通过，包含真实 GPU 上的随机初始化微型 Qwen3 生成与保存/重载测试。该微型模型只有 106880 个参数，没有加载目标权重，其显存和耗时不能用于估算正式模型性能。目标 Qwen3-0.6B 已在单独的真实运行中完成 Base/SFT 开发评测。

真实 Base、SFT 与资源结果分别见 `reports/phase3_report.md`、`phase4_report.md`。下一步以固定 SFT adapter 实现 GRPO，不能把已完成的 SFT 交叉熵训练描述为强化学习。
