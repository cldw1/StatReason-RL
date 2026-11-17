# 安装与运行

## 当前环境

项目目录 `D:\PythonProjects\StatReason-RL` 使用已有 `.venv\Scripts\python.exe`。Python 3.13.15、PyTorch 2.8.0+cu128 和指定 Hugging Face 训练依赖已经安装，GPU 检查、真实导入、API 字段和 pip check 均通过。已安装版本快照保存在 `requirements.lock.windows-py313-cu128.txt`。

已完成300步SFT，以及E0–E4五组独立RL运行，共430次RL参数更新；224个LoRA张量更新、检查点重载和Vanilla/E4短程恢复均有实测证据。统一评测完成12个检查点×400题，并生成最终报告和11张图。后文4步闭环、24步GRPO及192 Token评测均为历史开发实验。

## 已完成实验：查看结果

当前结果以[最终报告](../reports/ablation_round1/final_report.md)和[验收记录](../reports/ablation_round1/acceptance.json)为准。无需重新训练或重新评测即可阅读；下列状态命令不启动GPU任务。

```powershell
.\.venv\Scripts\python.exe scripts\24_run_ablations.py --status
.\.venv\Scripts\python.exe scripts\26_evaluate_ablations.py --status
```

E0、E1、E3、E4各完成100步；E2在30步达到补采上限而停止，保留其真实结果。所有组共同30步作一次比较，可用100步终点另列。正式评测使用同一400题、Non-thinking、256 Token、temperature=0.7、top_p=1、top_k=0、batch=4、每题一次采样。GSM部分来自官方train预留划分，不是官方test成绩。

如需由已保存结果重建报告，下面是CPU分析命令，不进行模型生成；完整Bootstrap和绘图需要运行时间。已有报告完整时无需重复执行。

```powershell
.\.venv\Scripts\python.exe scripts\27_finalize_ablation_report.py --audit-only
.\.venv\Scripts\python.exe scripts\27_finalize_ablation_report.py
```

## 正式训练入口：仅在明确重新实验时使用

先核对模型缓存、SFT adapter路径与哈希、RL池指纹和实验版本。以下为训练命令，会占用GPU；本轮结果已完成，不需要再次运行。

```powershell
.\.venv\Scripts\python.exe scripts\24_run_ablations.py --experiments E0 E1 E2 --dry-run
.\.venv\Scripts\python.exe scripts\24_run_ablations.py --experiments E0 E1 E2
.\.venv\Scripts\python.exe scripts\24_run_ablations.py --experiments E3 E4
```

入口将详细控制台输出写入日志，只显示中文进度。相同配置的完整运行会自动跳过；已有提前停止运行不会被自动覆盖或续训。本轮E2提前停止会使第一批命令返回非零状态，之后需要单独执行E3/E4命令，不能用仅在前一步成功时执行的连接方式。达到补采上限是已记录的训练结果，不应删除原目录以强迫重跑。

训练后先运行`scripts/25_analyze_ablation_training.py`核对训练并准备评测计划，再使用以下命令开始或继续已冻结计划的评测：

```powershell
.\.venv\Scripts\python.exe scripts\26_evaluate_ablations.py --execute
```

评测默认每次会话上限120分钟，保存批次可续跑。计划绑定模型、数据、代码及生成配置；更换模型或训练配置后应建立独立实验和评测版本，不能修改旧计划哈希来绕过校验。最终400题已使用，不再用它选择超参数。

## 复建依赖

新环境先从 PyTorch 官方 CUDA 12.8 索引安装对应 PyTorch wheel，再安装项目依赖。不要使用 --no-deps 忽略冲突。

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
python scripts\01_check_env.py --require-training-stack
```

现有环境无需重装 PyTorch。`scripts/install_environment.ps1` 可保存安装报告和前后 freeze，并保护已有 CUDA wheel；`-DryRun` 只解析依赖。

## 当前检查与数据命令

```powershell
.\.venv\Scripts\python.exe scripts\01_check_env.py --require-training-stack
.\.venv\Scripts\python.exe scripts\00_validate_configs.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe scripts\03_prepare_data.py --statistics prototype
.\.venv\Scripts\python.exe scripts\03_prepare_data.py --statistics full
.\.venv\Scripts\python.exe scripts\03_audit_rewards.py
```

默认测试临时目录在项目 outputs 下，避免当前 Windows 隔离临时卷的原子重命名限制。测试结束可查看 `outputs/setup/phase2_tests_verified.xml` 和日志。每次审计生成唯一 run_id；不覆盖已存在的环境运行目录。

统计数据版本已经存在时会校验内容和文件哈希，不重复覆盖。更改生成器导致题目内容变化时需要用 `--output-root` 指定新的版本目录，并同时更新实验的数据指纹。

## GSM8K 获取

GSM8K 已在本人终端成功获取并在当前环境离线复现。新机器缺失缓存时可运行：

```powershell
.\.venv\Scripts\python.exe scripts\03_prepare_data.py --statistics none --gsm8k
```

只从官方 `openai/gsm8k` 下载 main/train parquet。程序解析 commit 后固定该版本，保存 raw SHA256；不获取官方 test。完成后应看到 7473 条分别为 SFT 1000、RL 4000、验证 500、预留 1973。

如果本人终端也有代理错误，先检查代理客户端和终端既有配置，保留异常信息；项目不修改系统代理。不要提供代理密码或 Hugging Face Token。

当前 GSM8K 已成功获取并完成离线复现，来源 commit 已固定在 configs/data.yaml。以下离线导入仅用于迁移其他机器时的替代方案，需要提供完整来源 commit：

```powershell
.\.venv\Scripts\python.exe scripts\03_prepare_data.py --statistics none --gsm8k --gsm8k-local-train <官方训练parquet路径> --gsm8k-revision <40位来源commit>
```

尖括号内容为待替换参数，不能原样执行。离线来源会明确标记未经远程校验，不能把任意 parquet 当成官方版本。

## 模型缓存与早期闭环

`HF_HOME=D:\AIModels\HuggingFace`，Qwen3-0.6B 已缓存，commit 固定为 c1899de289a04d12100db370d81485cdf75e47ca。缺失缓存的新机器可运行：

```powershell
.\.venv\Scripts\python.exe scripts\04_evaluate_base.py --download-model --mode both
```

该历史入口固定模型commit、检查不超过2 GiB的文件预算，并执行两种模式各八题的开发冒烟评测。八题结果只用于工程排错，不作为能力结论，也不是本轮400题正式评测。已有缓存时省略下载参数。详细加载、EOS/截断、逐题记录和恢复说明见`model_and_evaluation.md`。

SFT的`configs/sft.yaml`为四步Level 0闭环，`configs/sft_local.yaml`为已完成的300步本地训练，设600秒训练预算上限。主模型导出验证loss选中的第150步adapter；不能将它描述为第300步模型。训练命令、checkpoint恢复与adapter评测见`lora_sft_explained.md`。4步GRPO闭环与24步本地实验为历史记录；E0–E4正式入口见本文前部。未启动付费云资源。

## 历史Vanilla GRPO开发实验

先核对`configs/grpo.yaml`的SFT adapter路径与SHA256。当前本机指向已选中的通用配比SFT第150步；迁移后应从新训练输出的`adapter_metadata.json`读取真实哈希，统一修改共享初始化，不为每个消融单独挑SFT。

```powershell
./.venv/Scripts/python.exe scripts/07_train_grpo.py --prepare-only
./.venv/Scripts/python.exe scripts/07_train_grpo.py
./.venv/Scripts/python.exe scripts/07_train_grpo.py --config configs/grpo_local.yaml
```

默认是4步Level0功能题组。`grpo_local.yaml`是Level1小规模实验：完整4400条RL池随机采样、24步、G=4、每rollout两轮，共12题48回答；不是4400题一遍。训练软时限600秒，保存与停止落在两轮复用周期边界。原SFT独立冻结为参考，使用双模型但不训练参考权重。

找到本次完成运行目录后评测和绘图，尖括号参数需要替换：

```powershell
./.venv/Scripts/python.exe scripts/09_evaluate_all.py --config configs/evaluation_diagnostic.yaml --adapter <GRPO目录>/adapter
./.venv/Scripts/python.exe scripts/14_analyze_grpo.py --run <GRPO目录> --sft-evaluation <原SFT的100题评测目录> --grpo-evaluation <本次100题评测目录>
```

这组历史100题开发评测使用非思考模式、192 Token、batch=4和固定批次seed，训练上限为128 Token。它不适用于本轮正式能力结论；当前12检查点统一评测为256 Token。不要将8题冒烟、100题开发和400题最终评测混合计算正确率。

恢复同一run使用相同配置：

```powershell
./.venv/Scripts/python.exe scripts/07_train_grpo.py --config configs/grpo_local.yaml --resume <原运行目录>
```

精确恢复验证使用**本次用当前源码新建且已完成的run**，并确认第20步检查点仍在：

```powershell
./.venv/Scripts/python.exe scripts/15_verify_grpo_resume.py --run <本次新GRPO目录> --config configs/grpo_local.yaml --checkpoint-step 20
```

该脚本创建独立验证目录、复制第20步及之前的审计前缀，只补跑4步，然后对比最终224个LoRA张量和新增rollout。旧run绑定训练时的源码哈希；若继续开发了训练代码，需使用其匹配源码版本恢复，不要改日志中的哈希绕过检查。已有同机等价验证记录位于`reports/grpo/resume_verification.json`。
