"""加载继承式配置并检查关键实验约束。

项目配置采用 JSON 语法的 YAML（JSON 是 YAML 1.2 的子集），因此尚未安装
PyYAML 时也能离线检查；安装后也接受普通 YAML，始终使用 safe_load。
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


def deep_merge(base: dict, override: dict) -> dict:
    """深拷贝合并，防止一个消融配置修改另一个实验的配置。"""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """extends 相对当前配置文件解析，检测循环继承。"""
    path = Path(path).resolve()
    if path in _seen:
        raise ValueError(f"配置循环继承: {path.name}")
    content = path.read_text(encoding="utf-8-sig")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("普通 YAML 需要安装 PyYAML；当前可使用 JSON 语法的 YAML") from exc
        data = yaml.safe_load(content)
    if not isinstance(data, dict):
        raise ValueError("配置根节点必须是字典")
    parent = data.pop("extends", None)
    if parent is not None:
        if not isinstance(parent, str):
            raise ValueError("extends 必须是一个相对配置路径")
        data = deep_merge(load_config(path.parent / parent, _seen + (path,)), data)
    return data


def validate_config(config: dict[str, Any]) -> None:
    """检查阶段一已定义的约束；不宣称配置已通过真实 Trainer 验证。"""
    if config.get("schema_version") != 1:
        raise ValueError("不支持的配置 schema_version")
    experiment = config.get("experiment", {})
    level = experiment.get("level", 0)
    if level not in (0, 1, 2):
        raise ValueError("实验 level 必须为 0、1 或 2")
    if level > 0 and not config.get("model", {}).get("revision"):
        raise ValueError("正式实验必须固定模型 commit revision")
    if level > 0 and not config.get("data", {}).get("fingerprint"):
        raise ValueError("正式实验必须固定数据指纹")
    if experiment.get("stage") != "rl":
        return
    trainer = config["trainer"]
    generations = trainer["num_generations"]
    if not isinstance(generations, int) or isinstance(generations, bool) or generations < 2:
        raise ValueError("GRPO num_generations 至少为 2")
    batch = trainer["per_device_train_batch_size"] * trainer["gradient_accumulation_steps"]
    if batch % generations:
        raise ValueError("单卡有效批量必须能被 num_generations 整除")
    if trainer["epsilon"] <= 0 or trainer["epsilon_high"] <= 0:
        raise ValueError("裁剪阈值必须为正")
    if trainer["loss_type"] not in ("grpo", "dapo"):
        raise ValueError("本实验仅支持 grpo 或 dapo 归一化")
    if trainer["num_iterations"] < 2:
        raise ValueError("本项目裁剪对照要求固定旧策略后至少进行两轮更新")
    if trainer["use_vllm"]:
        raise ValueError("本地初始配置使用 Transformers generate")
    if trainer["generation_kwargs"].get("do_sample") is not True:
        raise ValueError("RL rollout 必须启用随机采样")
    mechanisms = config["mechanisms"]
    if mechanisms["dynamic_sampling"]["enabled"]:
        if mechanisms["dynamic_sampling"]["max_resample_rounds"] < 0:
            raise ValueError("动态补采次数不能为负")
        if mechanisms["dynamic_sampling"]["max_generated_tokens_per_batch"] <= 0:
            raise ValueError("动态采样必须设置正数 Token 预算上限")
    overlong = mechanisms["overlong"]
    if overlong["enabled"] and not 0 < overlong["soft_length"] < trainer["max_completion_length"]:
        raise ValueError("软长度阈值必须介于 0 与最大生成长度之间")
    if level > 0:
        checkpoint = config.get("initialization", {})
        if not checkpoint.get("sft_adapter") or not checkpoint.get("sft_adapter_sha256"):
            raise ValueError("正式 RL 实验必须固定共同的 SFT adapter 及其指纹")

