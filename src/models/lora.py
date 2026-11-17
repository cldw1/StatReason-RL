"""LoRA 注入、更新证据及与固定 Base 绑定的 adapter 校验。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.utils.artifacts import file_sha256


def tensor_sha256(tensor) -> str:
    import torch
    value = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(value).hexdigest()


def attach_lora(model, config: dict):
    from peft import LoraConfig, get_peft_model
    available = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    if not set(config["target_modules"]) <= available:
        raise ValueError("目标 LoRA 模块在模型中不存在")
    wrapped = get_peft_model(model, LoraConfig(**config), autocast_adapter_dtype=True)
    trainable = [(name, parameter) for name, parameter in wrapped.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("可训练参数不是预期的 LoRA 参数")
    return wrapped


def capture_trainable(model) -> dict:
    return {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}


def parameter_update_evidence(model, initial: dict) -> dict:
    import torch
    rows = []
    for name, parameter in model.named_parameters():
        if name not in initial:
            continue
        current = parameter.detach().cpu()
        if not torch.isfinite(current).all():
            raise RuntimeError("LoRA 参数出现非有限值")
        difference = current.float() - initial[name].float()
        rows.append({"name": name, "elements": current.numel(), "dtype": str(current.dtype),
                     "max_abs_delta": difference.abs().max().item(), "delta_l2": difference.norm().item(),
                     "initial_sha256": tensor_sha256(initial[name]), "final_sha256": tensor_sha256(current)})
    changed = sum(row["max_abs_delta"] > 0 for row in rows)
    if changed == 0:
        raise RuntimeError("没有任何 LoRA 参数实际变化")
    return {"trainable_tensor_count": len(rows), "changed_tensor_count": changed,
            "trainable_parameter_count": sum(row["elements"] for row in rows),
            "max_abs_delta": max(row["max_abs_delta"] for row in rows), "parameters": rows,
            "all_non_lora_parameters_frozen": all(not parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name)}


def adapter_metadata(path: Path, repository: str, revision: str) -> dict:
    return {"base_repository": repository, "base_revision": revision,
            "adapter_safetensors_sha256": file_sha256(path / "adapter_model.safetensors"),
            "adapter_config_sha256": file_sha256(path / "adapter_config.json"),
            "format": "peft_lora", "quantization": "none_plain_bf16_lora"}


def validate_adapter(path: Path, repository: str, revision: str) -> dict:
    recorded = json.loads((path / "adapter_metadata.json").read_text(encoding="utf-8"))
    current = adapter_metadata(path, repository, revision)
    if any(recorded.get(key) != value for key, value in current.items()):
        raise ValueError("adapter 权重/配置哈希或 Base 模型版本不一致")
    return current
