"""不依赖训练框架的审计，以及真实 CUDA/BF16/优化器测试。"""
from __future__ import annotations

from dataclasses import fields
import importlib.metadata as metadata
import inspect
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from .artifacts import file_sha256, utc_now

TRAINING_PACKAGES = (
    "torch", "transformers", "trl", "peft", "accelerate", "datasets", "numpy",
    "scipy", "pandas", "matplotlib", "tensorboard", "pytest", "PyYAML", "psutil",
)


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in TRAINING_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _command(args: list[str]) -> dict:
    """只运行参数列表，不记录 Token 或完整环境变量。"""
    try:
        result = subprocess.run(args, capture_output=True, text=True, errors="replace", timeout=30)
        return {"exit_code": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"exit_code": None, "error": str(exc)}


def collect_environment(project_root: Path) -> dict:
    """只读缓存目录和版本信息，不下载权重。"""
    cache_root = Path(os.environ.get("HF_HOME", "D:/AIModels/HuggingFace"))
    cache_inventory = []
    for model_dir in sorted((cache_root / "hub").glob("models--*")):
        cache_inventory.append({
            "repository_cache": model_dir.name,
            "snapshots": sorted(path.name for path in (model_dir / "snapshots").glob("*") if path.is_dir()),
        })
    disks = {}
    for anchor in dict.fromkeys((project_root.anchor, Path(sys.executable).anchor)):
        if anchor:
            usage = shutil.disk_usage(anchor)
            disks[anchor] = {"total_bytes": usage.total, "free_bytes": usage.free}
    freeze = _command([sys.executable, "-m", "pip", "freeze"])
    return {
        "recorded_at_utc": utc_now(), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "machine": platform.machine(), "logical_cpu_count": os.cpu_count(),
        "project_root": str(project_root), "packages": package_versions(),
        "nvidia_smi": _command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,compute_cap", "--format=csv,noheader"]),
        "hf_home": str(cache_root), "model_cache_inventory": cache_inventory,
        "qwen3_0_6b_cache_present": (cache_root / "hub" / "models--Qwen--Qwen3-0.6B").exists(),
        "disk": disks, "pip_freeze": freeze.get("stdout", ""),
        "pip_freeze_status": freeze["exit_code"],
    }


def test_cuda() -> dict:
    """小矩阵与小网络测试，不加载模型；不可用作 LLM 吞吐估计。"""
    import torch
    import torch.nn.functional as functional

    start = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() 为 False")
    # 可用性查询和 manual_seed 不一定初始化 CUDA；内存统计要求先初始化。
    torch.cuda.init()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device)
    capability = torch.cuda.get_device_capability(device)
    arches = torch.cuda.get_arch_list()
    native_arch = f"sm_{capability[0]}{capability[1]}"
    torch.backends.cuda.matmul.allow_tf32 = False
    a, b = torch.randn(64, 64), torch.randn(64, 64)
    reference = a @ b
    actual = a.to(device) @ b.to(device)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-4)
    x = torch.tensor([1.0, 2.0, 3.0], device=device, requires_grad=True)
    loss = x.square().sum()
    loss.backward()
    torch.testing.assert_close(x.grad, 2 * x.detach())
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU/PyTorch 不支持 BF16")
    bf_result = a.to(device, torch.bfloat16) @ b.to(device, torch.bfloat16)
    bf_relative_error = ((bf_result.float().cpu() - reference).norm() / reference.norm()).item()
    if not torch.isfinite(bf_result).all() or bf_relative_error > 0.02:
        raise RuntimeError(f"BF16 矩阵计算误差异常: {bf_relative_error}")
    # 验证 FP32 主参数 + BF16 autocast 和 AdamW，不等于 LoRA 训练。
    layer = torch.nn.Linear(64, 16).to(device)
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.01)
    before = layer.weight.detach().clone()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        prediction = layer(a.to(device))
        training_loss = prediction.float().square().mean()
    training_loss.backward()
    if not all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in layer.parameters()):
        raise RuntimeError("BF16 autocast 反传出现无效梯度")
    optimizer.step()
    parameter_delta = (layer.weight.detach() - before).abs().max().item()
    if parameter_delta <= 0:
        raise RuntimeError("优化器没有实际更新参数")
    q = torch.randn(1, 2, 32, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    k, v = torch.randn_like(q, requires_grad=True), torch.randn_like(q, requires_grad=True)
    attention = functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    attention.float().square().mean().backward()
    for tensor in (attention, q.grad, k.grad, v.grad):
        if tensor is None or not torch.isfinite(tensor).all():
            raise RuntimeError("SDPA 前向/反向存在无效值")
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(device)
    return {
        "status": "passed", "torch_version": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda, "gpu_name": properties.name,
        "gpu_memory_bytes": properties.total_memory, "compute_capability": list(capability),
        "compiled_architectures": arches, "native_architecture_present": native_arch in arches,
        "bf16_supported": True, "fp32_matmul_cpu_comparison": "passed",
        "analytic_gradient": x.grad.cpu().tolist(), "bf16_matmul_relative_norm_error": bf_relative_error,
        "bf16_autocast_loss": training_loss.item(), "adamw_max_parameter_delta": parameter_delta,
        "bf16_sdpa_forward_backward": "passed", "duration_seconds": time.perf_counter() - start,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "scope": "small_tensor_checks_only_not_model_training",
    }


def audit_training_api() -> dict:
    """安装后检查真实导入和源码哈希；缺包时标记 blocked。"""
    versions = package_versions()
    missing = [name for name, version in versions.items() if version is None]
    if missing:
        return {"status": "blocked_missing_dependencies", "missing_packages": missing, "versions": versions}
    try:
        from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
        required = {
            "loss_type", "scale_rewards", "beta", "epsilon", "epsilon_high",
            "num_generations", "num_iterations", "max_completion_length",
            "mask_truncated_completions", "steps_per_generation",
        }
        available = {field.name for field in fields(GRPOConfig)}
        absent = sorted(required - available)
        if absent:
            return {"status": "failed", "missing_grpo_config_fields": absent}
        implementations = (GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer)
        sources = {}
        for implementation in implementations:
            path = inspect.getsourcefile(implementation)
            sources[implementation.__name__] = {"source_file": path, "sha256": file_sha256(path) if path else None}
        return {"status": "passed", "versions": versions, "sources": sources, "grpo_fields": sorted(required), "scope": "imports_and_api_presence_only_not_training_compatibility"}
    except Exception as exc:
        return {"status": "failed", "versions": versions, "error": f"{type(exc).__name__}: {exc}"}

