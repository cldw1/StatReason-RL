"""真实 GPU 运算测试，使用随机初始化的微型 Qwen3，绝非目标预训练模型结果。"""
from pathlib import Path
import time

import pytest


@pytest.mark.gpu
def test_tiny_random_qwen3_bf16_generation_save_reload(tmp_path):
    import torch
    from transformers import AutoModelForCausalLM, Qwen3Config
    from src.utils.artifacts import write_json

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 unavailable")
    torch.cuda.init()
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    config = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                        max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0,
                        attention_dropout=0.0)
    config._attn_implementation = "sdpa"
    # 使用 HF 的 dtype 初始化路径，保留 RoPE 的 FP32 频率 buffer。
    # 对整模型 .to(dtype=BF16) 会把非持久 buffer 降精度，造成重载前后差异。
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16,
                                            attn_implementation="sdpa").to("cuda:0").eval()
    ids = torch.tensor([[1, 5, 8, 13]], device="cuda:0")
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        logits = model(ids, attention_mask=mask).logits
        generated = model.generate(ids, attention_mask=mask, max_new_tokens=3, do_sample=False,
                                   pad_token_id=0, eos_token_id=2)
    assert logits.shape == (1, 4, 256) and torch.isfinite(logits).all()
    assert torch.equal(generated[0, :4], ids[0]) and generated.shape[1] > 4
    checkpoint = tmp_path / "tiny_random_fixture"
    model.save_pretrained(checkpoint, safe_serialization=True)
    reloaded = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True,
                                                    dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
    assert model.model.rotary_emb.inv_freq.dtype == reloaded.model.rotary_emb.inv_freq.dtype == torch.float32
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, reloaded.state_dict()[name], rtol=0, atol=0)
    with torch.inference_mode():
        reloaded_logits = reloaded(ids, attention_mask=mask).logits
    torch.testing.assert_close(logits, reloaded_logits, rtol=0, atol=0)
    torch.cuda.synchronize()
    output = Path(__file__).resolve().parents[1] / "outputs/gpu_smoke/tiny_random_qwen3.json"
    write_json(output, {"scope": "random_initialized_tiny_architecture_fixture_not_qwen3_0_6b_weights",
                        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                        "dtype": str(logits.dtype), "forward_finite": True, "generation_succeeded": True,
                        "saved_reloaded_logits_exact_match": True, "training_performed": False,
                        "duration_seconds": time.perf_counter() - started,
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)})
