"""Qwen3 离线加载与实际生成；所有判题只使用新生成 Token。"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time

from .prompts import build_messages
from .snapshot import ModelSnapshot


@dataclass(frozen=True)
class CompletionBoundary:
    token_ids: list[int]
    text_token_ids: list[int]
    eos_seen: bool
    truncated: bool
    stop_reason: str


def extract_completion(sequence: list[int], prompt_length: int, eos_ids: list[int], max_new_tokens: int) -> CompletionBoundary:
    """EOS 恰好出现在最大长度处时是正常结束，不误判为截断。"""
    if prompt_length < 0 or prompt_length > len(sequence) or max_new_tokens <= 0:
        raise ValueError("非法输入/生成长度")
    generated = sequence[prompt_length:]
    for index, token in enumerate(generated):
        if token in eos_ids:
            return CompletionBoundary(generated[:index + 1], generated[:index], True, False, "eos")
    reason = "max_new_tokens" if len(generated) >= max_new_tokens else "stopped_without_eos"
    return CompletionBoundary(generated, generated, False, True, reason)


def example_seed(base_seed: int, example_id: str, sample_index: int = 0) -> int:
    """按题目派生 seed，使断点恢复和执行顺序不改变单题的初始随机状态。"""
    digest = hashlib.sha256(f"{base_seed}:{example_id}:{sample_index}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def load_qwen(snapshot: ModelSnapshot, model_config: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("此入口要求 CUDA，不能静默退回 CPU 并误报 GPU 资源")
    if model_config["dtype"] != "bfloat16" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前配置要求可运行的 BF16")
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot.path), local_files_only=True, trust_remote_code=False,
                                               padding_side="left", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(snapshot.path), local_files_only=True, trust_remote_code=False,
                                                 dtype=torch.bfloat16, attn_implementation=model_config["attn_implementation"])
    expected_architecture = "qwen2" if snapshot.repository == "Qwen/Qwen2.5-Math-1.5B-Instruct" else "qwen3"
    if model.config.model_type != expected_architecture:
        raise ValueError(f"快照架构与候选模型不符: {model.config.model_type}")
    model.to("cuda:0")
    model.eval()
    torch.cuda.synchronize()
    return model, tokenizer, {"load_seconds": time.perf_counter() - started,
                               "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                               "dtype": str(next(model.parameters()).dtype), "device": str(next(model.parameters()).device),
                               "model_type": model.config.model_type, "attention": model.config._attn_implementation,
                               "load_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                               "load_peak_reserved_bytes": torch.cuda.max_memory_reserved(0)}


def generate_completion(model, tokenizer, record: dict, generation: dict, *, enable_thinking: bool, seed: int) -> dict:
    return generate_completions(model, tokenizer, [record], generation, enable_thinking=enable_thinking, seed=seed)[0]


def generate_completions(model, tokenizer, records: list[dict], generation: dict, *, enable_thinking: bool, seed: int,
                         messages_override: list[list[dict]] | None = None) -> list[dict]:
    """左侧 padding 批量生成。种子属于固定批次，耗时按批次均摊。

    每行在其首个 EOS 结束；其他行继续解码产生的 pad 不计入有效生成。
    改变 batch_size 会改变采样，模型间必须固定题序及批次。
    """
    import torch
    from transformers import GenerationConfig, set_seed
    if not records or tokenizer.padding_side != 'left':
        raise ValueError('生成要求非空批次及左侧 padding')
    messages = [build_messages(record) for record in records] if messages_override is None else messages_override
    if len(messages) != len(records):
        raise ValueError('自定义消息数量必须与题目数量一致')
    if model.config.model_type == 'qwen2' and enable_thinking:
        raise ValueError('Qwen2.5-Math 没有 Qwen3 thinking 模式开关')
    if len(records) == 1:
        messages = messages[0]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            padding=True, enable_thinking=enable_thinking, return_tensors="pt", return_dict=True)
    prompt_length = encoded["input_ids"].shape[1]
    if prompt_length > generation["max_prompt_tokens"]:
        raise ValueError('批次 prompt 超出上限；禁止静默截断题面')
    inputs = {key: value.to(model.device) for key, value in encoded.items()}
    eos = model.generation_config.eos_token_id or tokenizer.eos_token_id
    eos_ids = [eos] if isinstance(eos, int) else list(eos)
    parameters = {
        "max_new_tokens": generation["max_new_tokens"], "max_time": generation["max_time_per_completion_seconds"],
        "do_sample": generation["do_sample"], "num_beams": 1, "num_return_sequences": 1,
        "temperature": generation["temperature"], "top_p": generation["top_p"],
        "top_k": generation["top_k"], "min_p": generation["min_p"],
        "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "use_cache": True,
        "eos_token_id": eos_ids, "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id if tokenizer.bos_token_id is not None else model.generation_config.bos_token_id,
        "return_dict_in_generate": False,
        "output_scores": False,
    }
    config = GenerationConfig(**parameters)
    set_seed(seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, generation_config=config)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if not torch.equal(output[:, :prompt_length], inputs["input_ids"]):
        raise RuntimeError("生成序列前缀与实际 prompt 不匹配")
    results = []
    for index, sequence in enumerate(output.detach().cpu().tolist()):
        boundary = extract_completion(sequence, prompt_length, eos_ids, generation['max_new_tokens'])
        results.append({
        "completion": tokenizer.decode(boundary.text_token_ids, skip_special_tokens=True),
        "raw_completion": tokenizer.decode(boundary.token_ids, skip_special_tokens=False),
        "completion_token_ids": boundary.token_ids, "prompt_token_count": int(encoded['attention_mask'][index].sum()),
        "generated_token_count": len(boundary.token_ids), "text_token_count": len(boundary.text_token_ids),
        "eos_seen": boundary.eos_seen, "truncated": boundary.truncated, "stop_reason": boundary.stop_reason,
        "generation_seconds": elapsed / len(records),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(model.device),
        "seed": seed, "enable_thinking": enable_thinking,
        "prompt_text": tokenizer.decode(encoded['input_ids'][index][encoded['attention_mask'][index].bool()], skip_special_tokens=False),
        "generation_config": config.to_dict(),
        'generation_batch_size':len(records), 'generation_batch_seconds':elapsed,
        'timing_scope':'batch_wall_time_amortized_per_completion',
        'seed_scope':'fixed_batch' if len(records)>1 else 'question',
        'padded_prompt_width':prompt_length, 'batch_decode_width':output.shape[1]-prompt_length,
        })
    return results
