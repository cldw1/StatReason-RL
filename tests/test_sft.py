"""助手标签、padding、数据隔离及微型模型的真实 LoRA 更新/重载测试。"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.data.statistics import generate_statistics
from src.data.storage import save_dataset
from src.models.lora import adapter_metadata, attach_lora, capture_trainable, parameter_update_evidence, validate_adapter
from src.training.sft_data import AssistantCollator, balanced_indices, encode_sft_record, prepare_sft_data
from src.utils.artifacts import write_json


class ToyTemplateTokenizer:
    """只用于验证标签边界；不冒充 Qwen Tokenizer。"""
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        prefix = [5, 6, 7]
        if kwargs.get("add_generation_prompt"):
            return prefix
        return prefix + [ord(character) + 10 for character in messages[-1]["content"]] + [2, 9]

    def decode(self, ids, **kwargs):
        return "".join(chr(value - 10) for value in ids if value >= 10)


def test_labels_mask_prompt_keep_answer_and_eos():
    record = generate_statistics({"prototype": 1})[0]
    encoded, audit = encode_sft_record(ToyTemplateTokenizer(), record, 10000)
    assert encoded["labels"][:3] == [-100, -100, -100]
    assert encoded["labels"][3:] == encoded["input_ids"][3:]
    assert encoded["labels"][-1] == 2 and encoded["input_ids"][-1] == 2
    assert audit["supervised_tokens"] > 1 and audit["prompt_fully_masked"]
    assert "FINAL_ANSWER:" in audit["supervised_text"]


def test_overlong_skips_whole_record_instead_of_cutting_answer():
    record = generate_statistics({"prototype": 1})[0]
    encoded, audit = encode_sft_record(ToyTemplateTokenizer(), record, 10)
    assert encoded is None and audit["excluded_overlong"]
    assert "FINAL_ANSWER:" in audit["supervised_text"]


def test_template_mismatch_and_invalid_reference_fail():
    record = generate_statistics({"prototype": 1})[0]
    class BadTokenizer(ToyTemplateTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            return [88] + result if not kwargs.get("add_generation_prompt") else result
    with pytest.raises(ValueError, match="前缀"):
        encode_sft_record(BadTokenizer(), record, 10000)
    record["solution"] = "not a valid reference"
    with pytest.raises(ValueError, match="标准解答"):
        encode_sft_record(ToyTemplateTokenizer(), record, 10000)


def test_collator_masks_padding_even_when_pad_equals_eos():
    batch = AssistantCollator(pad_token_id=2, multiple=4)([
        {"input_ids": [5, 6, 2], "labels": [-100, 6, 2]},
        {"input_ids": [5, 2], "labels": [-100, 2]},
    ])
    assert batch["input_ids"].shape == (2, 4)
    assert batch["labels"].tolist() == [[-100, 6, 2, -100], [-100, 2, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 0, 0]]
    with pytest.raises(ValueError):
        AssistantCollator(0)([{"input_ids": [1], "labels": [-100]}])


def test_balanced_smoke_selection_and_all_data_selection_reproduce():
    records = generate_statistics({"sft": 28})
    chosen = balanced_indices(records, 7, 42)
    assert len({records[index]["task_type"] for index in chosen}) == 7
    assert chosen == balanced_indices(records, 7, 42)
    assert sorted(balanced_indices(records, None, 42)) == list(range(len(records)))


def test_dataset_preparation_retains_sft_validation_isolation(tmp_path):
    records = generate_statistics({"sft": 7, "validation": 7})
    save_dataset(tmp_path / "statistics_full", records, {"fixture": True})
    gsm = []
    for split in ("sft", "validation"):
        row = deepcopy(next(record for record in records if record["split"] == split))
        row.update(id=f"gsm-fixture-{split}", source="openai/gsm8k", task_type="gsm8k_math", question=f"GSM fixture {split}")
        gsm.append(row)
    save_dataset(tmp_path / "gsm8k", gsm, {"fixture": True})
    config = {"assistant_only_loss": True, "packing": False, "max_length": 10000, "max_train_examples": 8, "max_validation_examples": 8}
    train, validation, selection, audits = prepare_sft_data(ToyTemplateTokenizer(), tmp_path, config, 42)
    assert len(train) == len(validation) == 8
    assert not set(selection["sft"]["selected_ids"]) & set(selection["validation"]["selected_ids"])
    assert all(audit["prompt_fully_masked"] and audit["supervision_ends_with_eos"] for audit in audits)


def test_tiny_cpu_qwen_lora_real_update_and_reload(tmp_path):
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM, Qwen3Config
    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                        max_position_embeddings=32, eos_token_id=2, pad_token_id=0)
    base = AutoModelForCausalLM.from_config(config)
    base_path = tmp_path / "base_fixture"
    base.save_pretrained(base_path)
    base.name_or_path = str(base_path)
    model = attach_lora(base, {"r": 2, "lora_alpha": 4, "lora_dropout": 0.0,
                               "target_modules": ["q_proj", "v_proj"], "bias": "none", "task_type": "CAUSAL_LM"})
    initial = capture_trainable(model)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=0.01)
    ids = torch.tensor([[1, 5, 8, 10, 2]])
    labels = torch.tensor([[-100, -100, 8, 10, 2]])
    loss = model(ids, attention_mask=torch.ones_like(ids), labels=labels).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters() if parameter.requires_grad)
    optimizer.step()
    evidence = parameter_update_evidence(model, initial)
    assert evidence["changed_tensor_count"] > 0 and evidence["all_non_lora_parameters_frozen"]
    model.eval()
    with torch.inference_mode():
        before = model(ids).logits
    path = tmp_path / "adapter_fixture"
    model.save_pretrained(path, safe_serialization=True, save_embedding_layers=False)
    metadata = adapter_metadata(path, "random_test_fixture", "not_pretrained")
    write_json(path / "adapter_metadata.json", metadata)
    assert validate_adapter(path, "random_test_fixture", "not_pretrained") == metadata
    with pytest.raises(ValueError):
        validate_adapter(path, "wrong_model", "not_pretrained")
    fresh = AutoModelForCausalLM.from_pretrained(base_path, local_files_only=True)
    reloaded = PeftModel.from_pretrained(fresh, str(path), is_trainable=False, local_files_only=True).eval()
    for key, value in get_peft_model_state_dict(model).items():
        torch.testing.assert_close(value, get_peft_model_state_dict(reloaded)[key], rtol=0, atol=0)
    with torch.inference_mode():
        after = reloaded(ids).logits
    torch.testing.assert_close(before, after, rtol=0, atol=0)
