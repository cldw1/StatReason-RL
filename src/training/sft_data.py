"""SFT 助手回答标签、长度检查及动态右侧 padding；不截掉最终答案。"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from src.data.schema import canonical_json, content_fingerprint
from src.data.storage import read_jsonl
from src.models.prompts import build_messages
from src.rewards.numeric import verify_completion
from src.utils.artifacts import file_sha256


def encode_sft_record(tokenizer, record: dict, max_length: int) -> tuple[dict | None, dict]:
    """完整 chat template 必须与推理前缀相同；超长整题排除，不截断答案。"""
    solution = record["solution"]
    if record["source"] == "openai/gsm8k":
        solution = re.sub(r"<<[^<>]*>>", "", solution)
    if not verify_completion(solution, record).strict_correct:
        raise ValueError(f"训练标准解答判题失败: {record['id']}")
    messages = build_messages(record)
    prompt = tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": solution}],
                                         tokenize=True, enable_thinking=False, add_generation_prompt=False)
    if full[:len(prompt)] != prompt:
        raise ValueError("SFT 完整消息与推理 chat template 前缀不一致，禁止猜测标签边界")
    response = full[len(prompt):]
    if response.count(tokenizer.eos_token_id) != 1:
        raise ValueError("助手回答必须只有一个结束 EOS")
    eos_index = response.index(tokenizer.eos_token_id)
    full = full[:len(prompt) + eos_index + 1]
    labels = [-100] * len(prompt) + full[len(prompt):]
    supervised_text = tokenizer.decode(full[len(prompt):-1], skip_special_tokens=True)
    if not verify_completion(supervised_text, record).strict_correct:
        raise ValueError("实际受监督 Token 解码后没有正确的唯一最终答案")
    audit = {"id": record["id"], "source": record["source"], "task_type": record["task_type"],
             "prompt_tokens": len(prompt), "sequence_tokens": len(full),
             "supervised_tokens": sum(label != -100 for label in labels),
             "prompt_fully_masked": all(label == -100 for label in labels[:len(prompt)]),
             "supervision_ends_with_eos": labels[-1] == tokenizer.eos_token_id,
             "supervised_text": supervised_text, "excluded_overlong": len(full) > max_length}
    if audit["supervised_tokens"] <= 1 or not audit["prompt_fully_masked"]:
        raise ValueError("有效助手标签为空或 prompt 未完全屏蔽")
    if len(full) > max_length:
        return None, audit
    return {"input_ids": full, "attention_mask": [1] * len(full), "labels": labels}, audit


def balanced_indices(records: list[dict], maximum: int | None, seed: int) -> list[int]:
    """smoke 按题型轮流取样；正式 maximum=None 时使用全部并确定性打乱。"""
    rng = random.Random(seed)
    if maximum is None:
        indices = list(range(len(records)))
        rng.shuffle(indices)
        return indices
    if maximum < 1:
        raise ValueError("max_examples 必须为正或 null")
    groups = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["task_type"]].append(index)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < min(maximum, len(records)):
        for group in sorted(groups):
            if groups[group]:
                selected.append(groups[group].pop())
                if len(selected) == min(maximum, len(records)):
                    return selected
    return selected


def _load_split(root: Path, split: str) -> tuple[list[dict], dict]:
    if split not in ("sft", "validation"):
        raise ValueError("SFT 处理入口不读取 RL 或测试集")
    records, sources = [], {}
    for name, source in (("statistics_full", "statreason_synthetic"), ("gsm8k", "openai/gsm8k")):
        directory = root / name
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        path = directory / f"{split}.jsonl"
        if file_sha256(path) != manifest["file_sha256"][path.name]:
            raise ValueError(f"{name}/{split} 文件哈希不一致")
        subset = read_jsonl(path)
        if any(record["split"] != split or record["source"] != source for record in subset):
            raise ValueError("SFT 数据来源或 split 标签不一致")
        records.extend(subset)
        sources[name] = manifest["statistics"]["fingerprint"]
    return records, sources


def prepare_sft_data(tokenizer, root: Path, config: dict, seed: int) -> tuple[list[dict], list[dict], dict, list[dict]]:
    if config["assistant_only_loss"] is not True or config["packing"] is not False:
        raise ValueError("本实现要求助手回答损失且不 packing")
    datasets, selection, audits = {}, {}, []
    for split, maximum in (("sft", config["max_train_examples"]), ("validation", config["max_validation_examples"])):
        records, sources = _load_split(root, split)
        eligible_records, features, split_audits = [], [], []
        for record in records:
            encoded, audit = encode_sft_record(tokenizer, record, config["max_length"])
            split_audits.append(audit)
            if encoded is not None:
                eligible_records.append(record)
                features.append(encoded)
        if not features:
            raise ValueError("没有可训练或验证的非空助手标签样本")
        indices = balanced_indices(eligible_records, maximum, seed + int(split == "validation"))
        datasets[split] = [features[index] for index in indices]
        selected_records = [eligible_records[index] for index in indices]
        ids = {record["id"] for record in selected_records}
        audits.extend({**audit, "split": split} for audit in split_audits if audit["id"] in ids)
        encoded_fingerprint = hashlib.sha256(canonical_json(datasets[split]).encode()).hexdigest()
        selection[split] = {
            "source_counts": dict(Counter(record["source"] for record in records)),
            "candidate_count": len(records), "eligible_count": len(features), "selected_count": len(indices),
            "excluded_overlong_ids": [audit["id"] for audit in split_audits if audit["excluded_overlong"]],
            "selected_ids": [record["id"] for record in selected_records],
            "selected_by_source": dict(Counter(record["source"] for record in selected_records)),
            "selected_by_task": dict(Counter(record["task_type"] for record in selected_records)),
            "record_fingerprint": content_fingerprint(selected_records), "encoded_fingerprint": encoded_fingerprint,
            "source_fingerprints": sources,
            "max_selected_sequence_length": max(len(features[index]["input_ids"]) for index in indices),
            "total_selected_supervised_tokens": sum(sum(label != -100 for label in features[index]["labels"]) for index in indices),
        }
    if set(selection["sft"]["selected_ids"]) & set(selection["validation"]["selected_ids"]):
        raise ValueError("SFT 和验证题 ID 交叉")
    return datasets["sft"], datasets["validation"], selection, audits


class EncodedDataset:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class AssistantCollator:
    """右侧 padding 的 input/mask/labels 同步处理，pad 标签恒为 -100。"""
    def __init__(self, pad_token_id: int, multiple: int = 8):
        self.pad_token_id, self.multiple = pad_token_id, multiple

    def __call__(self, features: list[dict]) -> dict:
        import torch
        maximum = max(len(feature["input_ids"]) for feature in features)
        maximum = (maximum + self.multiple - 1) // self.multiple * self.multiple
        output = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            size = len(feature["input_ids"])
            if len(feature["labels"]) != size or not any(label != -100 for label in feature["labels"]):
                raise ValueError("标签长度不符或全部为 -100")
            output["input_ids"].append(feature["input_ids"] + [self.pad_token_id] * (maximum - size))
            output["attention_mask"].append([1] * size + [0] * (maximum - size))
            output["labels"].append(feature["labels"] + [-100] * (maximum - size))
        return {key: torch.tensor(value, dtype=torch.long) for key, value in output.items()}
