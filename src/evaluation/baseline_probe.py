"""效果诊断：仅用SFT示例构造few-shot提示，保留旧评测与严格判题。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

from src.data.schema import canonical_json
from src.data.storage import read_jsonl
from src.utils.artifacts import file_sha256


def load_training_demonstrations(release: Path) -> tuple[list[dict],dict]:
    records,hashes=[],{}
    for directory in ('statistics_full','gsm8k'):
        p=release/directory/'sft.jsonl'
        manifest=json.loads((p.parent/'manifest.json').read_text(encoding='utf-8'))
        if file_sha256(p)!=manifest['file_sha256'][p.name]: raise ValueError('SFT示例哈希不符')
        rows=read_jsonl(p)
        if any(r['split']!='sft' for r in rows): raise ValueError('示例只能来自SFT训练集')
        records.extend(rows);hashes[directory]=file_sha256(p)
    return records,hashes


def choose_demonstrations(record: dict,training: list[dict],seed: int=42,count: int=2) -> list[dict]:
    pool=sorted((r for r in training if r['task_type']==record['task_type']),key=lambda r:r['id'])
    if any(r['split']!='sft' or r['id']==record['id'] for r in pool): raise ValueError('示例混入非SFT或目标题')
    rng=random.Random(f'{seed}:{record["task_type"]}:two_shot_probe')
    return rng.sample(pool,min(count,len(pool)))


def add_demonstrations(record: dict,demonstrations: list[dict]) -> dict:
    """此函数不访问目标题answer/solution。示例答案属于训练题，不属于待测题。"""
    if any(d['split']!='sft' or d['id']==record['id'] for d in demonstrations): raise ValueError('示例来源不合法')
    pieces=['参考以下训练例题的计算方式，逐步代入并核算中间结果。回答最后写唯一一行 FINAL_ANSWER: 数值。']
    for i,d in enumerate(demonstrations,1):
        pieces.extend([f'示例{i}题目：{d["question"]}',f'示例{i}解答：{d["solution"]}'])
    pieces.extend(['现在独立解答下面的新题。不要复制示例的答案。',record['question']])
    return {**record,'question':'\n\n'.join(pieces),'original_question':record['question'],
            'demonstration_ids':[d['id'] for d in demonstrations]}


def protocol_hash(value: dict) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()
