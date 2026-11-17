"""只读 RL 划分；先验证完整题面长度，禁止 TRL 静默左截断问题。"""
from collections import Counter
import json
from pathlib import Path
import random

from src.data.schema import content_fingerprint
from src.data.storage import read_jsonl
from src.models.prompts import build_messages
from src.utils.artifacts import file_sha256
from .sft_data import balanced_indices


def prepare_rl_data(tokenizer, root: Path, settings: dict, maximum_prompt: int, seed: int):
    records, fingerprints=[],{}
    for directory,source in [('statistics_full','statreason_synthetic'),('gsm8k','openai/gsm8k')]:
        manifest=json.loads((root/directory/'manifest.json').read_text(encoding='utf-8'))
        path=root/directory/'rl.jsonl'
        if file_sha256(path)!=manifest['file_sha256']['rl.jsonl']:
            raise ValueError('RL 文件哈希不符')
        rows=read_jsonl(path)
        if any(r['split']!='rl' or r['source']!=source for r in rows):
            raise ValueError('只能使用来源正确的RL训练题')
        records.extend(rows)
        fingerprints[directory]=file_sha256(path)
    if len({r['id'] for r in records})!=len(records): raise ValueError('RL id 重复')
    encoded,eligible,excluded=[],[],[]
    for record in records:
        prompt=tokenizer.apply_chat_template(build_messages(record),tokenize=False,enable_thinking=False,add_generation_prompt=True)
        ids=tokenizer.encode(prompt,add_special_tokens=False)
        if len(ids)>maximum_prompt:
            excluded.append(record['id']); continue
        encoded.append({'prompt':prompt,'record_id':record['id'],'prompt_token_count':len(ids)})
        eligible.append(record)
    maximum=settings['max_train_examples']
    quotas=settings.get('source_quotas')
    if quotas is not None:
        if maximum is None or sum(quotas.values())!=maximum or any(not isinstance(n,int) or n<0 for n in quotas.values()):
            raise ValueError('source quotas must sum to the declared pool size')
        indices=[]
        rng=random.Random(seed)
        for source,count in sorted(quotas.items()):
            candidates=[i for i,r in enumerate(eligible) if r['source']==source]
            if count>len(candidates): raise ValueError('source quota exceeds eligible RL rows')
            indices.extend(rng.sample(candidates,count))
        rng.shuffle(indices)
    else:
        indices=balanced_indices(eligible,maximum,seed)
    if maximum is not None:
        priority={task:i for i,task in enumerate(settings['smoke_task_priority'])}
        width=len(priority)
        indices=[i for start in range(0,len(indices),width)
                 for i in sorted(indices[start:start+width],key=lambda i:priority.get(eligible[i]['task_type'],width))]
    selected=[eligible[i] for i in indices]
    if not selected: raise ValueError('没有完整题面的RL样本')
    selection={'selected_ids':[r['id'] for r in selected],'candidate_count':len(records),'eligible_count':len(eligible),
               'selected_count':len(selected),'excluded_overlong_ids':excluded,
               'selected_by_source':dict(Counter(r['source'] for r in selected)),
               'selected_by_task':dict(Counter(r['task_type'] for r in selected)),
               'max_selected_prompt_tokens':max(encoded[i]['prompt_token_count'] for i in indices),
               'source_rl_file_sha256':fingerprints,'record_fingerprint':content_fingerprint(selected),
               'prompt_truncation':'reject_whole_record_before_training',
               'selection_note':settings['selection_note'],'official_test_accessed':False}
    return [encoded[i] for i in indices], {r['id']:r for r in selected}, selection
