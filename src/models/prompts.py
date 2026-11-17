"""统一训练/评测提示，只包含题面，不读取标准答案或解答。"""
from __future__ import annotations

import hashlib

SYSTEM_PROMPT = (
    "你是一名数学计算助手。请根据题目给出简短、必要的计算步骤。"
    "最终只在最后一行输出一次 FINAL_ANSWER: <数值>。"
    "不要在推导或思考中重复最终答案标记。"
    "数值可以是整数、小数、分数或科学计数法。"
    "只有题目明确允许时才使用带百分号的百分数。"
    "最后答案行后不要再添加说明。"
)
PROMPT_SHA256 = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def build_messages(record: dict) -> list[dict[str, str]]:
    if not isinstance(record.get("question"), str) or not record["question"].strip():
        raise ValueError("题面必须是非空字符串")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": record["question"]}]
