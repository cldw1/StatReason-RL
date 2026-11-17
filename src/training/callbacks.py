"""本地真实训练日志、有限值检查与资源记录；不向外部服务上传。"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

from transformers import TrainerCallback

from src.evaluation.storage import append_prediction
from src.utils.artifacts import utc_now


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise FloatingPointError("训练日志出现 NaN/Inf")
    return value


class LocalTrainingCallback(TrainerCallback):
    def __init__(self, run: Path, max_train_seconds: float, stop_at_step_multiple: int = 1):
        self.run = run
        self.started = time.perf_counter()
        self.last_step = self.started
        self.step_durations = []
        self.max_train_seconds = max_train_seconds
        self.budget_stopped = False
        self.stop_at_step_multiple = stop_at_step_multiple

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = self.last_step = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        import torch
        torch.cuda.synchronize()
        now = time.perf_counter()
        duration = now - self.last_step
        self.last_step = now
        self.step_durations.append(duration)
        if now - self.started > self.max_train_seconds and state.global_step % self.stop_at_step_multiple == 0:
            self.budget_stopped = True
            control.should_training_stop = True
        append_prediction(self.run / "train_metrics.jsonl", {
            "event": "optimizer_step", "step": state.global_step, "recorded_at_utc": utc_now(),
            "step_seconds": duration, "elapsed_train_seconds": now - self.started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0)})

    def on_log(self, args, state, control, logs=None, **kwargs):
        record = {"event": "trainer_log", "step": state.global_step, "recorded_at_utc": utc_now(), **_plain(logs or {})}
        append_prediction(self.run / "train_metrics.jsonl", record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
