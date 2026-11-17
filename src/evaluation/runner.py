"""原始模型开发评测入口；结果逐题持久化，缺权重时诚实保存阻塞状态。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from src.data.schema import canonical_json
from src.models.inference import example_seed, generate_completions, load_qwen
from src.models.prompts import PROMPT_SHA256
from src.models.snapshot import ModelNotCachedError, resolve_snapshot, snapshot_metadata
from src.rewards.numeric import verify_completion
from src.utils.artifacts import create_run_dir, utc_now, write_json
from src.utils.config import load_config, validate_config
from src.utils.environment import collect_environment, package_versions
from .metrics import summarize_predictions
from .panel import select_development_panel
from .storage import append_prediction, export_csv, read_predictions


def assemble_prediction(record: dict, generated: dict, *, mode: str, model_name: str,
                        revision: str, signature: str) -> dict:
    """从 generated['completion'] 判题，不能从 prompt_text 或标准解答判题。"""
    result = verify_completion(generated["completion"], record, truncated=generated["truncated"])
    fields = ("id", "source", "task_type", "template_family", "question", "parameters", "answer",
              "answer_type", "answer_unit", "absolute_tolerance", "relative_tolerance", "allow_percentage")
    return {**{key: record[key] for key in fields}, 'evaluation_stratum':record.get('evaluation_stratum','unspecified'), **generated, "mode": mode,
            "model_name": model_name, "model_revision": revision, "evaluation_signature": signature,
            "verification": result.to_dict(), "recorded_at_utc": utc_now()}


def _safe_error(error: BaseException) -> str:
    message = re.sub(r"(https?://)[^/\s@]+@", r"\1[redacted]@", str(error))
    message = re.sub(r"hf_[A-Za-z0-9]+", "[redacted_token]", message)
    return re.sub(r"(?i)(token|api_key|access_token)=([^&\s]+)", r"\1=[redacted]", message)


def _write_observations(run: Path, rows: list[dict], config: dict) -> None:
    if rows:
        write_json(run / "evaluation_summary.json", summarize_predictions(rows, sampled=config["evaluation"]["do_sample"]))
        export_csv(run / "evaluation_predictions.csv", rows)


def run_cli(project_root: Path, *, run_type: str, default_max_examples: int | None = None, allow_adapter: bool = False) -> int:
    parser = argparse.ArgumentParser(description="Qwen3-0.6B 验证集开发评测，不读取官方测试集")
    parser.add_argument("--config", type=Path, default=project_root / "configs/evaluation.yaml")
    parser.add_argument("--mode", choices=("non-thinking", "thinking", "both"))
    parser.add_argument("--download-model", action="store_true", help="允许下载目标模型并固定 commit")
    parser.add_argument("--prepare-only", action="store_true", help="只准备模型快照，不加载或生成")
    parser.add_argument("--resume", type=Path, help="恢复同一个运行目录，保留已完成的逐题记录")
    if allow_adapter:
        parser.add_argument("--adapter", type=Path, required=True, help="含真实校验元数据的 LoRA adapter 目录")
    args = parser.parse_args()
    config = load_config(args.config)
    validate_config(config)
    if config['evaluation']['num_return_sequences'] != 1 or config['evaluation']['batch_size'] < 1:
        raise ValueError('开发入口每题一次生成，batch_size 必须为正')
    if default_max_examples is not None:
        config["evaluation"]["max_examples"] = default_max_examples
    selected_mode = args.mode or config["evaluation"]["default_mode"]
    modes = ["non-thinking", "thinking"] if selected_mode == "both" else [selected_mode]
    if args.resume:
        run = args.resume.resolve()
        if not run.is_dir() or not (run / "run_status.json").exists():
            raise ValueError("不是可恢复的评测运行目录")
    else:
        run = create_run_dir(project_root / config["paths"]["evaluation_root"], run_type)
    status_path = run / "run_status.json"
    status = {"run_id": run.name, "run_type": run_type, "scope": "level0_development_validation_only",
              "status": "started", "modes": modes, "model_training_performed": False,
              "official_test_accessed": False, "started_at_utc": utc_now()}
    started = time.perf_counter()
    rows = []
    load_metrics = None
    snapshot_info = None
    exit_code = 0
    try:
        panel, selection = select_development_panel(project_root / config["paths"]["dataset_release"], config["evaluation"], config["reproducibility"]["seed"])
        if not args.resume:
            write_json(run / "resolved_config.yaml", config)
            write_json(run / "environment.json", collect_environment(project_root))
            write_json(run / "selection.json", selection)
            write_json(status_path, status)
        snapshot = resolve_snapshot(config["model"], allow_download=args.download_model,
                                    max_download_gib=config["resources"]["max_model_download_gib"])
        snapshot_info = snapshot_metadata(snapshot)
        adapter_info = None
        if allow_adapter:
            from src.models.lora import validate_adapter
            adapter_info = validate_adapter(args.adapter.resolve(), snapshot.repository, snapshot.revision)
        config["model"]["revision"] = snapshot.revision
        config["data"]["fingerprint"] = selection["panel_fingerprint"]
        signature_data = {"config": config, "modes": modes, "selection": selection, "snapshot": snapshot_info,
                          "prompt_sha256": PROMPT_SHA256, "versions": package_versions(), "run_type": run_type}
        if adapter_info is not None:
            signature_data["adapter"] = adapter_info
        if config['evaluation']['batch_size'] > 1:
            signature_data['sampling_protocol'] = 'fixed_order_left_padded_batch_seed_v1'
        signature = hashlib.sha256(canonical_json(signature_data).encode()).hexdigest()
        if args.resume:
            previous = json.loads((run / "evaluation_identity.json").read_text(encoding="utf-8"))
            if previous["signature"] != signature:
                raise ValueError("恢复时模型、数据、提示、依赖或评测配置已改变，必须新建运行")
            rows = read_predictions(run / "evaluation_predictions.jsonl", repair_partial_last_line=True)
            if any(row["evaluation_signature"] != signature for row in rows):
                raise ValueError("预测记录与当前实验标识不一致")
            summarize_predictions(rows, sampled=config["evaluation"]["do_sample"])
        else:
            write_json(run / "resolved_config.yaml", config)
            write_json(run / "model_snapshot.json", snapshot_info)
            write_json(run / "evaluation_identity.json", {"signature": signature, "prompt_sha256": PROMPT_SHA256})
            if adapter_info is not None:
                write_json(run / "adapter_identity.json", adapter_info)
        write_json(project_root / "data/metadata/model_snapshot.json", snapshot_info)
        if args.prepare_only:
            status["status"] = "snapshot_prepared_no_generation"
        else:
            print(f"Loading {snapshot.repository} @ {snapshot.revision}", flush=True)
            model, tokenizer, load_metrics = load_qwen(snapshot, config["model"])
            if adapter_info is not None:
                import torch
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(args.adapter.resolve()), is_trainable=False, local_files_only=True)
                model.eval()
                model.config.use_cache = True
                load_metrics["adapter"] = adapter_info
                load_metrics["load_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
                load_metrics["load_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            print(json.dumps({"model_load": load_metrics}, ensure_ascii=False), flush=True)
            write_json(run / "model_load.json", load_metrics)
            completed = {(row["mode"], row["id"]) for row in rows}
            batch_size = config['evaluation']['batch_size']
            for mode in modes:
                for start in range(0,len(panel),batch_size):
                    batch = panel[start:start+batch_size]
                    if all((mode,r['id']) in completed for r in batch):
                        continue
                    if time.perf_counter()-started > config['resources'].get('max_evaluation_seconds',float('inf')):
                        raise TimeoutError('达到本次评测时限，已保存回答可用 --resume 恢复')
                    key = batch[0]['id'] if len(batch)==1 else canonical_json([r['id'] for r in batch])
                    seed = example_seed(config['reproducibility']['seed'],key)
                    generated_batch = generate_completions(model,tokenizer,batch,config['evaluation'],enable_thinking=mode=='thinking',seed=seed)
                    append_prediction(run/'generation_attempts.jsonl',{'mode':mode,'ids':[r['id'] for r in batch],
                        'generated_tokens':sum(g['generated_token_count'] for g in generated_batch),
                        'seconds':sum(g['generation_seconds'] for g in generated_batch),'seed':seed,
                        'recomputed_ids':[r['id'] for r in batch if (mode,r['id']) in completed], 'recorded_at_utc':utc_now()})
                    for record,generated in zip(batch,generated_batch,strict=True):
                        if (mode,record['id']) in completed:
                            continue
                        row = assemble_prediction(record,generated,mode=mode,model_name=snapshot.repository,revision=snapshot.revision,signature=signature)
                        append_prediction(run/'evaluation_predictions.jsonl',row)
                        rows.append(row)
                        completed.add((mode,record['id']))
                    _write_observations(run, rows, config)
                    print(json.dumps({'mode':mode,'saved':len(rows),'total':len(panel)*len(modes),
                                      'batch_tokens':sum(g['generated_token_count'] for g in generated_batch)},ensure_ascii=False),flush=True)
            status["status"] = "completed"
    except KeyboardInterrupt:
        status.update(status="interrupted", error="KeyboardInterrupt")
        exit_code = 130
    except Exception as error:
        blocked = isinstance(error, ModelNotCachedError) or type(error).__name__ in {"ProxyError", "ConnectTimeout", "ReadTimeout", "ConnectionError"}
        status.update(status="blocked" if blocked else "failed", error_type=type(error).__name__, error=_safe_error(error))
        exit_code = 2 if blocked else 1
    finally:
        status.update(finished_at_utc=utc_now(), completed_prediction_count=len(rows), snapshot_resolved=snapshot_info is not None,
                      model_loaded=load_metrics is not None)
        write_json(status_path, status)
        _write_observations(run, rows, config)
        attempts=read_predictions(run/'generation_attempts.jsonl',repair_partial_last_line=True)
        write_json(run / "resource_metrics.json", {
            "session_wall_seconds": time.perf_counter() - started,
            "model_load": load_metrics, "recorded_generated_tokens": sum(row["generated_token_count"] for row in rows),
            "recorded_generation_seconds": sum(row["generation_seconds"] for row in rows),
            'all_returned_attempt_tokens_including_recomputed_batches':sum(a['generated_tokens'] for a in attempts),
            'all_returned_attempt_seconds_including_recomputed_batches':sum(a['seconds'] for a in attempts),
            'attempt_log_covers_run': bool(attempts) and sum(len(a['ids']) for a in attempts)>=len(rows),
            "peak_allocated_bytes": max([row["peak_allocated_bytes"] for row in rows] + ([load_metrics["load_peak_allocated_bytes"]] if load_metrics else []), default=None),
            "peak_reserved_bytes": max([row["peak_reserved_bytes"] for row in rows] + ([load_metrics["load_peak_reserved_bytes"]] if load_metrics else []), default=None),
            "scope": "completed_generation_records_including_resumed_rows_session_wall_is_current_process_only",
            "inflight_interrupted_generation_cost": "not_observable_if_interrupted_before_return", "paid_service_started": False,
        })
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    print(f"Run records: {run}", flush=True)
    return exit_code
