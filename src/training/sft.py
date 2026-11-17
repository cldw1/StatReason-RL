"""Transformers Trainer + PEFT 的真实 LoRA SFT，含标签审计和重载校验。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from src.data.schema import canonical_json
from src.evaluation.storage import append_prediction
from src.models.inference import load_qwen
from src.models.lora import adapter_metadata, attach_lora, capture_trainable, parameter_update_evidence, tensor_sha256
from src.models.prompts import PROMPT_SHA256
from src.models.snapshot import resolve_snapshot, snapshot_metadata
from src.utils.artifacts import create_run_dir, file_sha256, utc_now, write_json
from src.utils.config import load_config, validate_config
from src.utils.environment import collect_environment, package_versions
from .callbacks import LocalTrainingCallback
from .sft_data import AssistantCollator, EncodedDataset, prepare_sft_data


def _probe_logits(model, feature: dict):
    import torch
    model.eval()
    model.config.use_cache = False
    ids = torch.tensor([feature["input_ids"][:32]], device=model.device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=1).logits.float().cpu()


def run_sft_cli(project_root: Path) -> int:
    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    parser = argparse.ArgumentParser(description="真实 LoRA SFT；默认只运行少步 smoke")
    parser.add_argument("--config", type=Path, default=project_root / "configs/sft.yaml")
    parser.add_argument("--prepare-only", action="store_true", help="只检查真实 Tokenizer 与标签，不加载模型训练")
    parser.add_argument("--resume", type=Path, help="恢复原 SFT 运行目录中的最近检查点")
    args = parser.parse_args()
    config = load_config(args.config)
    if config["experiment"]["stage"] != "sft":
        raise ValueError("此入口只接受 SFT 配置")
    if args.resume:
        run = args.resume.resolve()
        previous_status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
        if previous_status["status"] == "completed":
            print(json.dumps(previous_status, ensure_ascii=False, indent=2))
            return 0
    else:
        run = create_run_dir(project_root / config["paths"]["run_root"], config["experiment"]["name"])
    run = run.resolve()
    if not run.is_relative_to(project_root.resolve()):
        raise ValueError("训练及检查点轮换目录必须位于当前项目内")
    started = time.perf_counter()
    status = {"run_id": run.name, "run_type": "lora_sft", "level": config["experiment"]["level"],
              "status": "started", "started_at_utc": utc_now(), "official_test_accessed": False,
              "model_training_performed": False, "optimizer_steps": 0}
    model = trainer = None
    train_resources = None
    exit_code = 0
    try:
        snapshot = resolve_snapshot(config["model"])
        config["model"]["revision"] = snapshot.revision
        tokenizer = AutoTokenizer.from_pretrained(snapshot.path, local_files_only=True, trust_remote_code=False, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_rows, validation_rows, selection, label_audits = prepare_sft_data(
            tokenizer, project_root / config["paths"]["dataset_release"], config["sft"], config["reproducibility"]["seed"])
        config["data"]["fingerprint"] = selection["sft"]["encoded_fingerprint"]
        validate_config(config)
        identity_data = {"config": config, "selection": selection, "prompt_sha256": PROMPT_SHA256,
                         "versions": package_versions(), "preprocessor_sha256": file_sha256(project_root / "src/training/sft_data.py")}
        signature = hashlib.sha256(canonical_json(identity_data).encode()).hexdigest()
        checkpoint = None
        if args.resume:
            old_identity = json.loads((run / "training_identity.json").read_text(encoding="utf-8"))
            if old_identity["signature"] != signature:
                raise ValueError("恢复时配置、数据、Tokenizer 前处理或环境已改变，必须新建运行")
            checkpoint = get_last_checkpoint(str(run / "checkpoints"))
            if not checkpoint:
                raise ValueError("没有保存完整 Trainer 检查点，无法严格断点续训")
        else:
            write_json(run / "resolved_config.yaml", config)
            write_json(run / "environment.json", collect_environment(project_root))
            write_json(run / "model_snapshot.json", snapshot_metadata(snapshot))
            write_json(run / "data_selection.json", selection)
            write_json(run / "training_identity.json", {"signature": signature, "prompt_sha256": PROMPT_SHA256})
            for audit in label_audits:
                append_prediction(run / "label_audit.jsonl", audit)
        write_json(run / "run_status.json", status)
        summary_fields = ("candidate_count", "eligible_count", "selected_count", "selected_by_source", "max_selected_sequence_length", "encoded_fingerprint")
        data_summary = {split: {**{key: values[key] for key in summary_fields}, "excluded_overlong_count": len(values["excluded_overlong_ids"])} for split, values in selection.items()}
        print(json.dumps({"run": str(run), "data": data_summary}, ensure_ascii=False), flush=True)
        if args.prepare_only:
            status["status"] = "data_prepared_no_training"
        else:
            set_seed(config["reproducibility"]["seed"])
            base, _, load_resources = load_qwen(snapshot, config["model"])
            model = attach_lora(base, config["lora"])
            del base
            model.config.use_cache = False
            if config["trainer"]["gradient_checkpointing"]:
                model.enable_input_require_grads()
            initial = capture_trainable(model)
            base_hashes = {name: tensor_sha256(parameter) for name, parameter in model.named_parameters() if name.endswith("base_layer.weight") and ("layers.0." in name)}
            trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            total = sum(parameter.numel() for parameter in model.parameters())
            write_json(run / "parameter_counts.json", {"trainable": trainable, "total": total, "fraction": trainable / total,
                                                       "trainable_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters() if parameter.requires_grad})})
            print(json.dumps({"trainable_parameters": trainable, "total_parameters": total}, ensure_ascii=False), flush=True)
            callback = LocalTrainingCallback(run, max_train_seconds=config["resources"]["max_train_seconds"])
            trainer_args = dict(config["trainer"])
            trainer_args.update(output_dir=str(run / "checkpoints"), seed=config["reproducibility"]["seed"],
                                data_seed=config["reproducibility"]["data_seed"], per_device_eval_batch_size=1,
                                label_names=["labels"], prediction_loss_only=True, logging_nan_inf_filter=False,
                                disable_tqdm=True, save_safetensors=True, push_to_hub=False, torch_compile=False, tf32=False)
            training_args = TrainingArguments(**trainer_args)
            trainer = Trainer(model=model, args=training_args, train_dataset=EncodedDataset(train_rows),
                              eval_dataset=EncodedDataset(validation_rows), processing_class=tokenizer,
                              data_collator=AssistantCollator(tokenizer.pad_token_id), callbacks=[callback])
            before_loss = trainer.evaluate(metric_key_prefix="before_validation")
            write_json(run / "validation_before.json", before_loss)
            torch.cuda.reset_peak_memory_stats(0)
            training_started = time.perf_counter()
            status["training_attempted"] = True
            write_json(run / "run_status.json", status)
            output = trainer.train(resume_from_checkpoint=checkpoint)
            torch.cuda.synchronize()
            training_seconds = time.perf_counter() - training_started
            after_loss = trainer.evaluate(metric_key_prefix="after_validation")
            write_json(run / "validation_after.json", after_loss)
            train_resources = {"training_seconds": training_seconds,
                               "session_optimizer_steps": len(callback.step_durations), "global_optimizer_steps": trainer.state.global_step,
                               "step_seconds": callback.step_durations,
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated(0), "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                               "model_load": load_resources, "budget_stopped": callback.budget_stopped}
            status["optimizer_steps"] = trainer.state.global_step
            status["model_training_performed"] = trainer.state.global_step > 0
            write_json(run / "train_result.json", output.metrics)
            evidence = parameter_update_evidence(model, initial)
            current_base_hashes = {name: tensor_sha256(parameter) for name, parameter in model.named_parameters() if name in base_hashes}
            if current_base_hashes != base_hashes:
                raise RuntimeError("抽查的冻结 Base 参数发生变化")
            evidence["sampled_frozen_base_tensor_hashes_unchanged"] = True
            write_json(run / "parameter_updates.json", evidence)
            model.gradient_checkpointing_disable()
            model.eval()
            logits_before = _probe_logits(model, train_rows[0])
            adapter_state = {name: value.detach().cpu().clone() for name, value in get_peft_model_state_dict(model).items()}
            adapter_path = run / "adapter"
            model.save_pretrained(adapter_path, safe_serialization=True, save_embedding_layers=False)
            tokenizer.save_pretrained(adapter_path)
            metadata = adapter_metadata(adapter_path, snapshot.repository, snapshot.revision)
            best_checkpoint = trainer.state.best_model_checkpoint
            selected_step = int(Path(best_checkpoint).name.rsplit("-", 1)[1]) if best_checkpoint else trainer.state.global_step
            metadata.update(run_id=run.name, training_level=config["experiment"]["level"],
                            training_signature=signature, training_steps=selected_step,
                            optimization_steps_executed=trainer.state.global_step,
                            selected_checkpoint=best_checkpoint)
            write_json(adapter_path / "adapter_metadata.json", metadata)
            trainer.save_state()
            status["best_model_checkpoint"] = trainer.state.best_model_checkpoint
            status["selected_checkpoint_step"] = selected_step
            del trainer, model
            trainer = model = None
            gc.collect()
            torch.cuda.empty_cache()
            fresh_base, _, _ = load_qwen(snapshot, config["model"])
            reloaded = PeftModel.from_pretrained(fresh_base, str(adapter_path), is_trainable=False, local_files_only=True)
            reloaded.eval()
            state_after = get_peft_model_state_dict(reloaded)
            if set(adapter_state) != set(state_after):
                raise RuntimeError("重载 adapter 的参数键不同")
            for name, value in adapter_state.items():
                torch.testing.assert_close(value, state_after[name].detach().cpu(), rtol=0, atol=0)
            logits_after = _probe_logits(reloaded, train_rows[0])
            torch.testing.assert_close(logits_before, logits_after, rtol=0, atol=0)
            write_json(run / "reload_verification.json", {"adapter_parameters_exact_match": True,
                                                          "probe_logits_exact_match": True,
                                                          "max_abs_logits_error": (logits_before - logits_after).abs().max().item(),
                                                          "probe_prompt_tokens": min(32, len(train_rows[0]["input_ids"]))})
            status.update(status="budget_stopped" if callback.budget_stopped else "completed", adapter=str(adapter_path),
                          trainable_parameter_count=evidence["trainable_parameter_count"],
                          changed_parameter_tensors=evidence["changed_tensor_count"], reload_verified=True)
    except KeyboardInterrupt:
        status.update(status="interrupted", error="KeyboardInterrupt")
        exit_code = 130
    except Exception as error:
        from src.evaluation.runner import _safe_error
        status.update(status="failed", error_type=type(error).__name__, error=_safe_error(error))
        exit_code = 1
    finally:
        if trainer is not None:
            status["optimizer_steps"] = trainer.state.global_step
            status["model_training_performed"] = trainer.state.global_step > 0
        status.update(finished_at_utc=utc_now(), session_wall_seconds=time.perf_counter() - started)
        write_json(run / "run_status.json", status)
        write_json(run / "resource_metrics.json", {"session_wall_seconds": status["session_wall_seconds"],
                                                  "training": train_resources, "paid_service_started": False})
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        print(f"Run records: {run}", flush=True)
    return exit_code
