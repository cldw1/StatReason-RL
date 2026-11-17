"""构建统计题原型/正式候选数据，以及仅来源于 train 的 GSM8K 开发划分。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from _bootstrap import PROJECT_ROOT
from src.data.gsm8k import prepare_gsm8k
from src.data.quality import audit_statistics, training_review_markdown
from src.data.statistics import generate_statistics
from src.data.storage import save_dataset
from src.utils.artifacts import create_run_dir, file_sha256, utc_now, write_json
from src.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/data.yaml")
    parser.add_argument("--statistics", choices=["prototype", "full", "both", "none"], default="prototype")
    parser.add_argument("--gsm8k", action="store_true")
    parser.add_argument("--gsm8k-revision")
    parser.add_argument("--gsm8k-local-train", type=Path)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data/processed/statreason-v1")
    args = parser.parse_args()
    config = load_config(args.config)
    run = create_run_dir(PROJECT_ROOT / "outputs/data_preparation", "prepare")
    write_json(run / "resolved_config.yaml", {"data_config": config, "arguments": vars(args) | {key: str(value) for key, value in vars(args).items() if isinstance(value, Path)}})
    results = {"run_id": run.name, "recorded_at_utc": utc_now(), "statistics": {}, "gsm8k": {"status": "not_requested"},
               "official_test_downloaded": False, "model_training_performed": False}
    source_files = [PROJECT_ROOT / f"src/data/{name}.py" for name in ("statistics", "schema", "storage", "quality", "gsm8k")]
    hashes = {str(path.relative_to(PROJECT_ROOT)): file_sha256(path) for path in source_files}
    for mode in ("prototype", "full"):
        if args.statistics not in (mode, "both"):
            continue
        counts = {"prototype": config["statistics"]["prototype_count"]} if mode == "prototype" else {name: config["statistics"][f"{name}_count"] for name in ("sft", "rl", "validation", "test")}
        records = generate_statistics(counts, seed=config["seed"])
        quality = audit_statistics(records)
        destination = args.output_root / f"statistics_{mode}"
        manifest = save_dataset(destination, records, {"seed": config["seed"], "version": "statreason-v1",
                                "source_hashes": hashes, "split_unit": "template_family", "quality": quality,
                                "status": "prototype_only" if mode == "prototype" else "candidate_release_not_evaluation_frozen"})
        write_json(run / f"statistics_{mode}_quality.json", quality)
        if mode == "full":
            (run / "training_sample_review.md").write_text(training_review_markdown(records), encoding="utf-8")
        results["statistics"][mode] = {"status": "prepared", "directory": str(destination), "statistics": manifest["statistics"], "quality": quality}
    if args.gsm8k:
        try:
            gsm_config = config["gsm8k"]
            counts = {name: gsm_config[f"{name}_count"] for name in ("sft", "rl", "validation", "reserved")}
            records, provenance = prepare_gsm8k(PROJECT_ROOT / "data/raw/huggingface", counts, seed=config["seed"],
                                               revision=args.gsm8k_revision or gsm_config["revision"], local_train=args.gsm8k_local_train)
            manifest = save_dataset(args.output_root / "gsm8k", records, provenance | {"source_hashes": hashes})
            results["gsm8k"] = {"status": "prepared", "statistics": manifest["statistics"], "provenance": provenance}
        except Exception as exc:
            # 保留可诊断的错误原因，去掉 URL 凭证和常见 token 字符串。
            message = re.sub(r"(https?://)[^/\s@]+@", r"\1[redacted]@", str(exc))
            message = re.sub(r"hf_[A-Za-z0-9]+", "[redacted_token]", message)
            message = re.sub(r"(?i)(token|api_key|access_token)=([^&\s]+)", r"\1=[redacted]", message)
            results["gsm8k"] = {"status": "blocked_or_failed", "exception_type": type(exc).__name__,
                                 "error": message,
                                 "action": "check_network_or_supply_official_train_parquet_with_commit"}
            (run / "gsm8k_error.txt").write_text(f"{type(exc).__name__}\n{message}\n", encoding="utf-8")
    write_json(run / "preparation_summary.json", results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Run records: {run}")
    return 2 if results["gsm8k"]["status"] == "blocked_or_failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
