"""从已完成的真实 SFT 日志绘图；不生成或补齐不存在的指标。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.utils.artifacts import file_sha256, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()
    run = args.sft_run.resolve()
    status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed" or not status["model_training_performed"]:
        raise ValueError("只对已完成真实训练的运行绘图")
    if status["level"] == 0 and not args.include_smoke:
        raise ValueError("Level 0 图必须显式 include-smoke，不能放入正式性能展示")
    source = run / "train_metrics.jsonl"
    metrics = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    train = [(row["step"], row["loss"]) for row in metrics if row.get("event") == "trainer_log" and "loss" in row]
    validation = {}
    for row in metrics:
        for name in ("before_validation_loss", "eval_loss", "after_validation_loss"):
            if name in row:
                if name == "after_validation_loss":
                    # 正式训练末尾可能回载较早的最佳检查点，不能覆盖末步的实测曲线。
                    validation.setdefault(row["step"], row[name])
                else:
                    validation[row["step"]] = row[name]
    if not train or not validation:
        raise ValueError("缺少真实训练或验证损失，不能绘图")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].plot(*zip(*train), color="#245B9B", linewidth=1.7, marker="o", markersize=3)
    axes[0].set(title="Assistant-token training loss", xlabel="Optimizer step", ylabel="Cross entropy")
    points = sorted(validation.items())
    axes[1].plot(*zip(*points), color="#B35C25", linewidth=1.7, marker="o", markersize=4)
    axes[1].set(title="Fixed validation subset loss", xlabel="Optimizer step", ylabel="Cross entropy")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    label = "SMOKE TEST" if status["level"] == 0 else "Local SFT experiment"
    figure.suptitle(f"{label} | {run.name}", fontsize=9)
    figure.tight_layout()
    output = args.output.resolve() if args.output else run / "figures"
    output.mkdir(parents=True, exist_ok=True)
    path = output / "sft_loss.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    write_json(output / "sft_loss_source.json", {"run_id": run.name, "level": status["level"],
               "source_log": str(source), "source_sha256": file_sha256(source),
               "training_points": len(train), "validation_points": len(points),
               "chart_sha256": file_sha256(path), "simulated_data": False})
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
