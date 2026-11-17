"""环境审计入口：实际执行 CUDA 测试，保存 JSON 和依赖快照。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import PROJECT_ROOT
from src.utils.artifacts import create_run_dir, write_json
from src.utils.environment import collect_environment, test_cuda, audit_training_api


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="必须是尚不存在的审计目录")
    parser.add_argument("--skip-gpu", action="store_true", help="只记录环境，不执行 GPU 测试")
    parser.add_argument("--require-training-stack", action="store_true", help="缺少训练依赖也返回非零退出码")
    args = parser.parse_args()
    if args.output:
        run_dir = args.output.resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = create_run_dir(PROJECT_ROOT / "outputs" / "environment", "environment")
    environment = collect_environment(PROJECT_ROOT)
    write_json(run_dir / "environment.json", environment)
    (run_dir / "pip_freeze.txt").write_text(environment.pop("pip_freeze"), encoding="utf-8")
    if args.skip_gpu:
        gpu = {"status": "not_run", "reason": "--skip-gpu"}
    else:
        try:
            gpu = test_cuda()
        except Exception as exc:
            gpu = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    write_json(run_dir / "gpu_tests.json", gpu)
    api = audit_training_api()
    write_json(run_dir / "training_api.json", api)
    summary = {
        "run_id": run_dir.name, "run_type": "environment_audit",
        "gpu_test_status": gpu["status"], "training_stack_status": api["status"],
        "training_performed": False, "model_download_performed": False,
        "artifacts": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if gpu["status"] == "failed":
        return 1
    if args.require_training_stack and api["status"] != "passed":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
