"""原始 Qwen 开发题组评测，支持两种模式与中断恢复。"""
from _bootstrap import PROJECT_ROOT
from src.evaluation.runner import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli(PROJECT_ROOT, run_type="base_development"))
