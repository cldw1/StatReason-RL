"""一题模型加载/模板/生成探测；不是正式能力评测。"""
from _bootstrap import PROJECT_ROOT
from src.evaluation.runner import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli(PROJECT_ROOT, run_type="model_probe", default_max_examples=1))
