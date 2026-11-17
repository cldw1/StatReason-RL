"""使用与 Base 相同的开发题组、提示和采样设置评测真实 SFT adapter。"""
from _bootstrap import PROJECT_ROOT
from src.evaluation.runner import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli(PROJECT_ROOT, run_type="sft_development", allow_adapter=True))
