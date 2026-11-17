"""LoRA SFT 真实训练入口；默认四步功能闭环。"""
from _bootstrap import PROJECT_ROOT
from src.training.sft import run_sft_cli

if __name__ == "__main__":
    raise SystemExit(run_sft_cli(PROJECT_ROOT))
