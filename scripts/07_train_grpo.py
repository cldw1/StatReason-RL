"""Vanilla GRPO：默认4步功能训练，不使用测试集。"""
from _bootstrap import PROJECT_ROOT
from src.training.grpo import run_grpo_cli

if __name__=='__main__':
    raise SystemExit(run_grpo_cli(PROJECT_ROOT))
