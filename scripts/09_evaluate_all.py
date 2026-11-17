"""统一adapter开发评测入口；每次加载一个checkpoint，避免模型同时占用显存。"""
from _bootstrap import PROJECT_ROOT
from src.evaluation.runner import run_cli

if __name__=='__main__':
    raise SystemExit(run_cli(PROJECT_ROOT,run_type='post_training_development',allow_adapter=True))
