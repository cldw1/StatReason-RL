"""简洁中文训练入口：完整输出写入文件，终端只显示必要进度。"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from _bootstrap import PROJECT_ROOT
from src.utils.artifacts import create_run_dir, utc_now, write_json
from src.utils.config import load_config


LABELS = {'E0': 'Vanilla GRPO', 'E1': 'Clip-Higher', 'E2': '动态采样',
          'E3': 'Token级损失', 'E4': 'DAPO组合'}
STATES = {'completed': '已完成', 'sampling_budget_stopped': '达到补采上限，提前停止',
          'budget_stopped': '达到时间上限，提前停止', 'failed': '运行失败',
          'interrupted': '已中断', 'started': '运行中或上次未正常退出'}


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None


def latest_run(config):
    prefix = config['experiment']['name'] + '_*'
    candidates = list((PROJECT_ROOT / config['paths']['run_root']).glob(prefix))
    return max(candidates, key=lambda p: p.name) if candidates else None


def progress(run):
    if run is None:
        return {}
    path = run / 'train_metrics.jsonl'
    latest = {}
    try:
        with path.open(encoding='utf-8') as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get('event') == 'optimizer_step':
                    latest = row
    except OSError:
        pass
    return latest


def show_status():
    for key, label in LABELS.items():
        config = load_config(PROJECT_ROOT / f'configs/ablations/{key}_100.yaml')
        run = latest_run(config)
        status = read_json(run / 'run_status.json') if run else None
        if status:
            print(f"{key} {label}：{STATES.get(status['status'], status['status'])}，"
                  f"{status.get('optimizer_steps', 0)}/{config['trainer']['max_steps']}步", flush=True)
        else:
            print(f'{key} {label}：尚未运行', flush=True)


def monitor(command, log, config, interval):
    before = set((PROJECT_ROOT / config['paths']['run_root']).glob(config['experiment']['name'] + '_*'))
    environment = {**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'}
    run = None
    started = last_output = time.monotonic()
    printed_step = 0
    interrupted = False
    with log.open('wb') as handle:
        child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT, env=environment)
        try:
            while child.poll() is None:
                if run is None:
                    created = set((PROJECT_ROOT / config['paths']['run_root']).glob(config['experiment']['name'] + '_*')) - before
                    if len(created) == 1:
                        run = created.pop()
                row = progress(run)
                step = row.get('step', 0)
                now = time.monotonic()
                if step >= printed_step + interval:
                    memory = row.get('peak_allocated_bytes', 0) / 2**30
                    print(f"  已完成 {step}/{config['trainer']['max_steps']}步｜"
                          f"用时{(now-started)/60:.1f}分钟｜峰值显存{memory:.2f} GiB", flush=True)
                    printed_step, last_output = step, now
                elif now - last_output >= 30:
                    print(f"  已完成 {step}/{config['trainer']['max_steps']}步，正在生成、筛选或计算下一批回答……", flush=True)
                    last_output = now
                time.sleep(1)
        except KeyboardInterrupt:
            interrupted = True
            print('\n收到中断，正在结束子进程并保留现有文件。', flush=True)
            try:
                # The terminal Ctrl+C also reaches the child in the same console.
                child.wait(timeout=15)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                child.terminate()
        return_code = child.wait()
    if run is None:
        created = set((PROJECT_ROOT / config['paths']['run_root']).glob(config['experiment']['name'] + '_*')) - before
        run = created.pop() if len(created) == 1 else None
    status = read_json(run / 'run_status.json') if run else None
    return run, status, 130 if interrupted else return_code


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', nargs='+', choices=list(LABELS), default=['E3', 'E4'])
    parser.add_argument('--status', action='store_true', help='只查看已有进度，不启动训练')
    parser.add_argument('--dry-run', action='store_true', help='只显示计划，不启动训练')
    parser.add_argument('--print-every', type=int, default=10, help='每多少步显示一条进度')
    args = parser.parse_args()
    if args.status:
        show_status()
        return 0
    if args.print_every < 1 or len(args.experiments) != len(set(args.experiments)):
        parser.error('进度间隔必须为正整数，实验组不能重复')
    planned = []
    for key in args.experiments:
        path = PROJECT_ROOT / f'configs/ablations/{key}_100.yaml'
        config = load_config(path)
        run = latest_run(config)
        status = read_json(run / 'run_status.json') if run else None
        if status:
            if status['status'] == 'completed' and status.get('optimizer_steps') == config['trainer']['max_steps']:
                saved = read_json(run / 'resolved_config.yaml')
                if saved != config:
                    print(f'{key}已有完成结果，但配置不同；已保留，请先核对实验版本。', flush=True)
                    return 2
                print(f'{key}已完成，自动跳过。', flush=True)
                continue
            print(f"{key}已有{status.get('optimizer_steps', 0)}步结果：{STATES.get(status['status'], status['status'])}。"
                  '本入口不自动覆盖或原配置重跑，请先处理该组。', flush=True)
            return 2
        planned.append((key, path, config))
    if not planned:
        print('所选实验都已完成。', flush=True)
        return 0
    print('本次依次运行：' + ' → '.join(key for key, _, _ in planned) + '。每组100步。', flush=True)
    if args.dry_run:
        print('仅核对计划，未启动训练。', flush=True)
        return 0
    batch = create_run_dir(PROJECT_ROOT / 'outputs/batches', 'ablations_quiet')
    record = {'started_at_utc': utc_now(), 'runs': [], 'purpose': 'quiet local training wrapper'}
    write_json(batch / 'batch_status.json', record)
    for key, path, config in planned:
        log = batch / f'{key}_console.log'
        command = [sys.executable, '-u', str(PROJECT_ROOT / 'scripts/07_train_grpo.py'), '--config', str(path)]
        print(f'\n开始 {key}：{LABELS[key]}。详细输出自动写入文件。', flush=True)
        run, status, code = monitor(command, log, config, args.print_every)
        record['runs'].append({'experiment': key, 'run': str(run) if run else None,
                               'console': str(log), 'return_code': code, 'status': status})
        write_json(batch / 'batch_status.json', record)
        if status and status['status'] == 'completed' and status.get('optimizer_steps') == config['trainer']['max_steps'] and code == 0:
            print(f"{key}完成：{status['optimizer_steps']}步，结果已保存。", flush=True)
            continue
        if status:
            print(f"{key}：{STATES.get(status['status'], status['status'])}，实际完成{status.get('optimizer_steps', 0)}步。", flush=True)
            if status['status'] == 'sampling_budget_stopped':
                print('原因：回答组始终同对或同错，达到补采上限。已有结果保留，尚未完成100步。', flush=True)
        else:
            print(f'{key}未正常完成，现有日志已保留。', flush=True)
        print(f'详细日志：{log}\n已停止后续实验，请把最后这几行发来。', flush=True)
        return code if code else 2
    record['finished_at_utc'] = utc_now()
    write_json(batch / 'batch_status.json', record)
    print(f'\n本次所选实验完成。批次记录：{batch}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
