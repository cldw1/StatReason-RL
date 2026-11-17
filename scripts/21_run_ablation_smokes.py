"""Run five bounded independent implementation checks, never formal ablations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from _bootstrap import PROJECT_ROOT
from src.utils.artifacts import utc_now, write_json, file_sha256
from src.utils.config import load_config
from src.training.dapo_mechanisms import validate_ablation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root = PROJECT_ROOT
    destination = root / 'reports/mechanism_smoke_round1'
    configs = []
    for name in ['E0', 'E1', 'E2', 'E3', 'E4']:
        path = root / f'configs/ablations/{name}_smoke.yaml'
        config = load_config(path)
        validate_ablation(config)
        configs.append((name, path, config))
    print(json.dumps({'scope': 'implementation_smokes_not_performance_evidence',
                      'max_optimizer_steps_per_arm': 4, 'arms': [n for n, _, _ in configs],
                      'paid_resources': False, 'execute': args.execute}), flush=True)
    if not args.execute:
        return 0
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {'started_at_utc': utc_now(), 'scope': 'implementation_smokes_not_formal_experiments',
                'runs': {}, 'paid_service_started': False, 'final_test_accessed': False}
    target = destination / 'manifest.json'
    if target.exists():
        raise ValueError('preserve existing smoke manifest; use individual commands for follow-up checks')
    write_json(target, manifest)
    for name, path, config in configs:
        before = {p.name for p in (root / 'outputs/runs').glob(config['experiment']['name'] + '_*')}
        log = destination / f'{name}_console.log'
        command = [sys.executable, str(root / 'scripts/07_train_grpo.py'), '--config', str(path)]
        timed_out = False
        with log.open('wb') as handle:
            try:
                returned = subprocess.run(command, cwd=root, stdout=handle, stderr=subprocess.STDOUT,
                                          timeout=config['resources']['max_train_seconds'] + 120)
                return_code = returned.returncode
            except subprocess.TimeoutExpired:
                return_code, timed_out = None, True
        created = [p for p in (root / 'outputs/runs').glob(config['experiment']['name'] + '_*') if p.name not in before]
        if len(created) != 1:
            raise RuntimeError('cannot identify exactly one new run')
        run = created[0]
        status = json.loads((run / 'run_status.json').read_text(encoding='utf-8'))
        manifest['runs'][name] = {'run': run.relative_to(root).as_posix(), 'command': command,
                                 'return_code': return_code, 'timeout': timed_out, 'status': status,
                                 'console': log.relative_to(root).as_posix(), 'console_sha256': file_sha256(log)}
        write_json(target, manifest)
        print(json.dumps({'arm': name, 'status': status['status'], 'optimizer_steps': status['optimizer_steps'],
                          'run': str(run), 'return_code': return_code, 'timeout': timed_out}), flush=True)
        if timed_out or return_code != 0:
            print('Stopped sequence after a failed check. All artifacts retained.', flush=True)
            return 1
    manifest['finished_at_utc'] = utc_now()
    write_json(target, manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
