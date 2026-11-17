"""统一封存评测；中文进度、完整日志落盘，同一命令可恢复。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from _bootstrap import PROJECT_ROOT
from src.data.schema import canonical_json
from src.evaluation.ablation_protocol import prepare_plan, load_frozen_panel, read
from src.evaluation.storage import append_prediction, read_predictions, export_csv
from src.evaluation.metrics import summarize_predictions
from src.utils.artifacts import file_sha256, write_json, utc_now


def summary(rows):
    value = summarize_predictions(rows)
    value.update(scope='frozen_heldout_records_not_official_gsm8k_test',
                 limitations='one training seed and one sampled answer per question; related structures exist in prior development')
    for mode, part in value['by_mode'].items():
        part['by_structural_difficulty'] = {
            name: summarize_predictions([r for r in rows if r['mode'] == mode and r['difficulty'] == name])['by_mode'][mode]
            for name in sorted({r['difficulty'] for r in rows if r['mode'] == mode})}
    return value


def verify_policy(root, policy):
    if not policy['adapter']:
        return
    path = root / policy['adapter']
    for filename, key in [('adapter_model.safetensors', 'adapter_safetensors_sha256'), ('adapter_config.json', 'adapter_config_sha256')]:
        if file_sha256(path / filename) != policy[key]:
            raise ValueError('frozen policy weights or LoRA configuration changed')
    if path.name.startswith('checkpoint-') and read(path / 'trainer_state.json')['global_step'] != policy['optimizer_step']:
        raise ValueError('checkpoint optimizer step changed')


def worker(max_session_minutes):
    root = PROJECT_ROOT
    plan = prepare_plan(root)
    output = root / 'outputs/ablation_evaluation_round1'
    started = time.perf_counter()
    state = {'status': 'started', 'plan_signature': plan['signature'], 'started_at_utc': utc_now(),
             'current_model': None, 'model_completed_questions': 0, 'questions_per_model': len(plan['ordered_question_ids']),
             'models_completed': 0, 'models_total': len(plan['models']), 'model_training_performed': False,
             'final_test_bodies_accessed': False, 'official_gsm8k_test_accessed': False}
    write_json(output / 'study_status.json', state)
    model = tokenizer = None
    code = 0
    try:
        panel, panel_identity = load_frozen_panel(root, plan)
        state['final_test_bodies_accessed'] = True
        write_json(output / 'panel_identity.json', panel_identity)
        panel_path = output / 'panel.jsonl'
        serialized = ''.join(canonical_json(r) + '\n' for r in panel)
        if panel_path.exists() and panel_path.read_text(encoding='utf-8') != serialized:
            raise ValueError('saved final panel differs from frozen source')
        if not panel_path.exists():
            panel_path.write_text(serialized, encoding='utf-8')
        import torch
        from peft import PeftModel
        from transformers import GenerationConfig
        from src.models.inference import load_qwen, generate_completions, example_seed
        from src.models.snapshot import resolve_snapshot
        from src.evaluation.runner import assemble_prediction
        from src.training.dapo_mechanisms import freeze_generation_defaults
        snapshot = resolve_snapshot(plan['model_config'])
        size = plan['generation']['batch_size']
        expected_ids = {r['id'] for r in panel}
        for index, (name, policy) in enumerate(plan['models'].items(), 1):
            directory = output / name
            directory.mkdir(exist_ok=True)
            state.update(current_model=name, model_index=index)
            verify_policy(root, policy)
            signature = hashlib.sha256(canonical_json({'plan': plan['signature'], 'policy': policy, 'name': name}).encode()).hexdigest()
            identity = {'signature': signature, 'plan_signature': plan['signature'], 'policy': policy,
                        'panel_identity': panel_identity, 'generation': plan['generation']}
            identity_path = directory / 'evaluation_identity.json'
            if identity_path.exists() and read(identity_path) != identity:
                raise ValueError('existing evaluation identity differs from frozen model or panel')
            if not identity_path.exists():
                write_json(identity_path, identity)
            rows = read_predictions(directory / 'evaluation_predictions.jsonl', repair_partial_last_line=True)
            if any(r['evaluation_signature'] != signature for r in rows):
                raise ValueError('mixed policy identities in prediction file')
            if len({r['id'] for r in rows}) != len(rows) or {r['id'] for r in rows} - expected_ids:
                raise ValueError('duplicate or foreign evaluation questions')
            state['model_completed_questions'] = len(rows)
            write_json(output / 'study_status.json', state)
            if len(rows) == len(panel):
                write_json(directory / 'evaluation_summary.json', summary(rows))
                export_csv(directory / 'evaluation_predictions.csv', rows)
                if not (directory / 'resource_metrics.json').exists():
                    attempts = read_predictions(directory / 'generation_attempts.jsonl')
                    write_json(directory / 'resource_metrics.json', {
                        'total_returned_attempt_tokens_including_recomputed_batches': sum(a['generated_tokens'] for a in attempts),
                        'total_returned_attempt_seconds_including_recomputed_batches': sum(a['seconds'] for a in attempts),
                        'recorded_unique_prediction_tokens': sum(r['generated_token_count'] for r in rows),
                        'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in rows),
                        'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in rows),
                        'timing_scope': 'reconstructed_from_completed_saved_rows_no_new_inference',
                        'inflight_interrupted_generation_cost': 'unknown if interrupted before a returned batch',
                        'paid_service_started': False})
                write_json(directory / 'run_status.json', {'status': 'completed', 'completed_prediction_count': len(rows), 'plan_signature': plan['signature']})
                state['models_completed'] += 1
                continue
            if time.perf_counter() - started > max_session_minutes * 60:
                raise TimeoutError('本轮运行时间达到上限；重新执行同一命令可继续')
            model_started = time.perf_counter()
            model, tokenizer, load_metrics = load_qwen(snapshot, plan['model_config'])
            if policy['adapter']:
                model = PeftModel.from_pretrained(model, str(root / policy['adapter']), is_trainable=False, local_files_only=True).eval()
            model.config.use_cache = True
            eos = deepcopy(model.generation_config.eos_token_id)
            frozen = GenerationConfig(do_sample=True, temperature=.7, top_p=1., top_k=0, min_p=0.,
                                      repetition_penalty=1., no_repeat_ngram_size=0, num_beams=1, num_return_sequences=1,
                                      max_new_tokens=256, max_time=60, use_cache=True,
                                      eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
                                      bos_token_id=tokenizer.bos_token_id)
            audit = freeze_generation_defaults(model, frozen)
            write_json(directory / 'effective_generation_audit.json', audit)
            write_json(directory / 'model_load.json', load_metrics)
            completed = {r['id'] for r in rows}
            write_json(directory / 'run_status.json', {'status': 'started', 'completed_prediction_count': len(rows), 'plan_signature': plan['signature']})
            for start in range(0, len(panel), size):
                batch = panel[start:start+size]
                if all(r['id'] in completed for r in batch):
                    continue
                if time.perf_counter() - started > max_session_minutes * 60:
                    raise TimeoutError('本轮运行时间达到上限；重新执行同一命令可继续')
                seed = example_seed(plan['seed'], canonical_json([r['id'] for r in batch]))
                generated = generate_completions(model, tokenizer, batch, plan['generation'], enable_thinking=False, seed=seed)
                append_prediction(directory / 'generation_attempts.jsonl', {
                    'ids': [r['id'] for r in batch], 'seed': seed, 'generated_tokens': sum(g['generated_token_count'] for g in generated),
                    'seconds': sum(g['generation_seconds'] for g in generated),
                    'recomputed_ids': [r['id'] for r in batch if r['id'] in completed], 'recorded_at_utc': utc_now()})
                for record, generation in zip(batch, generated, strict=True):
                    if record['id'] in completed:
                        continue
                    row = assemble_prediction(record, generation, mode='non-thinking', model_name=snapshot.repository,
                                              revision=snapshot.revision, signature=signature)
                    row.update(difficulty=record['difficulty'], split=record['split'], checkpoint=name,
                               holdout_scope=plan['gsm8k_scope'], plan_signature=plan['signature'])
                    append_prediction(directory / 'evaluation_predictions.jsonl', row)
                    rows.append(row)
                    completed.add(record['id'])
                state['model_completed_questions'] = len(rows)
                write_json(output / 'study_status.json', state)
            write_json(directory / 'evaluation_summary.json', summary(rows))
            export_csv(directory / 'evaluation_predictions.csv', rows)
            attempts = read_predictions(directory / 'generation_attempts.jsonl')
            write_json(directory / 'resource_metrics.json', {
                'current_session_model_seconds': time.perf_counter() - model_started,
                'total_returned_attempt_tokens_including_recomputed_batches': sum(a['generated_tokens'] for a in attempts),
                'total_returned_attempt_seconds_including_recomputed_batches': sum(a['seconds'] for a in attempts),
                'recorded_unique_prediction_tokens': sum(r['generated_token_count'] for r in rows),
                'peak_allocated_bytes': max([load_metrics['load_peak_allocated_bytes']] + [r['peak_allocated_bytes'] for r in rows]),
                'peak_reserved_bytes': max([load_metrics['load_peak_reserved_bytes']] + [r['peak_reserved_bytes'] for r in rows]),
                'inflight_interrupted_generation_cost': 'unknown if interrupted before a returned batch', 'paid_service_started': False})
            write_json(directory / 'run_status.json', {'status': 'completed', 'completed_prediction_count': len(rows), 'plan_signature': plan['signature']})
            state['models_completed'] += 1
            write_json(output / 'study_status.json', state)
            del model, tokenizer
            model = tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
        state['status'] = 'completed'
    except KeyboardInterrupt:
        state.update(status='interrupted', error='用户中断；已有逐题结果保留，可用同一命令继续')
        code = 130
    except Exception as error:
        from src.evaluation.runner import _safe_error
        state.update(status='budget_stopped' if isinstance(error, TimeoutError) else 'failed',
                     error_type=type(error).__name__, error=_safe_error(error))
        code = 2 if isinstance(error, TimeoutError) else 1
    finally:
        state.update(finished_at_utc=utc_now(), current_session_seconds=time.perf_counter()-started)
        write_json(output / 'study_status.json', state)
    return code


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='开始或继续本地GPU评测')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--status', action='store_true', help='只查看状态')
    parser.add_argument('--max-session-minutes', type=float, default=120, help='本轮最长运行分钟数，可用同一命令续跑')
    args = parser.parse_args()
    if args.max_session_minutes <= 0:
        parser.error('运行时限必须大于0')
    output = PROJECT_ROOT / 'outputs/ablation_evaluation_round1'
    if args.status:
        path = output / 'study_status.json'
        if not path.exists():
            print('统一评测尚未运行。')
        else:
            s = read(path)
            print(f"状态：{s['status']}；完成{s['models_completed']}/{s['models_total']}个检查点；"
                  f"当前{s['current_model']}，{s['model_completed_questions']}/{s['questions_per_model']}题。")
        return 0
    if args.worker:
        return worker(args.max_session_minutes)
    plan = prepare_plan(PROJECT_ROOT)
    print(f"已冻结{len(plan['models'])}个模型检查点，同一批{len(plan['ordered_question_ids'])}题。", flush=True)
    if not args.execute:
        print('未启动GPU评测，未读取最终题目正文。执行时加 --execute。', flush=True)
        return 0
    log = output / f"console_{int(time.time())}.log"
    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
               '--max-session-minutes', str(args.max_session_minutes)]
    last_key = None
    last_print = time.monotonic()
    print('开始评测；完整日志自动落盘，可中断后使用同一命令继续。', flush=True)
    with log.open('wb') as handle:
        child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1'})
        try:
            while child.poll() is None:
                try:
                    s = read(output / 'study_status.json')
                    key = (s['current_model'], s['model_completed_questions'] // 40, s['models_completed'])
                    if key != last_key or time.monotonic() - last_print >= 30:
                        print(f"  检查点{s.get('model_index', 0)}/{s['models_total']}：{s['current_model'] or '准备题目'}，"
                              f"已完成{s['model_completed_questions']}/{s['questions_per_model']}题。", flush=True)
                        last_key, last_print = key, time.monotonic()
                except (OSError, json.JSONDecodeError):
                    pass
                time.sleep(1)
        except KeyboardInterrupt:
            print('\n正在中断，已保存的回答保留。', flush=True)
            try:
                child.wait(timeout=15)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                child.terminate()
        code = child.wait()
    status_path = output / 'study_status.json'
    s = read(status_path) if status_path.exists() else {'status': 'failed'}
    if s['status'] == 'completed' and code == 0:
        print('统一评测完成。请告诉我“评测完成”，接着做统计分析和图表。', flush=True)
    else:
        print(f"评测状态：{s['status']}。{s.get('error', '')}", flush=True)
        print(f'详细日志：{log}', flush=True)
        if s['status'] in ['interrupted', 'budget_stopped']:
            print('下次执行同一命令即可继续，已保存的批次会自动跳过。', flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
