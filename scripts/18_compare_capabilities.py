"""固定40道新开发题，真实运行候选模型与生成预算；可重复执行继续未完成实验。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from _bootstrap import PROJECT_ROOT
from src.data.schema import canonical_json
from src.evaluation.answer_diagnostics import diagnose_answer
from src.evaluation.capability_round import MODEL_NAMES, experiment_arms, messages_for, prepare_panel, summarize_arm, paired_recipe_comparison
from src.evaluation.runner import assemble_prediction, _safe_error
from src.evaluation.storage import append_prediction, read_predictions
from src.models.inference import example_seed, generate_completions, load_qwen
from src.models.snapshot import ModelNotCachedError, resolve_snapshot, snapshot_metadata
from src.utils.artifacts import utc_now, write_json, file_sha256
from src.utils.environment import package_versions


def record_availability(root: Path, model_key: str, event: dict):
    append_prediction(root / 'availability_events.jsonl', {'model': model_key, **event})
    write_json(root / f'{model_key}_availability.json', event)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download-models', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--download-only', action='store_true', help='仅下载并固定候选权重，不加载GPU或生成')
    parser.add_argument('--official-math-greedy', action='store_true', help='额外运行Math官方默认greedy、boxed提示、512/1024 Token诊断臂')
    parser.add_argument('--models', nargs='+', choices=list(MODEL_NAMES), default=list(MODEL_NAMES))
    parser.add_argument('--arms', nargs='+', help='可选：只运行指定实验名')
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'outputs/capability_round1')
    parser.add_argument('--max-arm-seconds', type=float, default=2400)
    args = parser.parse_args()
    root = args.root.resolve()
    if PROJECT_ROOT.resolve() not in root.parents:
        raise ValueError('实验目录必须位于当前工程内')
    panel, selection = prepare_panel(PROJECT_ROOT, root)
    arms = experiment_arms()
    if args.official_math_greedy:
        for budget in (512, 1024):
            arms.append({'name': f'qwen25_math_15b_native_greedy_{budget}', 'model': 'qwen25_math_15b',
                         'thinking': False, 'max_new_tokens': budget, 'prompt': 'official_boxed',
                         'do_sample': False, 'temperature': 1., 'top_p': 1., 'top_k': 50, 'min_p': 0.})
    plan_path = root / 'planned_arms.json'
    previous_plan = json.loads(plan_path.read_text(encoding='utf-8')) if plan_path.exists() else []
    planned = {arm['name']: arm for arm in previous_plan}
    for arm in arms:
        if arm['name'] in planned and planned[arm['name']] != arm:
            raise ValueError('已登记的实验臂不能静默修改')
        planned[arm['name']] = arm
    write_json(plan_path, list(planned.values()))
    arms = list(planned.values())
    if args.arms and any(name not in planned for name in args.arms):
        raise ValueError('未知实验臂，请检查 --arms 名称或先登记可选臂')
    if args.prepare_only:
        print(json.dumps(selection, ensure_ascii=False, indent=2))
        return 0
    import torch
    versions = package_versions()
    blocked = []
    for model_key in args.models:
        selected_arms = [a for a in arms if a['model'] == model_key and (not args.arms or a['name'] in args.arms)]
        if not selected_arms:
            continue
        model_config = {'name_or_path': MODEL_NAMES[model_key], 'revision': None,
                        'hf_home': str(PROJECT_ROOT / 'outputs/model_cache'), 'dtype': 'bfloat16',
                        'attn_implementation': 'sdpa', 'trust_remote_code': False}
        # Existing 0.6B cache is read-only; new downloads stay within the project.
        if model_key == 'qwen3_06b':
            model_config.update(hf_home='D:/AIModels/HuggingFace', revision='c1899de289a04d12100db370d81485cdf75e47ca')
        model = tokenizer = None
        try:
            snapshot = resolve_snapshot(model_config, allow_download=args.download_models, max_download_gib=4.0)
            snapshot_info = snapshot_metadata(snapshot)
            snapshot_path = root / f'{model_key}_snapshot.json'
            if snapshot_path.exists() and json.loads(snapshot_path.read_text(encoding='utf-8')) != snapshot_info:
                raise ValueError('模型快照内容改变，禁止覆盖已有实验的模型身份')
            write_json(snapshot_path, snapshot_info)
            if args.download_only:
                record_availability(root, model_key, {'status': 'snapshot_prepared', 'at': utc_now(), 'model_loaded': False})
                continue
            model, tokenizer, loaded = load_qwen(snapshot, model_config)
            write_json(root / f'{model_key}_load.json', loaded)
            record_availability(root, model_key, {'status': 'available', 'at': utc_now(), 'model_loaded': True})
            for arm in selected_arms:
                directory = root / arm['name']
                directory.mkdir(parents=True, exist_ok=True)
                generation = {**{k: arm[k] for k in ('max_new_tokens', 'do_sample', 'temperature', 'top_p', 'top_k', 'min_p')},
                              'max_prompt_tokens': 1536, 'max_time_per_completion_seconds': 240}
                identity = {'arm': arm, 'generation': generation, 'panel': selection['panel_fingerprint'],
                            'model': snapshot.repository, 'revision': snapshot.revision, 'batch_size': 4,
                            'prompt_sha256': hashlib.sha256(messages_for(panel[0], arm)[0]['content'].encode()).hexdigest(),
                            'template_sha256': hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
                            'code_sha256': {p: file_sha256(PROJECT_ROOT / p) for p in
                                           ('src/evaluation/answer_diagnostics.py', 'src/models/inference.py', 'src/evaluation/capability_round.py')},
                            'versions': versions, 'seed': 42,
                            'strict_policy': 'numeric.verify_completion_on_final_response_after_thinking_close',
                            'numeric_policy': 'unique_terminal_marker_or_boxed_same_tolerance_diagnostic_only'}
                signature = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
                identity_path = directory / 'identity.json'
                if identity_path.exists():
                    if json.loads(identity_path.read_text(encoding='utf-8'))['signature'] != signature:
                        raise ValueError('恢复协议发生变化，必须另建实验目录')
                else:
                    write_json(identity_path, {'signature': signature, **identity})
                rows = read_predictions(directory / 'evaluation_predictions.jsonl', repair_partial_last_line=True)
                if len({r['id'] for r in rows}) != len(rows) or any(r['evaluation_signature'] != signature for r in rows):
                    raise ValueError('恢复预测重复或身份不一致')
                completed = {r['id'] for r in rows}
                if not completed.issubset({r['id'] for r in panel}):
                    raise ValueError('恢复预测不属于冻结题组')
                started = time.perf_counter()
                status = {'status': 'running', 'started_at_utc': utc_now(), 'count': len(rows), 'expected': len(panel)}
                write_json(directory / 'status.json', status)
                try:
                    for start in range(0, len(panel), 4):
                        batch = panel[start:start + 4]
                        if all(r['id'] in completed for r in batch):
                            continue
                        if time.perf_counter() - started > args.max_arm_seconds:
                            raise TimeoutError('达到本次单实验软时限；重跑同命令继续')
                        seed = example_seed(42, canonical_json([r['id'] for r in batch]))
                        generated = generate_completions(model, tokenizer, batch, generation,
                            enable_thinking=arm['thinking'], seed=seed,
                            messages_override=[messages_for(r, arm) for r in batch])
                        append_prediction(directory / 'attempts.jsonl', {'ids': [r['id'] for r in batch],
                            'recomputed_ids': [r['id'] for r in batch if r['id'] in completed],
                            'tokens': sum(g['generated_token_count'] for g in generated),
                            'seconds': sum(g['generation_seconds'] for g in generated), 'at': utc_now()})
                        for record, output in zip(batch, generated, strict=True):
                            if record['id'] in completed:
                                continue
                            diagnostics = diagnose_answer(output['completion'], record, truncated=output['truncated'], thinking=arm['thinking'])
                            row = assemble_prediction(record, output, mode='thinking' if arm['thinking'] else 'non-thinking',
                                model_name=snapshot.repository, revision=snapshot.revision, signature=signature)
                            row['legacy_full_completion_verification'] = row['verification']
                            row.update(verification=diagnostics['strict'], answer_diagnostics=diagnostics, arm=arm['name'])
                            append_prediction(directory / 'evaluation_predictions.jsonl', row)
                            rows.append(row)
                            completed.add(row['id'])
                        summary = summarize_arm(rows)
                        write_json(directory / 'summary.json', summary)
                        write_json(directory / 'status.json', {**status, 'count': len(rows), 'last_saved_at_utc': utc_now()})
                        print(json.dumps({'arm': arm['name'], 'saved': len(rows), 'strict_correct': summary['strict_correct'],
                                          'truncated': summary['truncated'], 'tokens': summary['total_generated_tokens']}, ensure_ascii=False), flush=True)
                    status['status'] = 'completed'
                except KeyboardInterrupt:
                    status.update(status='interrupted')
                    raise
                except Exception as error:
                    status.update(status='incomplete' if isinstance(error, TimeoutError) else 'failed', error=_safe_error(error))
                finally:
                    summary = summarize_arm(rows)
                    attempts = read_predictions(directory / 'attempts.jsonl', repair_partial_last_line=True)
                    write_json(directory / 'summary.json', summary)
                    write_json(directory / 'status.json', {**status, 'count': len(rows), 'finished_at_utc': utc_now()})
                    write_json(directory / 'resources.json', {'current_session_seconds': time.perf_counter() - started,
                        'all_attempt_generated_tokens': sum(a['tokens'] for a in attempts),
                        'all_attempt_generation_seconds': sum(a['seconds'] for a in attempts),
                        'inflight_interrupted_cost': 'unobservable_before_generate_returns', 'paid_service_started': False})
        except Exception as error:
            status = {'status': 'blocked' if isinstance(error, ModelNotCachedError) or 'Proxy' in type(error).__name__ or 'Connection' in type(error).__name__ else 'failed',
                      'error_type': type(error).__name__, 'error': _safe_error(error), 'at': utc_now()}
            record_availability(root, model_key, status)
            blocked.append(model_key)
            print(json.dumps({'model': model_key, **status}, ensure_ascii=False), flush=True)
        finally:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    summaries, predictions = {}, {}
    for arm in arms:
        directory = root / arm['name']
        if (directory / 'summary.json').exists():
            summaries[arm['name']] = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
            if json.loads((directory / 'status.json').read_text(encoding='utf-8'))['status'] == 'completed':
                predictions[arm['name']] = read_predictions(directory / 'evaluation_predictions.jsonl')
    comparisons = {}
    for first, second in [('qwen3_06b_nonthink_256', 'qwen3_06b_nonthink_512'),
                          ('qwen3_06b_think_1024', 'qwen3_06b_think_2048'),
                          ('qwen3_06b_nonthink_512', 'qwen3_17b_nonthink_512'),
                          ('qwen3_06b_nonthink_512', 'qwen25_math_15b_common_512')]:
        if first in predictions and second in predictions:
            comparisons[f'{second}_vs_{first}'] = paired_recipe_comparison(predictions[first], predictions[second])
    write_json(root / 'round_summary.json', {'arms': summaries, 'comparisons': comparisons, 'blocked_models_this_session': blocked,
                                            'all_models_compared': all(a['name'] in predictions for a in arms), 'test_accessed': False})
    requested = [a['name'] for a in arms if a['model'] in args.models and (not args.arms or a['name'] in args.arms)]
    incomplete = [name for name in requested if name not in predictions]
    return 2 if blocked else (1 if incomplete and not args.download_only else 0)


if __name__ == '__main__':
    raise SystemExit(main())
