"""12题开发配置探针：few-shot与思考预算；不是正式能力评测或参数训练。"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from _bootstrap import PROJECT_ROOT
from src.data.schema import canonical_json
from src.data.storage import read_jsonl
from src.evaluation.baseline_probe import load_training_demonstrations,choose_demonstrations,add_demonstrations,protocol_hash
from src.evaluation.panel import select_development_panel
from src.evaluation.runner import assemble_prediction
from src.evaluation.storage import append_prediction
from src.models.inference import load_qwen,generate_completions,example_seed
from src.models.lora import validate_adapter
from src.models.prompts import PROMPT_SHA256
from src.models.snapshot import resolve_snapshot
from src.utils.artifacts import create_run_dir,write_json,utc_now
from src.utils.config import load_config
from src.utils.environment import collect_environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    c=load_config(PROJECT_ROOT/'configs/evaluation_diagnostic.yaml')
    run=create_run_dir(PROJECT_ROOT/'outputs/evaluations','baseline_recipe_probe')
    panel,selection=select_development_panel(PROJECT_ROOT/c['paths']['dataset_release'],c['evaluation'],42)
    panel=panel[:12]  # 与旧100题前3批完全一致，可复用原始192Token基线回答。
    cached=read_jsonl(PROJECT_ROOT/'outputs/evaluations/sft_development_20261116T080656Z_fbd523ce/evaluation_predictions.jsonl')
    baseline=cached[:12]
    if [r['id'] for r in baseline]!=[r['id'] for r in panel]: raise ValueError('缓存面板顺序不符')
    training,hashes=load_training_demonstrations(PROJECT_ROOT/'data/processed/statreason-diagnostic-v2_1')
    demos={r['id']:choose_demonstrations(r,training) for r in panel}
    if {d['id'] for group in demos.values() for d in group}&set(selection['selected_ids']): raise ValueError('示例/开发面板交叉')
    initial=load_config(PROJECT_ROOT/'configs/grpo.yaml')['initialization']
    adapter=PROJECT_ROOT/initial['sft_adapter']
    snapshot=resolve_snapshot(c['model'])
    identity=validate_adapter(adapter,snapshot.repository,snapshot.revision)
    profiles={
        'two_shot_non_thinking_512':{'thinking':False,'max_new_tokens':512,'temperature':.7,'top_p':.8,'few_shot':True},
        'thinking_2048':{'thinking':True,'max_new_tokens':2048,'temperature':.6,'top_p':.95,'few_shot':False},
    }
    protocol={'scope':'development_recipe_probe_not_final_benchmark','panel_ids':[r['id'] for r in panel],
              'baseline_cache':'sft_development_20261116T080656Z_fbd523ce','profiles':profiles,'adapter':identity,
              'demonstration_release':'statreason-diagnostic-v2_1','sft_demonstration_file_hashes':hashes,
              'demonstration_ids':{k:[r['id'] for r in ds] for k,ds in demos.items()},
              'prompt_sha256':PROMPT_SHA256,'max_prompt_tokens':2048,'maximum_new_token_budget':12*(512+2048),
              'generation_wall_soft_budget_seconds':390,'notes':'recipe changes multiple factors; no causal single_factor_or_training_gain_claim'}
    write_json(run/'resolved_config.yaml',protocol)
    write_json(run/'environment.json',collect_environment(PROJECT_ROOT))
    for row in baseline: append_prediction(run/'baseline_cached_predictions.jsonl',row)
    status={'status':'prepared','official_test_accessed':False,'model_training_performed':False,'started_at_utc':utc_now()}
    write_json(run/'run_status.json',status)
    rows=[];started=time.perf_counter()
    try:
        if not args.prepare_only:
            import torch
            from peft import PeftModel
            model,tokenizer,load=load_qwen(snapshot,c['model'])
            model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True).eval()
            write_json(run/'model_load.json',load)
            for name,profile in profiles.items():
                settings={**c['evaluation'],**{k:profile[k] for k in ['max_new_tokens','temperature','top_p']},'max_prompt_tokens':2048}
                for start in range(0,len(panel),4):
                    if time.perf_counter()-started>390: raise TimeoutError('探针达到390秒软时限')
                    original=panel[start:start+4]
                    batch=[add_demonstrations(r,demos[r['id']]) if profile['few_shot'] else deepcopy(r) for r in original]
                    seed=example_seed(42,canonical_json([r['id'] for r in original]))
                    generated=generate_completions(model,tokenizer,batch,settings,enable_thinking=profile['thinking'],seed=seed)
                    for record,generation in zip(original,generated,strict=True):
                        row=assemble_prediction(record,generation,mode=name,model_name=snapshot.repository,
                                                revision=snapshot.revision,signature=protocol_hash(protocol))
                        row['demonstration_ids']=[d['id'] for d in demos[record['id']]] if profile['few_shot'] else []
                        append_prediction(run/'evaluation_predictions.jsonl',row);rows.append(row)
                    print(json.dumps({'profile':name,'completed':sum(r['mode']==name for r in rows),
                                      'strict_correct':sum(r['verification']['strict_correct'] for r in rows if r['mode']==name)},ensure_ascii=False),flush=True)
            status['status']='completed'
    except BaseException as error:
        status.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        def summarize(rs):
            return {'questions':len(rs),'strict_correct':sum(r['verification']['strict_correct'] for r in rs),
                    'format_valid':sum(r['verification']['format_valid'] for r in rs),'truncated':sum(r['truncated'] for r in rs),
                    'generated_tokens':sum(r['generated_token_count'] for r in rs),
                    'mean_amortized_generation_seconds':sum(r['generation_seconds'] for r in rs)/len(rs) if rs else None}
        summary={'baseline_cached_192':summarize(baseline),**{name:summarize([r for r in rows if r['mode']==name]) for name in profiles}}
        write_json(run/'evaluation_summary.json',summary)
        write_json(run/'resource_metrics.json',{'wall_seconds':time.perf_counter()-started,
                   'new_generated_tokens':sum(r['generated_token_count'] for r in rows),'cached_baseline_additional_cost':0,
                   'peak_allocated_bytes':max((r['peak_allocated_bytes'] for r in rows),default=0),'paid_service_started':False})
        status.update(finished_at_utc=utc_now());write_json(run/'run_status.json',status)
        print(json.dumps({'run':str(run),'status':status,'summary':summary},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
