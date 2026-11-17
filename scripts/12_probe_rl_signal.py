"""在 RL 训练题上真实采样多回答，诊断组内信号；不执行参数更新。"""
import argparse
import json
import time

from _bootstrap import PROJECT_ROOT
from src.data.schema import canonical_json
from src.evaluation.storage import append_prediction
from src.models.inference import example_seed, generate_completions, load_qwen
from src.models.lora import validate_adapter
from src.models.snapshot import resolve_snapshot, snapshot_metadata
from src.models.prompts import PROMPT_SHA256
from src.training.group_rewards import relative_advantages
from src.training.sft_data import balanced_indices
from src.data.storage import read_jsonl
from src.utils.artifacts import create_run_dir, file_sha256, write_json, utc_now
from src.utils.config import load_config
from src.utils.environment import collect_environment


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter',required=True,type=__import__('pathlib').Path)
    parser.add_argument('--config',default=PROJECT_ROOT/'configs/evaluation_diagnostic.yaml')
    parser.add_argument('--groups',type=int,default=16)
    parser.add_argument('--generations',type=int,default=4)
    parser.add_argument('--max-new-tokens',type=int,default=128)
    args=parser.parse_args()
    if not 1<=args.groups<=32 or not 2<=args.generations<=8: raise ValueError('本地诊断要求1..32组，每组2..8回答')
    config=load_config(args.config)
    if not 16<=args.max_new_tokens<=256: raise ValueError('本地探针生成长度要求16..256')
    config['evaluation']['max_new_tokens']=args.max_new_tokens
    config['probe']={'groups':args.groups,'generations_per_group':args.generations,'seed':42,
                     'maximum_generated_token_budget':args.groups*args.generations*args.max_new_tokens,
                     'selection_distribution':'equal_task_types_not_natural_RL_pool_mixture'}
    root=PROJECT_ROOT/config['paths']['dataset_release']
    pool=[]
    manifests={}
    for source in ('statistics_full','gsm8k'):
        manifest=json.loads((root/source/'manifest.json').read_text(encoding='utf-8'))
        path=root/source/'rl.jsonl'
        if file_sha256(path)!=manifest['file_sha256'][path.name]: raise ValueError('RL 数据哈希不符')
        manifests[source]=manifest['statistics']['fingerprint']
        records=read_jsonl(path)
        if any(r['split']!='rl' for r in records): raise ValueError('只接受 RL 训练题')
        pool.extend(records)
    panel=[pool[i] for i in balanced_indices(pool,args.groups,42)]
    snapshot=resolve_snapshot(config['model'])
    identity=validate_adapter(args.adapter.resolve(),snapshot.repository,snapshot.revision)
    run=create_run_dir(PROJECT_ROOT/'outputs/runs','rl_signal_probe')
    write_json(run/'resolved_config.yaml',config)
    write_json(run/'environment.json',collect_environment(PROJECT_ROOT))
    write_json(run/'selection.json',{'ids':[r['id'] for r in panel],'source_fingerprints':manifests,
                                     'selection':'balanced_task_types_before_any_reward_observation','official_test_accessed':False})
    write_json(run/'model_identity.json',{'snapshot':snapshot_metadata(snapshot),'adapter':identity,'prompt_sha256':PROMPT_SHA256})
    status={'status':'started','model_training_performed':False,'purpose':'rollout_reward_signal_diagnostic_not_grpo_training'}
    write_json(run/'run_status.json',status)
    groups=[]
    started=time.perf_counter()
    try:
        import torch
        from peft import PeftModel
        from src.rewards.numeric import verify_completion
        model,tokenizer,load_metrics=load_qwen(snapshot,config['model'])
        model=PeftModel.from_pretrained(model,str(args.adapter.resolve()),is_trainable=False,local_files_only=True).eval()
        for record in panel:
            if time.perf_counter()-started>600: raise TimeoutError('奖励诊断达到600秒时限')
            generated=generate_completions(model,tokenizer,[record]*args.generations,config['evaluation'],enable_thinking=False,
                                           seed=example_seed(42,'rl_probe:'+record['id']))
            verified=[verify_completion(g['completion'],record,truncated=g['truncated']).to_dict() for g in generated]
            rewards=[float(v['strict_correct']) for v in verified]
            group={'id':record['id'],'task_type':record['task_type'],'source':record['source'],'question':record['question'],
                   'answer':record['answer'],'rewards':rewards,**relative_advantages(rewards),
                   'completions':[{**g,'verification':v} for g,v in zip(generated,verified,strict=True)],'recorded_at_utc':utc_now()}
            append_prediction(run/'rollout_groups.jsonl',group)
            groups.append(group)
            print(json.dumps({'group':len(groups),'task':record['task_type'],'rewards':rewards,'zero_variance':group['zero_variance']}),flush=True)
        status['status']='completed'
    except BaseException as error:
        status.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        completions=[c for g in groups for c in g['completions']]
        summary={'completed_groups':len(groups),'planned_groups':len(panel),'generations_per_group':args.generations,
                 'mixed_reward_groups':sum(not g['zero_variance'] for g in groups),'all_wrong_groups':sum(not any(g['rewards']) for g in groups),
                 'all_correct_groups':sum(all(g['rewards']) for g in groups),
                 'raw_effective_group_rate':sum(not g['zero_variance'] for g in groups)/len(groups) if groups else None,
                 'total_generated_tokens':sum(c['generated_token_count'] for c in completions),
                 'strict_correct_completions':sum(c['verification']['strict_correct'] for c in completions),
                 'wall_seconds':time.perf_counter()-started,
                 'max_new_tokens':args.max_new_tokens,
                 'truncation_rate':sum(c['truncated'] for c in completions)/len(completions) if completions else None,
                 'peak_allocated_bytes':max((c['peak_allocated_bytes'] for c in completions),default=0),
                 'no_dynamic_resampling':True,'no_parameter_update':True,'not_validation_accuracy':True}
        summary['selection_distribution']='equal_task_types_not_an_estimate_of_whole_RL_pool_effective_rate'
        summary['by_task_type']={task:{'groups':len(subset),
            'mixed_reward_groups':sum(not g['zero_variance'] for g in subset),
            'strict_correct_completions':sum(sum(g['rewards']) for g in subset)}
            for task in sorted({g['task_type'] for g in groups})
            if (subset:=[g for g in groups if g['task_type']==task])}
        write_json(run/'probe_summary.json',summary)
        write_json(run/'resource_metrics.json',{'wall_seconds':summary['wall_seconds'],
                   'total_generated_tokens':summary['total_generated_tokens'],
                   'effective_training_tokens':0,'extra_resampling_tokens':0,
                   'peak_allocated_bytes':summary['peak_allocated_bytes'],
                   'peak_reserved_bytes':max((c['peak_reserved_bytes'] for c in completions),default=0),
                   'maximum_generated_token_budget':config['probe']['maximum_generated_token_budget'],
                   'paid_service_started':False,'scope':'reward_diagnostic_generation_only'})
        write_json(run/'run_status.json',{**status,'finished_at_utc':utc_now()})
        print(json.dumps({'run':str(run),'summary':summary},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__': main()
