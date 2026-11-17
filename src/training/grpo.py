"""真实单卡 Vanilla GRPO 入口；固定 SFT 初始化/参考，输出审计与检查点。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import time

from src.data.schema import canonical_json
from src.data.storage import read_jsonl
from src.models.inference import load_qwen
from src.models.lora import validate_adapter, capture_trainable, parameter_update_evidence, tensor_sha256, adapter_metadata
from src.models.prompts import PROMPT_SHA256
from src.models.snapshot import resolve_snapshot,snapshot_metadata
from src.utils.artifacts import create_run_dir,write_json,file_sha256,utc_now
from src.utils.config import load_config,validate_config
from src.utils.environment import collect_environment,package_versions
from .callbacks import LocalTrainingCallback
from .rl_data import prepare_rl_data
from .sft import _probe_logits


def validate_vanilla(config):
    validate_config(config)
    t=config['trainer']
    if t['loss_type']!='grpo' or t['epsilon']!=t['epsilon_high'] or any(config['mechanisms'][k]['enabled'] for k in ('dynamic_sampling','overlong')):
        raise ValueError('此入口只运行 Vanilla GRPO，机制消融使用独立入口')
    if t['steps_per_generation']!=t['gradient_accumulation_steps']:
        raise ValueError('当前单卡实现要求每轮rollout对应完整优化器累积批次')
    if t['max_steps']%t['num_iterations'] or t['save_steps']%t['num_iterations']:
        raise ValueError('训练终点和检查点必须落在rollout复用周期边界')
    if config['initialization']['reference_policy']!='frozen_copy_of_same_sft_adapter' or t['beta']<=0 or t['sync_ref_model']:
        raise ValueError('本对照明确要求非零KL及固定SFT参考')
    if t['shuffle_dataset'] is False and config['experiment']['level']>0:
        raise ValueError('正式实验必须随机采样完整RL池，不能沿用诊断题优先顺序')


def _reference_hashes(model):
    return {name:tensor_sha256(p) for name,p in model.named_parameters() if 'lora_' in name or name.endswith('base_layer.weight') and 'layers.0.' in name}


def resolve_rl_fingerprint(config,actual):
    """已有声明必须一致，禁止悄悄覆盖冻结实验的数据指纹。"""
    declared=config['data'].get('fingerprint')
    if declared is not None and declared!=actual:
        raise ValueError('RL数据指纹与冻结配置不同，必须新建实验配置')
    config['data']['fingerprint']=actual


def run_grpo_cli(project_root: Path):
    import torch
    from datasets import Dataset
    from peft import PeftModel,get_peft_model_state_dict
    from transformers import AutoTokenizer,set_seed
    from transformers.trainer_utils import get_last_checkpoint
    from trl import GRPOConfig,GRPOTrainer
    from .grpo_trainer import AuditedGRPOTrainer
    from .dapo_mechanisms import MechanismGRPOTrainer, validate_ablation, DynamicSamplingBudgetExhausted

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=project_root/'configs/grpo.yaml')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    config=load_config(args.config)
    ablation=config['experiment'].get('ablation_id')
    run=args.resume.resolve() if args.resume else create_run_dir(project_root/config['paths']['run_root'],config['experiment']['name'])
    if not run.resolve().is_relative_to(project_root.resolve()): raise ValueError('运行目录须在项目内')
    if args.resume and json.loads((run/'run_status.json').read_text())['status']=='completed':
        print('Run already completed:',run); return 0
    started=time.perf_counter()
    status={'run_id':run.name,'run_type':('controlled_ablation_'+ablation) if ablation else 'vanilla_grpo','level':config['experiment']['level'],'status':'started',
            'started_at_utc':utc_now(),'model_training_performed':False,'official_test_accessed':False,'optimizer_steps':0}
    trainer=None; model=None; reference=None; callback=None; training_seconds=None; exit_code=0
    try:
        snapshot=resolve_snapshot(config['model'])
        adapter_path=project_root/config['initialization']['sft_adapter']
        adapter_info=validate_adapter(adapter_path,snapshot.repository,snapshot.revision)
        if adapter_info['adapter_safetensors_sha256']!=config['initialization']['sft_adapter_sha256']:
            raise ValueError('SFT初始化权重哈希与共同配置不同')
        tokenizer=AutoTokenizer.from_pretrained(snapshot.path,local_files_only=True,padding_side='left')
        rows,records,selection=prepare_rl_data(tokenizer,project_root/config['paths']['dataset_release'],config['rl_data'],
                                               config['trainer']['max_prompt_length'],config['reproducibility']['seed'])
        resolve_rl_fingerprint(config,selection['record_fingerprint'])
        (validate_ablation if ablation else validate_vanilla)(config)
        api_path=Path(inspect.getfile(GRPOTrainer))
        code_hashes={name:file_sha256(project_root/'src/training'/name) for name in ['grpo.py','grpo_trainer.py','grpo_math.py','rl_data.py','group_rewards.py','dapo_mechanisms.py']}
        support_hashes={name:file_sha256(project_root/name) for name in ['src/rewards/numeric.py','src/models/inference.py',
                        'src/models/lora.py','src/models/prompts.py','src/utils/config.py','src/training/callbacks.py']}
        identity={'config':config,'selection':selection,'adapter':adapter_info,'prompt_sha256':PROMPT_SHA256,
                  'versions':package_versions(),'code_hashes':code_hashes,'support_source_sha256':support_hashes,
                  'trl_source_sha256':file_sha256(api_path)}
        signature=hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        checkpoint=None
        if args.resume:
            previous=json.loads((run/'training_identity.json').read_text(encoding='utf-8'))
            if previous['signature']!=signature: raise ValueError('恢复时配置、数据、代码、依赖或SFT已改变')
            checkpoint=get_last_checkpoint(str(run/'checkpoints'))
            if not checkpoint: raise ValueError('没有完整检查点')
            step=json.loads((Path(checkpoint)/'trainer_state.json').read_text())['global_step']
            if step%config['trainer']['num_iterations']: raise ValueError('不能从rollout周期中间恢复')
            if ablation and (run/'sampling_audit.jsonl').exists():
                if any(r['optimizer_step_before_batch']>=step for r in read_jsonl(run/'sampling_audit.jsonl')):
                    raise ValueError('uncommitted sampling follows checkpoint; preserve this run and start independently')
            status['resumed_from']=checkpoint
        else:
            write_json(run/'resolved_config.yaml',config)
            write_json(run/'environment.json',collect_environment(project_root))
            write_json(run/'data_selection.json',selection)
            write_json(run/'model_snapshot.json',snapshot_metadata(snapshot))
            write_json(run/'training_identity.json',{'signature':signature,**identity})
            code_snapshot=run/'code_snapshot'
            code_snapshot.mkdir(exist_ok=True)
            for name in code_hashes:
                shutil.copyfile(project_root/'src/training'/name,code_snapshot/name)
            for name in support_hashes:
                target=code_snapshot/'dependencies'/name
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(project_root/name,target)
        write_json(run/'run_status.json',status)
        print(json.dumps({'run':str(run),'selected_rl_examples':len(rows),'max_prompt_tokens':selection['max_selected_prompt_tokens'],
                          'selection_note':selection['selection_note']},ensure_ascii=False),flush=True)
        if args.prepare_only:
            status['status']='data_prepared_no_training'
        else:
            set_seed(config['reproducibility']['seed'])
            base,_,load_resources=load_qwen(snapshot,config['model'])
            model=PeftModel.from_pretrained(base,str(adapter_path),is_trainable=True,local_files_only=True)
            del base
            base_ref,_,ref_load_resources=load_qwen(snapshot,config['model'])
            reference=PeftModel.from_pretrained(base_ref,str(adapter_path),is_trainable=False,local_files_only=True).eval()
            del base_ref
            reference.requires_grad_(False)
            model.config.use_cache=False
            if config['trainer']['gradient_checkpointing']:
                model.enable_input_require_grads()
            initial=capture_trainable(model)
            ref_hashes=_reference_hashes(reference)
            base_hashes={name:tensor_sha256(p) for name,p in model.named_parameters() if name.endswith('base_layer.weight') and 'layers.0.' in name}
            if not initial or any('lora_' not in k for k in initial): raise RuntimeError('训练参数必须为LoRA')
            callback=LocalTrainingCallback(run,config['resources']['max_train_seconds'],stop_at_step_multiple=config['trainer']['num_iterations'])
            trainer_values={**config['trainer'],'output_dir':str(run/'checkpoints'),
                            'seed':config['reproducibility']['seed'],'data_seed':config['reproducibility']['data_seed'],
                            'logging_nan_inf_filter':False,'save_safetensors':True,'push_to_hub':False,
                            'disable_tqdm':True,'torch_compile':False,'tf32':False,'per_device_eval_batch_size':1}
            trainer_class=MechanismGRPOTrainer if ablation else AuditedGRPOTrainer
            extra={'experiment_config':config} if ablation else {}
            trainer=trainer_class(run=run,records=records,frozen_reference=reference,model=model,**extra,
                                      args=GRPOConfig(**trainer_values),train_dataset=Dataset.from_list(rows),
                                      processing_class=tokenizer,callbacks=[callback])
            write_json(run/'trl_resolved_arguments.json',trainer.args.to_dict())
            write_json(run/'generation_config.json',trainer.generation_config.to_dict())
            if ablation:
                write_json(run/'effective_generation_audit.json',trainer.generation_protocol_audit)
            write_json(run/'checkpointing_compatibility.json',trainer.audit_checkpointing_configuration)
            write_json(run/'reference_identity.json',{'kind':'independent_frozen_sft_copy','adapter':adapter_info,
                       'all_reference_parameters_frozen':all(not p.requires_grad for p in trainer.ref_model.parameters()),
                       'reference_tensor_hashes_before':ref_hashes,'reference_shared_parameters_with_policy':
                       sum(p.data_ptr()==q.data_ptr() for p,q in zip(model.parameters(),reference.parameters(),strict=True))})
            torch.cuda.reset_peak_memory_stats(0)
            t0=time.perf_counter()
            status['training_attempted']=True
            write_json(run/'run_status.json',status)
            sampling_stopped=False
            try:
                result=trainer.train(resume_from_checkpoint=checkpoint)
            except DynamicSamplingBudgetExhausted as error:
                sampling_stopped=True
                status['sampling_stop_reason']=str(error)
                if trainer.state.global_step==0:
                    raise
                from types import SimpleNamespace
                result=SimpleNamespace(metrics={'sampling_budget_stopped':True,'actual_optimizer_steps':trainer.state.global_step})
            torch.cuda.synchronize()
            training_seconds=time.perf_counter()-t0
            write_json(run/'train_result.json',result.metrics)
            status['optimizer_steps']=trainer.state.global_step
            status['model_training_performed']=trainer.state.global_step>0
            evidence=parameter_update_evidence(model,initial)
            current_base={name:tensor_sha256(p) for name,p in model.named_parameters() if name in base_hashes}
            if current_base!=base_hashes: raise RuntimeError('冻结Base抽查权重变化')
            if _reference_hashes(trainer.ref_model)!=ref_hashes: raise RuntimeError('参考SFT权重变化')
            evidence.update(sampled_frozen_base_tensor_hashes_unchanged=True,reference_adapter_and_sampled_base_hashes_unchanged=True)
            write_json(run/'parameter_updates.json',evidence)
            frozen_reference=trainer.ref_model
            write_json(run/'reference_verification.json',{'all_reference_parameters_frozen':True,'all_reference_gradients_none':all(p.grad is None for p in frozen_reference.parameters()),
                       'reference_hashes_unchanged':True,'initial_policy_reference_max_logprob_error':trainer.audit_first_reference_error})
            model.gradient_checkpointing_disable()
            model.eval()
            probe={'input_ids':tokenizer.encode(rows[0]['prompt'],add_special_tokens=False)}
            logits_before=_probe_logits(model,probe)
            state={name:t.detach().cpu().clone() for name,t in get_peft_model_state_dict(model).items()}
            destination=run/'adapter'
            model.save_pretrained(destination,safe_serialization=True,save_embedding_layers=False)
            tokenizer.save_pretrained(destination)
            meta=adapter_metadata(destination,snapshot.repository,snapshot.revision)
            meta.update(run_id=run.name,training_level=config['experiment']['level'],training_signature=signature,
                        training_steps=trainer.state.global_step,initialization_adapter_sha256=adapter_info['adapter_safetensors_sha256'],
                        training_algorithm=('controlled_ablation_'+ablation) if ablation else 'vanilla_grpo',reference_policy='frozen_same_sft',initialization_adapter=str(adapter_path))
            write_json(destination/'adapter_metadata.json',meta)
            trainer.save_state()
            status['training_audit']={'rollouts':trainer.audit_rollouts,'generated_tokens':trainer.audit_generated_tokens,
                  'effective_training_tokens_including_reuse':trainer.audit_effective_training_tokens,'microsteps':trainer.audit_microsteps,
                  'initial_reference_logprob_max_error':trainer.audit_first_reference_error,
                  'independent_loss_max_error':trainer.audit_objective_max_error}
            status['peak_allocated_bytes']=torch.cuda.max_memory_allocated(0)
            status['peak_reserved_bytes']=torch.cuda.max_memory_reserved(0)
            del trainer,model,reference,frozen_reference
            trainer=model=reference=None
            gc.collect(); torch.cuda.empty_cache()
            fresh,_,_=load_qwen(snapshot,config['model'])
            reloaded=PeftModel.from_pretrained(fresh,str(destination),is_trainable=False,local_files_only=True).eval()
            after=get_peft_model_state_dict(reloaded)
            if set(after)!=set(state): raise RuntimeError('重载参数键不同')
            for name,value in state.items(): torch.testing.assert_close(value,after[name].cpu(),rtol=0,atol=0)
            logits_after=_probe_logits(reloaded,probe)
            torch.testing.assert_close(logits_before,logits_after,rtol=0,atol=0)
            write_json(run/'reload_verification.json',{'adapter_weights_exact_match':True,'probe_logits_exact_match':True,'max_abs_logits_error':float((logits_before-logits_after).abs().max())})
            status.update(status='sampling_budget_stopped' if sampling_stopped else 'budget_stopped' if callback.budget_stopped else 'completed',adapter=str(destination),
                          changed_parameter_tensors=evidence['changed_tensor_count'],reload_verified=True)
    except DynamicSamplingBudgetExhausted as error:
        status.update(status='sampling_budget_stopped',sampling_stop_reason=str(error));exit_code=0
    except KeyboardInterrupt:
        status.update(status='interrupted',error='KeyboardInterrupt');exit_code=130
    except Exception as error:
        from src.evaluation.runner import _safe_error
        status.update(status='failed',error_type=type(error).__name__,error=_safe_error(error));exit_code=1
    finally:
        if trainer is not None:
            status.update(optimizer_steps=trainer.state.global_step,model_training_performed=trainer.state.global_step>0)
        status.update(finished_at_utc=utc_now(),session_wall_seconds=time.perf_counter()-started)
        write_json(run/'run_status.json',status)
        rollouts=read_jsonl(run/'rollout_groups.jsonl') if (run/'rollout_groups.jsonl').exists() else []
        sampling=read_jsonl(run/'sampling_audit.jsonl') if (run/'sampling_audit.jsonl').exists() else []
        write_json(run/'resource_metrics.json',{'session_wall_seconds':status['session_wall_seconds'],'training_seconds':training_seconds,
                   'total_generated_tokens':sum(r['generated_tokens'] for r in rollouts),
                   'extra_resampling_tokens':sum(r['generated_tokens'] for r in sampling if r['attempt_index_in_batch']>0),
                   'discarded_rollout_tokens':sum(r['generated_tokens'] for r in sampling if not r['accepted_for_training']),
                   'sampling_attempts':len(sampling),'accepted_groups':sum(r['accepted_for_training'] for r in sampling),
                   'generation_and_scoring_seconds':sum(r['generation_and_scoring_seconds'] for r in rollouts),
                   'effective_training_tokens_including_reuse':status.get('training_audit',{}).get('effective_training_tokens_including_reuse'),
                   'peak_allocated_bytes':status.get('peak_allocated_bytes'),'peak_reserved_bytes':status.get('peak_reserved_bytes'),
                   'paid_service_started':False,'scope':'single_card_grpo_current_session_training_time_all_returned_rollouts'})
        print(json.dumps(status,ensure_ascii=False,indent=2),flush=True)
        print('Run records:',run,flush=True)
    return exit_code
