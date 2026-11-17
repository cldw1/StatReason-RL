"""从已完成实验的倒数检查点恢复，验证真实Qwen的续训等价性。

创建独立验证目录；原始训练结果和检查点保持独立。恢复只从rollout周期边界开始。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from _bootstrap import PROJECT_ROOT
from src.data.storage import read_jsonl
from src.data.schema import canonical_json
from src.utils.artifacts import create_run_dir,file_sha256,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--checkpoint-step',type=int,required=True)
    parser.add_argument('--report-dir',type=Path,default=PROJECT_ROOT/'reports/grpo')
    args=parser.parse_args()
    source=args.run.resolve()
    status=json.loads((source/'run_status.json').read_text(encoding='utf-8'))
    config=json.loads((source/'resolved_config.yaml').read_text(encoding='utf-8'))
    step=args.checkpoint_step
    if status['status']!='completed' or step%config['trainer']['num_iterations'] or not 0<step<status['optimizer_steps']:
        raise ValueError('只从已完成运行的中间rollout周期边界验证')
    checkpoint=source/'checkpoints'/f'checkpoint-{step}'
    if not checkpoint.is_dir(): raise ValueError('指定检查点已被轮换或不存在')
    identity=json.loads((source/'training_identity.json').read_text(encoding='utf-8'))
    for name,expected in identity['code_hashes'].items():
        if file_sha256(PROJECT_ROOT/'src/training'/name)!=expected: raise ValueError('源代码已改变，不能声称精确续训')
    branch=create_run_dir(PROJECT_ROOT/'outputs/runs','grpo_resume_verification')
    if not branch.resolve().is_relative_to(PROJECT_ROOT.resolve()): raise ValueError('验证目录越界')
    for name in ['resolved_config.yaml','environment.json','data_selection.json','model_snapshot.json','training_identity.json']:
        shutil.copyfile(source/name,branch/name)
    destination=branch/'checkpoints'/checkpoint.name
    destination.parent.mkdir()
    shutil.copytree(checkpoint,destination)
    write_json(branch/'run_status.json',{'status':'resume_verification_pending','parent_run':source.name,'checkpoint_step':step})
    for name,predicate in [
        ('rollout_groups.jsonl',lambda r:r['optimizer_step']<step),
        ('policy_audit.jsonl',lambda r:r['optimizer_step']<step),
        ('train_metrics.jsonl',lambda r:r['step']<=step),
    ]:
        rows=[r for r in read_jsonl(source/name) if predicate(r)]
        (branch/name).write_text(''.join(canonical_json(r)+'\n' for r in rows),encoding='utf-8')
    if (source/'sampling_audit.jsonl').exists():
        sampling=[r for r in read_jsonl(source/'sampling_audit.jsonl') if r['optimizer_step_before_batch']<step]
        (branch/'sampling_audit.jsonl').write_text(''.join(canonical_json(r)+'\n' for r in sampling),encoding='utf-8')
    print(json.dumps({'source_run':str(source),'resume_branch':str(branch),'resume_step':step,'target_step':status['optimizer_steps']},ensure_ascii=False),flush=True)
    completed=subprocess.run([sys.executable,str(PROJECT_ROOT/'scripts/07_train_grpo.py'),'--config',str(args.config.resolve()),'--resume',str(branch)],cwd=PROJECT_ROOT)
    if completed.returncode:
        write_json(branch/'resume_equivalence.json',{'verified':False,'reason':'resumed_training_failed','exit_code':completed.returncode})
        return completed.returncode
    import torch
    from safetensors.torch import load_file
    expected=load_file(str(source/'adapter/adapter_model.safetensors'))
    actual=load_file(str(branch/'adapter/adapter_model.safetensors'))
    if set(expected)!=set(actual): raise RuntimeError('adapter参数键不同')
    differences={name:float((expected[name].float()-actual[name].float()).abs().max()) for name in expected}
    exact=all(torch.equal(expected[name],actual[name]) for name in expected)
    parent_rollouts=[r for r in read_jsonl(source/'rollout_groups.jsonl') if r['optimizer_step']>=step]
    resumed_rollouts=[r for r in read_jsonl(branch/'rollout_groups.jsonl') if r['optimizer_step']>=step]
    def generation_signature(rows):
        return [{'step':r['optimizer_step'],'groups':r['groups'],
                 'completions':[{'id':c['id'],'ids':c['completion_ids'],'reward':c['reward'],
                                 'old_logps':c['old_per_token_logps'],'ref_logps':c['ref_per_token_logps']} for c in r['completions']]} for r in rows]
    generation_equal=generation_signature(parent_rollouts)==generation_signature(resumed_rollouts)
    sampling_equal=True
    if (source/'sampling_audit.jsonl').exists():
        parent_sampling=[r for r in read_jsonl(source/'sampling_audit.jsonl') if r['optimizer_step_before_batch']>=step]
        resumed_sampling=[r for r in read_jsonl(branch/'sampling_audit.jsonl') if r['optimizer_step_before_batch']>=step]
        sampling_equal=parent_sampling==resumed_sampling
    result={'verified':exact and generation_equal and sampling_equal,'parent_run':source.name,'resumed_run':branch.name,
            'checkpoint_step':step,'target_step':status['optimizer_steps'],
            'adapter_tensors_compared':len(expected),'adapter_weights_bitwise_equal':exact,
            'online_sampling_decisions_equal':sampling_equal,
            'rollout_tokens_rewards_advantages_and_logps_equal':generation_equal,
            'max_abs_adapter_difference':max(differences.values(),default=0.),
            'scope':'same_machine_same_pinned_dependencies_at_rollout_boundary_not_cross_hardware_equivalence',
            'additional_optimizer_steps':status['optimizer_steps']-step,
            'additional_generated_tokens':sum(r['generated_tokens'] for r in resumed_rollouts)}
    resource=json.loads((branch/'resource_metrics.json').read_text(encoding='utf-8'))
    resource.update(session_generated_tokens=result['additional_generated_tokens'],
                    inherited_generated_tokens=resource['total_generated_tokens']-result['additional_generated_tokens'],
                    session_effective_training_tokens=sum(p['valid_tokens'] for p in read_jsonl(branch/'policy_audit.jsonl') if p['optimizer_step']>=step),
                    scope='current_session_time_but_total_generated_tokens_include_inherited_lineage_use_session_generated_tokens_for_additional_cost')
    write_json(branch/'resource_metrics.json',resource)
    write_json(branch/'resume_equivalence.json',result)
    branch_status=json.loads((branch/'run_status.json').read_text(encoding='utf-8'))
    branch_status.update(purpose='resume_equivalence_validation',exclude_from_effectiveness_comparison=True)
    write_json(branch/'run_status.json',branch_status)
    output=args.report_dir.resolve()
    if not output.is_relative_to(PROJECT_ROOT): raise ValueError('report directory must remain inside project')
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'resume_verification.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0 if result['verified'] else 1


if __name__=='__main__':
    raise SystemExit(main())
