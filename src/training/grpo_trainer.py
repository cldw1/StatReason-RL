"""在固定 TRL 0.23.1 上增加 SFT 参考策略与逐 Token 审计，保留其策略损失。"""
from __future__ import annotations

import time
import torch
from transformers import TopKLogitsWarper, TopPLogitsWarper
from trl import GRPOTrainer

from src.evaluation.storage import append_prediction
from src.data.storage import read_jsonl
from src.models.lora import tensor_sha256
from src.rewards.numeric import verify_completion
from .group_rewards import relative_advantages
from .grpo_math import distribution_entropy, grpo_objective, masked_mean


class AuditedGRPOTrainer(GRPOTrainer):
    """仅支持单卡文本 LoRA；参考模型必须另行加载冻结的同一 SFT。"""
    def __init__(self, *, run, records, frozen_reference, **kwargs):
        self.audit_run, self.audit_records=run,records
        prior=read_jsonl(run/'rollout_groups.jsonl') if (run/'rollout_groups.jsonl').exists() else []
        previous_policy=read_jsonl(run/'policy_audit.jsonl') if (run/'policy_audit.jsonl').exists() else []
        self.audit_rollouts=max((r['rollout_id'] for r in prior),default=0)
        self.audit_generated_tokens=sum(r['generated_tokens'] for r in prior)
        self.audit_rollout_seconds=sum(r['generation_and_scoring_seconds'] for r in prior)
        self.audit_old_rows={}
        self.audit_microsteps=len(previous_policy)
        self.audit_effective_training_tokens=sum(r['valid_tokens'] for r in previous_policy)
        self.audit_objective_max_error=0.
        self.audit_first_reference_error=None
        self.audit_reward_rows=[]
        self.audit_entropy=None
        super().__init__(reward_funcs=self.strict_numeric_reward,**kwargs)
        if self.accelerator.num_processes!=1 or self.use_vllm or self.use_transformers_paged:
            raise ValueError('当前审计训练器只验证单卡 Transformers 文本生成')
        if self.ref_model is not None or any(p.requires_grad for p in frozen_reference.parameters()):
            raise ValueError('需要单独、完全冻结的 SFT 参考模型')
        self.ref_model=self.accelerator.prepare_model(frozen_reference,evaluation_mode=True)
        self.ref_model.eval()
        self.restore_checkpointing_configuration()

    def restore_checkpointing_configuration(self):
        """TRL 0.23.1 的生成上下文会无参数重启 checkpoint，需恢复非重入配置。"""
        if self.args.gradient_checkpointing:
            model=self.accelerator.unwrap_model(self.model)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=self.args.gradient_checkpointing_kwargs)
            functions=[getattr(m,'_gradient_checkpointing_func') for m in model.modules() if hasattr(m,'_gradient_checkpointing_func')]
            expected=self.args.gradient_checkpointing_kwargs.get('use_reentrant',True)
            if not functions or any(getattr(f,'keywords',{}).get('use_reentrant',True)!=expected for f in functions):
                raise RuntimeError('未恢复配置中的梯度检查点语义')
            self.audit_checkpointing_configuration={'use_reentrant':expected,'modules_checked':len(functions)}

    def strict_numeric_reward(self, completions, completion_ids, record_id, **kwargs):
        rows=[]
        for text,ids,key in zip(completions,completion_ids,record_id,strict=True):
            record=self.audit_records[key]
            verification=verify_completion(text,record,truncated=self.eos_token_id not in ids).to_dict()
            rows.append({'id':key,'completion':text,'completion_ids':ids,'verification':verification,
                         'reward':float(verification['strict_correct'])})
        self.audit_reward_rows=rows
        return [r['reward'] for r in rows]

    def _get_per_token_logps_and_entropies(self,model,input_ids,attention_mask,logits_to_keep,batch_size=None,compute_entropy=False,**kwargs):
        """同 TRL 的温度缩放密集 logprob；额外熵在 no_grad 下计算。

        top-k/top-p 只作用于采样熵，不能声称密集策略概率就是过滤采样概率。
        FP32 log-softmax 避免直接在 BF16 上进行概率比率计算。
        """
        if any(v is not None for v in kwargs.values()): raise ValueError('尚未支持多模态输入')
        if model is self.model:
            self.restore_checkpointing_configuration()
        all_logps,all_entropy,raw_entropy,sampling_entropy=[],[],[],[]
        chunk=batch_size or input_ids.size(0)
        for start in range(0,len(input_ids),chunk):
            ids=input_ids[start:start+chunk]
            logits=model(input_ids=ids,attention_mask=attention_mask[start:start+chunk],
                         logits_to_keep=logits_to_keep+1,use_cache=False).logits[:,:-1,:][:,-logits_to_keep:,:].float()
            scaled=logits/self.temperature
            chosen=ids[:,-logits_to_keep:]
            logps=scaled.gather(-1,chosen.unsqueeze(-1)).squeeze(-1)-scaled.logsumexp(-1)
            all_logps.append(logps)
            if compute_entropy:
                with torch.no_grad():
                    raw_entropy.append(distribution_entropy(logits.detach()))
                    all_entropy.append(distribution_entropy(scaled.detach()))
                    # HF generation 先 top-k 后 top-p，与本实验的采样处理顺序一致。
                    flat=scaled.detach().reshape(-1,scaled.size(-1))
                    if self.top_k: flat=TopKLogitsWarper(self.top_k)(None,flat)
                    if self.top_p<1: flat=TopPLogitsWarper(self.top_p)(None,flat)
                    if self.min_p not in (0.,None): raise ValueError('当前采样熵只验证min_p=0')
                    sampling_entropy.append(distribution_entropy(flat).reshape(scaled.shape[:2]))
        result=torch.cat(all_logps)
        entropy=torch.cat(all_entropy) if compute_entropy else None
        if compute_entropy:
            self.audit_policy_logps=result.detach()
            self.audit_entropy={'raw':torch.cat(raw_entropy),'temperature':entropy,
                                'sampling':torch.cat(sampling_entropy)}
        return result,entropy

    def _generate_and_score_completions(self,inputs):
        if any(item['prompt_token_count']>self.max_prompt_length for item in inputs):
            raise ValueError('禁止让TRL截断题面')
        started=time.perf_counter()
        output=super()._generate_and_score_completions(inputs)
        self.restore_checkpointing_configuration()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        if 'old_per_token_logps' not in output or 'ref_per_token_logps' not in output:
            raise RuntimeError('本实验必须显式保存旧策略和SFT参考概率')
        self.audit_rollouts+=1
        rollout=self.audit_rollouts
        reward_rows=self.audit_reward_rows
        if [r['id'] for r in reward_rows]!=[i['record_id'] for i in inputs]: raise RuntimeError('奖励分组顺序不一致')
        for i,item in enumerate(inputs):
            actual=output['prompt_ids'][i][output['prompt_mask'][i].bool()].tolist()
            expected=self.processing_class.encode(item['prompt'],add_special_tokens=False)
            if actual!=expected: raise RuntimeError('实际prompt与完整题面不一致')
        groups=[]
        for start in range(0,len(inputs),self.num_generations):
            subset=reward_rows[start:start+self.num_generations]
            if len({r['id'] for r in subset})!=1: raise RuntimeError('同题回答不连续或混入其他题目')
            expected=relative_advantages([r['reward'] for r in subset])
            actual=output['advantages'][start:start+self.num_generations].float().cpu()
            if not torch.allclose(actual,torch.tensor(expected['advantages']),rtol=1e-5,atol=1e-6):
                raise RuntimeError('优势与独立计算不符')
            groups.append({'id':subset[0]['id'],'rewards':[r['reward'] for r in subset],**expected})
        old,reference=output['old_per_token_logps'],output['ref_per_token_logps']
        if rollout==1:
            error=(old-reference).abs()[output['completion_mask'].bool()].max().item()
            self.audit_first_reference_error=error
            if error>1e-4: raise RuntimeError(f'初始SFT策略与参考概率不同: {error}')
        for index,row in enumerate(reward_rows):
            mask=output['completion_mask'][index].bool()
            old_row=old[index][mask].float().cpu()
            self.audit_old_rows[(rollout,index)]=tensor_sha256(old_row)
            row.update(advantage=output['advantages'][index].item(),
                       old_per_token_logps=old_row.tolist(),
                       ref_per_token_logps=reference[index][mask].float().cpu().tolist(),
                       old_logps_sha256=self.audit_old_rows[(rollout,index)])
        tokens=sum(len(r['completion_ids']) for r in reward_rows)
        self.audit_generated_tokens+=tokens
        self.audit_rollout_seconds+=elapsed
        append_prediction(self.audit_run/'rollout_groups.jsonl',{'rollout_id':rollout,'optimizer_step':self.state.global_step,
                          'groups':groups,'completions':reward_rows,'generated_tokens':tokens,
                          'generation_and_scoring_seconds':elapsed,'reference_policy':'separate_frozen_sft_copy'})
        self._metrics['train']['audit/zero_variance_group_rate'].append(sum(g['zero_variance'] for g in groups)/len(groups))
        self._metrics['train']['audit/effective_group_rate'].append(sum(not g['zero_variance'] for g in groups)/len(groups))
        output['audit_rollout_id']=torch.full((len(inputs),),rollout,device=old.device)
        output['audit_sample_index']=torch.arange(len(inputs),device=old.device)
        return output

    def _compute_loss(self,model,inputs):
        loss=super()._compute_loss(model,inputs)
        if not loss.requires_grad or not torch.isfinite(loss):
            raise RuntimeError('策略loss无有效梯度图或出现非有限值')
        with torch.no_grad():
            mask=inputs['completion_mask'].float()
            old=inputs['old_per_token_logps']
            reference=inputs['ref_per_token_logps']
            for index,(rollout,sample) in enumerate(zip(inputs['audit_rollout_id'].tolist(),inputs['audit_sample_index'].tolist(),strict=True)):
                value=tensor_sha256(old[index][mask[index].bool()].float().cpu())
                if value!=self.audit_old_rows[(rollout,sample)]: raise RuntimeError('缓冲旧策略概率被修改')
            independent,parts=grpo_objective(self.audit_policy_logps,old,reference,inputs['advantages'],mask,
                                            beta=self.beta,epsilon=self.epsilon_low,epsilon_high=self.epsilon_high,
                                            loss_type=self.loss_type,token_normalizer=inputs.get('num_items_in_batch'))
            factor=self.current_gradient_accumulation_steps if self.loss_type=='grpo' else 1
            error=abs(loss.item()*factor-independent.item())
            self.audit_objective_max_error=max(error,self.audit_objective_max_error)
            if error>2e-5: raise RuntimeError('TRL loss 与独立目标不一致')
            self.audit_objective_max_error=max(self.audit_objective_max_error,error)
            ratios=parts['ratio'][mask.bool()].float()
            stats={'rollout_id':inputs['audit_rollout_id'].tolist(),'sample_index':inputs['audit_sample_index'].tolist(),
                   'optimizer_step':self.state.global_step,'micro_step':self.audit_microsteps,
                   'scaled_loss':loss.item(),'unscaled_policy_loss':parts['policy_loss'].item(),
                   'kl':masked_mean(parts['kl'],mask),'ratio_min':ratios.min().item(),'ratio_max':ratios.max().item(),
                   'ratio_quantiles':torch.quantile(ratios,torch.tensor([0.,.05,.5,.95,1.],device=ratios.device)).tolist(),
                   'ratio_low_outside':masked_mean(parts['low_outside'],mask),'ratio_high_outside':masked_mean(parts['high_outside'],mask),
                   'effective_low_clip':masked_mean(parts['effective_low_clip'],mask),
                   'effective_high_clip':masked_mean(parts['effective_high_clip'],mask),
                   'advantage_mean':inputs['advantages'].mean().item(), 'valid_tokens':int(mask.sum().item()),
                   'old_logps_immutable':True,'independent_loss_absolute_error':error,
                   'loss_type':self.loss_type,'gradient_accumulation_steps':self.current_gradient_accumulation_steps,
                   'optimizer_batch_token_normalizer':int(inputs['num_items_in_batch'].item())}
            for key,value in self.audit_entropy.items(): stats['policy_entropy_'+key]=masked_mean(value,mask)
            for sign,selector in [('positive',inputs['advantages']>0),('negative',inputs['advantages']<0)]:
                signmask=mask*selector.unsqueeze(1)
                stats[sign+'_advantage_tokens']=int(signmask.sum())
                stats[sign+'_effective_clip']=masked_mean(parts['effective_low_clip']|parts['effective_high_clip'],signmask) if signmask.sum()>0 else None
            append_prediction(self.audit_run/'policy_audit.jsonl',stats)
            for key in ['policy_entropy_raw','policy_entropy_sampling','ratio_low_outside','ratio_high_outside']:
                self._metrics['train']['audit/'+key].append(stats[key])
            self.audit_microsteps+=1
            self.audit_effective_training_tokens+=int(mask.sum())
        return loss
