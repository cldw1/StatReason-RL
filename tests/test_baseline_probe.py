from src.evaluation.baseline_probe import add_demonstrations,choose_demonstrations
import pytest


def test_fewshot_never_reads_target_gold_and_uses_training_only():
    target={'id':'target','task_type':'a','question':'target_question','answer':'secret_gold','solution':'secret_solution'}
    train=[{'id':str(i),'task_type':'a','split':'sft','question':f'example_{i}','solution':'FINAL_ANSWER: 1'} for i in range(3)]
    demos=choose_demonstrations(target,train)
    assert demos==choose_demonstrations(target,train)
    text=add_demonstrations(target,demos)['question']
    assert 'target_question' in text and 'secret_gold' not in text and 'secret_solution' not in text
    with pytest.raises(ValueError):add_demonstrations(target,[{**train[0],'split':'validation'}])
    with pytest.raises(ValueError):add_demonstrations(target,[{**train[0],'id':'target'}])
