import os
import argparse

def seed_everything(seed: int):  
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--module_name', type=str, default='TCRAG')
    parser.add_argument('-m', '--model_name', type=str, default='Qwen', choices=['Qwen', 'Xiaobei', 'Qwen2', 'Aliyun'])
    parser.add_argument('-d', '--dataset_name', type=str, default='HotpotQA', choices=['CMB', 'MMCU', 'Clin','2WikiMultihopQA','HotpotQA', 'StrategyQA','IIRC'])#StrategyQA
    parser.add_argument('-c', '--checkpoint', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    print(args)

    module_name = args.module_name
    model_name = args.model_name
    dataset_name = args.dataset_name

    checkpoint_dir = './checkpoint/{}/{}/{}'.format(module_name, dataset_name, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    args.checkpoint_dir = checkpoint_dir

    result_dir = './result/{}/{}/{}'.format(module_name, dataset_name, model_name)
    os.makedirs(result_dir, exist_ok=True)
    result_path = f"{result_dir}/raw_test_full.json"
    args.result_path = result_path
    ex_result_path = f"{result_dir}/process_test_full.json"
    args.ex_result_path = ex_result_path

    return args

from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu(reference, candidate):
    """
    计算候选答案与参考答案之间的 BLEU 分数（BLEU-1 到 BLEU-4）。

    参数：
    reference：标准答案列表（列表的列表）
    candidate：候选答案列表（列表）

    返回：
    bleu_scores：BLEU 分数（字典）
    """
    bleu_scores = {}
    for i in range(1, 5):
        # 使用指定的平滑函数
        smoothie = SmoothingFunction().method4
        bleu_scores[f'BLEU-{i}'] = sentence_bleu(reference, candidate, weights=(1 / i, 0, 0, 0),
                                                 smoothing_function=smoothie)

    return bleu_scores


def calculate_rouge(reference, candidate):
    """
    计算候选答案与参考答案之间的 ROUGE 分数。

    参数：
    reference：标准答案字符串
    candidate：候选答案字符串

    返回：
    rouge_scores：ROUGE 分数（字典）
    """
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference, avg=True)
    return scores

if __name__ == '__main__':
    # 示例数据
    query = "What is the capital of France?"
    answer = "The capital of France is Paris."
    label_answer = "Paris is the capital of France."
    #
    # 计算 BLEU 分数
    bleu_score = calculate_bleu([label_answer], answer)
    
    # 计算 ROUGE 分数
    rouge_scores = calculate_rouge(label_answer, answer)

    # 打印结果
    print("BLEU-1 score:", bleu_score['BLEU-1'])
    print("BLEU-4 score:", bleu_score['BLEU-4'])
    print("ROUGE scores:", rouge_scores['rouge-l']['r'])
