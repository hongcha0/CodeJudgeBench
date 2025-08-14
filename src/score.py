import argparse
import json
import os

import numpy as np
from datasets import load_dataset


def score(args):
    splits = load_dataset("mattymchen/codejudgebench", args.task).keys()
    all_score = []
    all_difficulty = []
    for s in splits:
        filepath = f"outputs/{args.model_name.split('/')[-1]}_{args.task}-{s}.jsonl"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                result = [json.loads(line) for line in f.readlines()]
            score = [item['pred'] == item['label'] for item in result]
            all_score.extend(score)
            all_difficulty.extend([item['difficulty'] for item in result])
            print(filepath, f"{np.mean(score)*100:.2f}")
        else:
            print("MISSING", filepath)

    print(f"================== {args.task} ==================")
    easy_score = [x for x, d in zip(all_score, all_difficulty) if d == "easy"]
    med_score = [x for x, d in zip(all_score, all_difficulty) if d == "medium"]
    hard_score = [x for x, d in zip(all_score, all_difficulty) if d == "hard"]
    print("Easy", f"{np.mean(easy_score)*100:.2f}")
    print("Medium", f"{np.mean(med_score)*100:.2f}")
    print("Hard", f"{np.mean(hard_score)*100:.2f}")

    print("Micro Avg", f"{np.mean(all_score)*100:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--task", type=str)
    args = parser.parse_args()
    score(args)
