import argparse
import json
import os

from datasets import load_dataset
from tqdm import tqdm

from src.models.factory import ModelFactory
from src.score import score
from src.tasks import get_task


def process_batch(batch, task, model, reverse=False):
    samples = [
        task.from_dict({**item, 'reverse': reverse}) for item in batch
    ]
    difficulty_list = [item['difficulty'] for item in batch]

    return [
        {**item, 'label': 'B' if reverse else 'A', 'difficulty': difficulty}
        for item, difficulty in zip(model.judge(samples), difficulty_list)
    ]


def main(args):
    dataset = load_dataset("mattymchen/codejudgebench", args.task)
    if args.split == "all":
        all_splits = dataset.keys()
    else:
        all_splits = [args.split]

    for split in all_splits:
        data = dataset[split].to_list()
        model = ModelFactory.get_model(args.model_name)(args)
        task = get_task(args.task)

        results = []
        batch_size = args.batch_size
        for start in tqdm(range(0, len(data), batch_size)):
            batch = data[start:start+batch_size]
            results.extend(process_batch(batch, task, model, reverse=False))
            results.extend(process_batch(batch, task, model, reverse=True))

        # Save results
        os.makedirs("outputs", exist_ok=True)
        output_file = f"outputs/{args.model_name.split('/')[-1]}_{args.task}-{split}.jsonl"
        with open(output_file, 'w') as f:
            for res in results:
                f.write(json.dumps(res) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--task", choices=['codegen', 'coderepair', 'testgen'], required=True)
    parser.add_argument("--split", type=str, default='all')
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    main(args)
    score(args)
