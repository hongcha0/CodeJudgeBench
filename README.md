# CodeJudgeBench
This repository contains the code for the paper "*CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks*". 
CodeJudgeBench is a benchmark aimed at evaluating LLM-based judges for coding related tasks.

<a target="_blank" href="https://arxiv.org/abs/2507.10535">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-red?style=flat&logo=arxiv">
</a>
<a target="_blank" href="https://huggingface.co/datasets/mattymchen/codejudgebench">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20Dataset-yellow?style=flat">
</a>

## Updates
- [13/02/2026] 🚀 CodeJudgeBench Adversarial is out — see [[HERE]](#codejudgebench-adversarial) | [[DATASET]](https://huggingface.co/datasets/mattymchen/codejudgebench-adversarial).

## Installation
```bash
pip install -r requirements.txt
```

## Quickstart
Simply change the `model_name` parameter to test any LLMs supported by `vllm`.
```bash
# To run codegen task
python run.py --model_name Qwen/Qwen3-8B --task codegen --batch_size 32

# To run coderepair task
python run.py --model_name Qwen/Qwen3-8B --task coderepair --batch_size 32

# To run testgen task
python run.py --model_name Qwen/Qwen3-8B --task testgen --batch_size 32

# To run a specific task split
python run.py --model_name Qwen/Qwen3-8B --task codegen --split gemini_2.5_pro --batch_size 32


# To get results for a specific model
python -m src.score --model_name Qwen3/Qwen3-8B --task codegen
python -m src.score --model_name Qwen3/Qwen3-8B --task coderepair
python -m src.score --model_name Qwen3/Qwen3-8B --task testgen
```

## Using OpenAI API
Create a `.env` file in the project root (e.g., to store OPENAI_API_KEY and any other required variables).
```bash
python run.py --model_name gpt-5-nano --task codegen --batch_size 32 --backend openai
```

## CodeJudgeBench-Adversarial
The adversarial version of CodeJudgeBench makes judging harder by having an LLM apply semantics-preserving adversarial transformations (e.g., variable renaming, modifying comments) to the original positive and negative code candidates to induce misjudgments.

Run the following commands to evaluate a judge on CodeJudgeBench-Adversarial. 
```bash
python run.py --model_name Qwen/Qwen3-8B --task codegen --dataset codejudgebench-adversarial --batch_size 32

# To get results for a specific model
python -m src.score --model_name Qwen3/Qwen3-8B --task codegen --dataset codejudgebench-adversarial

# To run baseline
python run.py --model_name Qwen/Qwen3-8B --task codegen --split gemini_2.5_pro --dataset codejudgebench --batch_size 32
python run.py --model_name Qwen/Qwen3-8B --task codegen --split claude_3.7_sonnet --dataset codejudgebench --batch_size 32
python run.py --model_name Qwen/Qwen3-8B --task codegen --split gemini_2.5_flash --dataset codejudgebench --batch_size 32
```

The results below are averaged across the `gemini_2.5_pro`, `claude_3.7_sonnet`, and `gemini_2.5_flash` splits.
| Model | CodeJudgeBench | CodeJudgeBench Adversarial |
|:--|--:|--:|
| Qwen3-32B | 74.93 | 52.82 |
| RM-R1-32B | 66.91 | 23.29 |
| QwQ-32B | 75.82 | 51.24 |
| AceReason-14B | 72.75 | 45.20 |

## Citation
If you find CodeJudgeBench useful or relevant to your work, please kindly cite our paper:
```bibtex
@article{jiang2025codejudgebench,
  title   = {CodeJudgeBench: Benchmarking LLM-as-a-Judge for Coding Tasks},
  author  = {Hongchao Jiang and Yiming Chen and Yushi Cao and Hung-yi Lee and Robby T. Tan},
  year    = {2025},
  journal = {arXiv preprint arXiv: 2507.10535}
}
```
