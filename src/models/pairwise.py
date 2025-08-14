import os

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from .base_model import BaseModel


class PairwiseModel(BaseModel):
    JUDGE_PROMPT = """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

Begin your evaluation by generating your own answer to the prompt. You must provide your answers before judging any answers.

When evaluating the assistants' answers, compare both assistants' answers with your answer. You must identify and correct any mistakes or inaccurate information.

Then consider if the assistant's answers are helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive.

Then consider the creativity and novelty of the assistant's answers when needed. Finally, identify any missing important information in the assistants' answers that would be beneficial to include when responding to the user prompt.

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is better: [[A>B]] 
2. Assistant B is better: [[B>A]]

Example output: "My final verdict is Assistant A is better: [[A>B]]".

<|User Prompt|>
{query}

<|The Start of Assistant A's Answer|>
{answer_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{answer_b}
<|The End of Assistant B's Answer|>
""".strip()
    RESPONSE_PATTERNS_A = ["[[A>B]]", "boxed{A>B}"]
    RESPONSE_PATTERNS_B = ["[[B>A]]", "boxed{B>A}"]
    SAMPLING_PARAMS = None

    def __init__(self, args):
        super().__init__()
        self.model = LLM(
            model=args.model_name,
            dtype='bfloat16',
            gpu_memory_utilization=0.95,
            tensor_parallel_size=self._get_gpu_count(),
        )
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if self.SAMPLING_PARAMS is None:
            self.SAMPLING_PARAMS = SamplingParams(
                n=1,
                max_tokens=32768,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k
            )

    def _get_gpu_count(self):
        cuda_visible = os.getenv('CUDA_VISIBLE_DEVICES', '')
        return len(cuda_visible.split(',')) if cuda_visible else 1

    def format_prompt(self, input):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": input}],
            tokenize=False,
            add_generation_prompt=True
        )

    def extract_response(self, output):
        if output.finish_reason != 'stop':
            return {
                "pred": "",
                "response": output.text
            }

        text = output.text
        if "</think>" in text:
            text = text.split("</think>")[1]

        pred_a = max([text.rfind(pattern) for pattern in self.RESPONSE_PATTERNS_A])
        pred_b = max([text.rfind(pattern) for pattern in self.RESPONSE_PATTERNS_B])

        pred = ""
        if pred_a > pred_b:
            pred = "A"
        elif pred_a < pred_b:
            pred = "B"

        return {
            "pred": pred,
            "response": output.text
        }

    def judge(self, samples):
        text = [self.format_prompt(s.get_prompt(self.JUDGE_PROMPT)) for s in samples]
        outputs = self.model.generate(text, self.SAMPLING_PARAMS)

        return [self.extract_response(o.outputs[0]) for o in outputs]
