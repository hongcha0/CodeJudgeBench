from vllm import SamplingParams

from .pairwise import PairwiseModel


class Qwen3(PairwiseModel):
    SAMPLING_PARAMS = SamplingParams(
        n=1,
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0
    )
