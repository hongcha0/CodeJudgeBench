from vllm import SamplingParams

from .pairwise import PairwiseModel


class AceReason(PairwiseModel):
    SAMPLING_PARAMS = SamplingParams(
        n=1,
        max_tokens=32768,
        temperature=0.6,
        top_p=0.95
    )

    def format_prompt(self, input):
        return "<｜User｜>\n\n" + input + "\n<｜Assistant｜><think>\n"
