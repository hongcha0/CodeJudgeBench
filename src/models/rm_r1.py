from vllm import SamplingParams

from .pairwise import PairwiseModel


class RMR1(PairwiseModel):
    JUDGE_PROMPT = (
        "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client question displayed below. \n\n"
        "[Client Question]\n{query}\n\n[The Start of Chatbot A's Response]\n{answer_a}\n[The End of Chatbot A's Response]\n\n"
        "[The Start of Chatbot B's Response]\n{answer_b}\n[The End of Chatbot B's Response]" + "\n\n"
        "Output your final verdict at last by strictly following this format: "
        "'<answer>[[A]]</answer>' if Chatbot A is better, or '<answer>[[B]]</answer>' if Chatbot B is better."
    )
    RESPONSE_PATTERNS_A = ["[[A]]", "boxed{A}", "[[A]", "[A]]"]
    RESPONSE_PATTERNS_B = ["[[B]]", "boxed{B}", "[[B]", "[B]]"]
    SAMPLING_PARAMS = SamplingParams(
        n=1,
        max_tokens=32768,
        temperature=0,
    )
