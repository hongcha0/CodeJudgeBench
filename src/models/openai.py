from .pairwise import PairwiseModel
import os
import openai
from dotenv import load_dotenv
from multiprocessing.pool import ThreadPool
from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    finish_reason: str

class OpenAIModel(PairwiseModel):
    def __init__(self, args):
        load_dotenv()
        self.args = args
        self.client = openai.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=args.base_url
        )
    
    def completion(self, prompt):
        try:
            messages=[
                {"role": "user", "content": prompt},
            ]
            response = self.client.chat.completions.create(
                model=self.args.model_name,
                messages=messages,
                max_tokens=32768,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                extra_body={"top_k": self.args.top_k},
                timeout=60000,
            )
            output = Completion(
                text=response.choices[0].message.content,
                finish_reason=response.choices[0].finish_reason
            )
        except Exception as e:
            print(e)
            output = Completion(
                text=f"Error: {str(e)}",
                finish_reason=""
            )

        return output

    def judge(self, samples):
        text = [s.get_prompt(self.JUDGE_PROMPT) for s in samples]
        with ThreadPool(processes=len(text)) as pool:
            outputs = list(pool.imap(self.completion, text))
        
        return [self.extract_response(o) for o in outputs]