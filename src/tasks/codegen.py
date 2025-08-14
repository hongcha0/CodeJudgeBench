from dataclasses import dataclass, fields


@dataclass
class CodeGenTask:
    question_content: str
    pos_response: str
    neg_response: str
    reverse: bool = False

    def get_prompt(self, prompt_template):
        if self.reverse:
            answer_a = self.neg_response
            answer_b = self.pos_response
        else:
            answer_a = self.pos_response
            answer_b = self.neg_response

        return prompt_template.format(
            query=self.question_content,
            answer_a=answer_a,
            answer_b=answer_b
        )

    @classmethod
    def from_dict(cls, d):
        cls_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in cls_fields}
        return cls(**filtered)
