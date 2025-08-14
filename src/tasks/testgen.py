from dataclasses import dataclass, fields


@dataclass
class TestGenTask:
    question_content: str
    pos_input: str
    neg_input: str
    pos_response: str
    neg_response: str
    task_prompt: str = """
For the below coding question, give a unit test including test input and test output.
### Question
{question_content}
""".strip()
    unit_test: str = """
### Test Input
```
{test_in}
```

### Test Output
```
{answer}
```
""".strip()
    reverse: bool = False

    def get_prompt(self, prompt_template):
        if self.reverse:
            in_a = self.neg_input
            answer_a = self.neg_response
            in_b = self.pos_input
            answer_b = self.pos_response
        else:
            in_a = self.pos_input
            answer_a = self.pos_response
            in_b = self.neg_input
            answer_b = self.neg_response

        return prompt_template.format(
            query=self.task_prompt.format(question_content=self.question_content),
            answer_a=self.unit_test.format(answer=answer_a, test_in=in_a),
            answer_b=self.unit_test.format(answer=answer_b, test_in=in_b),
        )

    @classmethod
    def from_dict(cls, d):
        cls_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in cls_fields}
        return cls(**filtered)
