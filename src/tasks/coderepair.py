from dataclasses import dataclass, field, fields


@dataclass
class CodeRepairTask:
    question_content: str
    wrong_code: str
    wrong_meta: dict
    pos_response: str
    neg_response: str
    error_message: str = field(init=False)
    task_prompt: str = """
You are provided with a coding question, a piece of code, and the associated error message. 
You need to generate a corrected version of the code.

### Question
{question_content}

### Program to be fixed:
```python
{wrong_code}
```


### Error Information:
{error_message}
""".strip()
    reverse: bool = False

    def __post_init__(self):
        self.error_message = self._get_error_message(self.wrong_meta)

    def get_prompt(self, prompt_template):
        if self.reverse:
            answer_a = self.neg_response
            answer_b = self.pos_response
        else:
            answer_a = self.pos_response
            answer_b = self.neg_response

        return prompt_template.format(
            query=self.task_prompt.format(
                question_content=self.question_content,
                wrong_code=self.wrong_code,
                error_message=self.error_message,
            ),
            answer_a=answer_a,
            answer_b=answer_b
        )

    def _get_error_message(self, metadata):
        if metadata["error_code"] == -1:
            # time limit exceeded
            message = f"The above code is incorrect and got the following compilation error.\n{metadata['error']}"
        elif metadata["error_code"] == -2:
            # wrong answer
            message = f"The above code is incorrect and got a wrong answer.\nInput:\n{metadata['inputs']}\n\nGenerated Output:\n{metadata['output']}\n\nExpected:\n{metadata['expected']}"
        elif metadata["error_code"] == -3:
            # time limit exceeded
            message = f"The above code is incorrect and got time limit exceeded.\n{metadata['error']}\nInput:\n{metadata['inputs']}\n\nExpected:\n{metadata['expected']}"
            pass
        elif metadata["error_code"] == -4:
            # runtime error
            if 'inputs' not in metadata:
                # print(metadata)
                message = f"The above code is incorrect and got a runtime error.\n{metadata['error_message']}"
                # message = ''
            else:
                message = f"The above code is incorrect and got a runtime error.\nInput:\n{metadata['inputs']}\n\nExpected:\n{metadata['expected']}\n\n{metadata['error']}"
        else:
            raise NotImplementedError(
                f"metadata['error_code'] = {metadata['error_code']} not implemented || {metadata=}"
            )
        return message

    @classmethod
    def from_dict(cls, d):
        cls_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in cls_fields}
        return cls(**filtered)
