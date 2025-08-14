from abc import ABC, abstractmethod
from typing import Dict, List, Protocol


class TaskProtocol(Protocol):
    def get_prompt(self, prompt_template: str) -> str: ...


class BaseModel(ABC):
    @abstractmethod
    def judge(self, samples: List[TaskProtocol]) -> List[Dict[str, str]]:
        pass
