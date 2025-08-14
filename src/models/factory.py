from .acereason import AceReason
from .pairwise import PairwiseModel
from .qwen3 import Qwen3
from .qwq import QwQ
from .rm_r1 import RMR1


class ModelFactory:
    registry = {
        "nvidia/AceReason-Nemotron": AceReason,
        "gaotang/RM-R1": RMR1,
        "Qwen/Qwen3": Qwen3,
        "Qwen/QwQ-32B": QwQ
    }

    @classmethod
    def get_model(cls, model_name):
        for k, v in cls.registry.items():
            if k in model_name:
                return v
        return PairwiseModel
