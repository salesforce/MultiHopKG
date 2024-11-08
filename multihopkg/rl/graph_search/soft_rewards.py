import torch
from abc import ABC, abstractmethod

from multihopkg.knowledge_graph import ITLKnowledgeGraph

class SoftRewardCalculator(ABC):
    @abstractmethod
    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

        
class Simple_SRC(SoftRewardCalculator):
    def __init__(self, knowledge_graph: ITLKnowledgeGraph, reward_shaping_threshold: float):
        # TODO: we might want to use a threshold for making sure penalty is not too high
        self.reward_shaping_threshold = reward_shaping_threshold
        self.knowledge_graph = knowledge_graph
        
    def __call__(self, resulting_state: torch.Tensor) -> torch.Tensor:
        """
        Will take the resulting state and return a reward that should denote how close the agent got to the goal.
        """
        return actions.abs().sum(dim=-1)
