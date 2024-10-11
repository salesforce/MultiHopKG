# Import Abstract Classes
from abc import ABC, abstractmethod

class Environment(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        pass


class QnAGraphEnvironment(Environment):
     def __init__(self):

        # NOTE: :We dont really need embeddings here since we are just sampling at random. 
        # TODO: But herpahs we want to get a ANN search here. Since once the action is taken we want to get the closes neighbor. 
        pass

    def reset(self):
        pass

    def step(self):
        return

    def _find_closest_neighbor(self, position: torch.Tensor):
        raise NotImplementedError
    
