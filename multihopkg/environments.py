# Import Abstract Classes
from abc import ABC, abstractmethod

class Environment(ABC):

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self):
        pass


