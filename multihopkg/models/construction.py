"""
LG:
Mostly meant as a way to organize functions that construct models. 
"""
from transformers import AutoTokenizer, PreTrainedTokenizer

def construct_env_model(tokenizer: PreTrainedTokenizer):
    """
    Will construct the Learning Framework History Recording Model
    """
    # Now maybe this one will be the same as GraphSearchPolicy. But just adding the langauge embedding

def construct_navagent(args):
    """
    Will construct the Learning Framework Graph Traversing Model
    """
    raise NotImplementedError

def construct_embedding_model(args):
    """
    Likely will simply be for loading. 
    """
    raise NotImplementedError
