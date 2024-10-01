import torch
import numpy as np

def set_seeds(seed):
    import torch
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
