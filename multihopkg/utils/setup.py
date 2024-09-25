

def set_seed(seed):
    import torch
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
