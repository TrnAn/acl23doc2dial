import random
import numpy as np
import torch
import os

seed_value = int(os.environ.get('SEED_VALUE', '0'))

def set_seed():
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False