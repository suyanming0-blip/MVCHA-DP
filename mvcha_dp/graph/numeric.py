import numpy as np
import torch


def _to_numpy_float32(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)
