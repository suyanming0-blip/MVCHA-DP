import numpy as np
import torch
_values = {}


def key(value):
    a = value.detach().cpu().numpy().astype(np.float32) if torch.is_tensor(value) else np.asarray(value, dtype=np.float32)
    return (a.shape, a.tobytes())


def register(views, similarities):
    for (name, sim) in similarities.items():
        k = key(views[name])
        sim.flags.writeable = False
        _values[k] = sim


def cached_similarity(value):
    k = key(value)
    return _values[k].copy() if k in _values else None


def clear():
    _values.clear()
