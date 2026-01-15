from __future__ import annotations
import os
import random
from typing import Optional

import numpy as np

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set seeds for reproducible dataset splitting and (optionally) training."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Torch seeding is optional to avoid hard dependency in dataset-only usage.
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def make_rng(seed: int = 42) -> random.Random:
    return random.Random(seed)
