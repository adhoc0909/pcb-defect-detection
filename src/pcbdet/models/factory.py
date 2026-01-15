from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

from ..utils.torch_compat import patch_torch_load_weights_only_false
from ..utils.wandb import disable_wandb

ModelKind = Literal["baseline", "spdconv"]

@dataclass
class ModelSpec:
    kind: ModelKind
    weights: str  # path or model name (e.g., 'yolov8s.pt')

def build_model(spec: ModelSpec):
    """Build YOLO model.

    Notes:
    - For PyTorch 2.6+, we patch torch.load default weights_only=False for trusted checkpoints.
    - For baseline: requires `pip install ultralytics`.
    - For spdconv: requires installing the SPDConv fork/patch so that YOLO can load SPD-modified models.
    """
    disable_wandb()
    patch_torch_load_weights_only_false()

    from ultralytics import YOLO  # import after patches

    return YOLO(spec.weights)
