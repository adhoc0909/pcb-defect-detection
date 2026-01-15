"""PyTorch compatibility helpers.

PyTorch 2.6 flips torch.load default weights_only=True, which can break loading Ultralytics .pt files
unless allowlisted. For trusted checkpoints, we patch torch.load default to weights_only=False to
match historical behavior.
"""
from __future__ import annotations

def patch_torch_load_weights_only_false() -> None:
    try:
        import torch
    except Exception:
        return

    orig = torch.load

    def _patched(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return orig(*args, **kwargs)

    torch.load = _patched  # type: ignore[attr-defined]
