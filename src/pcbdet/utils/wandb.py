from __future__ import annotations
import os

def disable_wandb() -> None:
    """Disable Weights & Biases auto-logging (common source of errors in Colab)."""
    os.environ.setdefault("WANDB_DISABLED", "true")
    os.environ.setdefault("WANDB_SILENT", "true")
