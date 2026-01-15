#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import os

from pcbdet.models.factory import build_model, ModelSpec
from pcbdet.utils.wandb import disable_wandb

def main():
    p = argparse.ArgumentParser(description="Evaluate a trained YOLO model on val/test split.")
    p.add_argument("--model", type=str, choices=["baseline", "spdconv"], default="baseline")
    p.add_argument("--weights", type=str, required=True, help="Path to trained weights (.pt)")
    p.add_argument("--data", type=str, required=True, help="Path to YOLO data.yaml")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--project", type=str, default="eval", help="Folder name for evaluation outputs")
    args = p.parse_args()

    disable_wandb()

    w = Path(args.weights)
    if not w.exists():
        raise FileNotFoundError(f"weights not found: {w}")
    d = Path(args.data)
    if not d.exists():
        raise FileNotFoundError(f"data.yaml not found: {d}")

    model = build_model(ModelSpec(kind=args.model, weights=str(w)))
    res = model.val(
        data=str(d),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=300,
        device=args.device,
        half=False,
        plots=False,
        save=False,
        verbose=True,
        project=args.project,
    )
    print("✅ Evaluation complete.")
    return res

if __name__ == "__main__":
    main()
