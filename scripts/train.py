#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import os
import yaml

from pcbdet.models.factory import build_model, ModelSpec
from pcbdet.utils.repro import set_seed
from pcbdet.utils.wandb import disable_wandb

def main():
    p = argparse.ArgumentParser(description="Train YOLOv8 (baseline or spdconv).")
    p.add_argument("--model", type=str, choices=["baseline", "spdconv"], default="baseline")
    p.add_argument("--weights", type=str, default="yolov8s.pt", help="Pretrained weights or model name")
    p.add_argument("--data", type=str, required=True, help="Path to YOLO data.yaml")
    p.add_argument("--cfg", type=str, default=None, help="YAML file with train args (overrides defaults)")
    p.add_argument("--project", type=str, default="runs_pcb", help="Ultralytics project name (NOT a path)")
    p.add_argument("--name", type=str, default=None, help="Run name under project/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="0")
    args = p.parse_args()

    disable_wandb()
    set_seed(args.seed, deterministic=False)

    train_args = dict(
        data=args.data,
        imgsz=640,
        epochs=100,
        batch=16,
        patience=50,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=5e-4,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=0.0, translate=0.1, scale=0.5, shear=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        device=args.device,
        workers=2,
        project=args.project,
    )
    if args.name:
        train_args["name"] = args.name

    if args.cfg:
        cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("--cfg must be a YAML dict")
        train_args.update(cfg)

    # IMPORTANT: avoid passing duplicate keys (common notebook bug)
    # Also: Ultralytics will create runs under {project}/{name}
    model = build_model(ModelSpec(kind=args.model, weights=args.weights))
    results = model.train(**train_args)
    print("✅ Training complete.")
    return results

if __name__ == "__main__":
    main()
