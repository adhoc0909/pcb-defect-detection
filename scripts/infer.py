#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

from pcbdet.infer.infer_one import InferConfig, infer_one_image

def main():
    p = argparse.ArgumentParser(description="Run inference on a single image (supports HEIC with pillow-heif).")
    p.add_argument("--model", type=str, choices=["baseline", "spdconv"], default="baseline")
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--img", type=str, required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.6)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--save_dir", type=str, default="./inference_out")
    args = p.parse_args()

    cfg = InferConfig(
        weights=args.weights,
        model_kind=args.model,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_dir=args.save_dir,
    )
    results, saved_img = infer_one_image(cfg, args.img)
    print(f"✅ Saved: {saved_img}")
    return results

if __name__ == "__main__":
    main()
