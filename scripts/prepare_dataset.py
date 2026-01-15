#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

from pcbdet.data.prepare import PrepareConfig, prepare_voc_to_yolo
from pcbdet.utils.repro import set_seed

def main():
    p = argparse.ArgumentParser(description="Prepare YOLO dataset from VOC folders (original only).")
    p.add_argument("--root", type=str, required=True, help="VOC root (contains images/ and Annotations/)")
    p.add_argument("--out", type=str, required=True, help="Output directory for YOLO dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--verify_img_size", action="store_true", help="Read image header to verify width/height (slower)")
    p.add_argument("--copy_metadata", action="store_true", help="Use shutil.copy2 instead of copyfile (slower)")
    args = p.parse_args()

    set_seed(args.seed, deterministic=False)

    cfg = PrepareConfig(
        root=Path(args.root),
        out_dir=Path(args.out),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        verify_img_size=args.verify_img_size,
        copy_metadata=args.copy_metadata,
    )
    yaml_path = prepare_voc_to_yolo(cfg)
    print(f"✅ Prepared YOLO dataset. data.yaml -> {yaml_path}")

if __name__ == "__main__":
    main()
