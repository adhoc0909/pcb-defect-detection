from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import json

def voc_bbox_to_yolo(xmin: float, ymin: float, xmax: float, ymax: float, w: int, h: int) -> Tuple[float, float, float, float]:
    """Convert VOC bbox to normalized YOLO (cx, cy, bw, bh) with clamping."""
    xmin = max(0.0, min(xmin, w - 1))
    ymin = max(0.0, min(ymin, h - 1))
    xmax = max(0.0, min(xmax, w - 1))
    ymax = max(0.0, min(ymax, h - 1))

    bw = xmax - xmin
    bh = ymax - ymin
    cx = xmin + bw / 2.0
    cy = ymin + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h

def save_splits_json(out_path: Path, train: List[str], val: List[str], test: List[str], seed: int) -> None:
    payload = {"seed": seed, "train": train, "val": val, "test": test}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def load_splits_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
