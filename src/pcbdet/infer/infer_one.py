from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

from ..models.factory import build_model, ModelSpec
from .heic import maybe_convert_heic

@dataclass
class InferConfig:
    weights: str
    model_kind: str = "baseline"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.6
    device: Union[int, str] = 0
    save_dir: str = "./inference_out"
    show: bool = False

def infer_one_image(
    *,
    model,
    image_path: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.6,
    device=0,
    save_dir: str = "./inference_out",
):
    img_path = maybe_convert_heic(image_path)
    img_p = Path(img_path)
    if not img_p.exists():
        raise FileNotFoundError(f"image not found: {img_p}")

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(img_p),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        save=True,
        project=str(out_dir),
        name="pred",
        exist_ok=True,
        verbose=False,
    )

    saved_img = out_dir / "pred" / img_p.name
    return results, saved_img
