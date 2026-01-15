from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil
import yaml

from .voc import VocParser
from .yolo import voc_bbox_to_yolo, save_splits_json
from ..utils.repro import make_rng

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

@dataclass
class PrepareConfig:
    root: Path                       # VOC root
    out_dir: Path                    # YOLO output root
    seed: int = 42
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    verify_img_size: bool = False    # if True, read image header for size (slower)
    copy_metadata: bool = False      # if True, use copy2; else copyfile (faster)

def _fast_copy(src: Path, dst: Path, copy_metadata: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy_metadata:
        shutil.copy2(src, dst)
    else:
        shutil.copyfile(src, dst)

def _build_stem_map(image_paths: List[Path]) -> Dict[str, Path]:
    # If duplicated stems exist, first one wins (logically fine if dataset is consistent)
    m: Dict[str, Path] = {}
    for p in image_paths:
        m.setdefault(p.stem, p)
    return m

def _find_image_for_xml(xml_path: Path, filename_from_xml: str, stem_map: Dict[str, Path]) -> Optional[Path]:
    if filename_from_xml:
        stem = Path(filename_from_xml).stem
        if stem in stem_map:
            return stem_map[stem]
    return stem_map.get(xml_path.stem, None)

def _get_image_size(path: Path, fallback_w: int, fallback_h: int) -> Tuple[int, int]:
    # Optional fast header-only size reading
    try:
        import imagesize  # type: ignore
    except Exception:
        return fallback_w, fallback_h

    try:
        w, h = imagesize.get(str(path))
        if w and h:
            return int(w), int(h)
    except Exception:
        pass
    return fallback_w, fallback_h

def prepare_voc_to_yolo(cfg: PrepareConfig) -> Path:
    """Prepare YOLO dataset from VOC-style folder.

    Expected VOC layout:
      root/images/{class_name}/*.(jpg|png|...)
      root/Annotations/{class_name}/*.xml

    Output YOLO layout:
      out/images/{train,val,test}/...
      out/labels/{train,val,test}/...
      out/data.yaml
      out/splits.json  (for exact reproducibility)
    """
    root = cfg.root
    out_dir = cfg.out_dir

    # Collect paths (avoid rglob for speed; structure is known)
    img_base = root / "images"
    ann_base = root / "Annotations"
    if not img_base.exists() or not ann_base.exists():
        raise FileNotFoundError(f"Expected {img_base} and {ann_base} to exist")

    images: List[Path] = []
    xmls: List[Path] = []

    for cls_dir in sorted([p for p in img_base.iterdir() if p.is_dir()]):
        for p in cls_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                images.append(p)

    for cls_dir in sorted([p for p in ann_base.iterdir() if p.is_dir()]):
        xmls.extend(sorted(cls_dir.glob("*.xml")))

    images = sorted(images, key=lambda p: str(p))
    xmls = sorted(xmls, key=lambda p: str(p))

    stem_map = _build_stem_map(images)

    parser = VocParser()

    # Build class mapping from ALL original xmls
    cls_set = set()
    bad_xml = 0
    for xp in xmls:
        try:
            ann = parser.parse(xp)
            for obj in ann.objects:
                if obj.cls:
                    cls_set.add(obj.cls)
        except Exception:
            bad_xml += 1

    classes = sorted(cls_set)
    cls2id = {c: i for i, c in enumerate(classes)}

    # Pair images with xmls
    pairs: List[Tuple[Path, Path]] = []
    missing = 0
    for xp in xmls:
        try:
            ann = parser.parse(xp)
            ip = _find_image_for_xml(xp, ann.filename, stem_map)
            if ip is None:
                missing += 1
                continue
            pairs.append((ip, xp))
        except Exception:
            continue

    # Deterministic shuffle & split
    rng = make_rng(cfg.seed)
    pairs = sorted(pairs, key=lambda t: str(t[0]))
    rng.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train + n_val]
    test_pairs = pairs[n_train + n_val:]

    # Prepare output dirs
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ["train", "val", "test"]:
        (out_dir / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (out_dir / f"labels/{split}").mkdir(parents=True, exist_ok=True)

    # Optional header-only size reader
    if cfg.verify_img_size:
        try:
            import imagesize  # noqa: F401
        except Exception as e:
            raise RuntimeError("verify_img_size=True requires `pip install imagesize`") from e

    def write_label(dst: Path, ann_w: int, ann_h: int, objects) -> None:
        lines: List[str] = []
        for obj in objects:
            if obj.cls not in cls2id:
                continue
            x, y, bw, bh = voc_bbox_to_yolo(obj.xmin, obj.ymin, obj.xmax, obj.ymax, ann_w, ann_h)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{cls2id[obj.cls]} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
        dst.write_text("\n".join(lines), encoding="utf-8")

    def convert_split(split_pairs: List[Tuple[Path, Path]], split: str) -> None:
        img_out = out_dir / f"images/{split}"
        lab_out = out_dir / f"labels/{split}"

        for ip, xp in split_pairs:
            dst_img = img_out / ip.name
            _fast_copy(ip, dst_img, cfg.copy_metadata)

            ann = parser.parse(xp)
            w, h = ann.width, ann.height
            if cfg.verify_img_size:
                w, h = _get_image_size(ip, w, h)

            dst_txt = lab_out / (ip.stem + ".txt")
            write_label(dst_txt, w, h, ann.objects)

    convert_split(train_pairs, "train")
    convert_split(val_pairs, "val")
    convert_split(test_pairs, "test")

    # Save splits for exact reproducibility
    splits_path = out_dir / "splits.json"
    save_splits_json(
        splits_path,
        train=[p[0].name for p in train_pairs],
        val=[p[0].name for p in val_pairs],
        test=[p[0].name for p in test_pairs],
        seed=cfg.seed,
    )

    data_yaml = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(classes)},
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # Return yaml path
    return yaml_path
