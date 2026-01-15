from __future__ import annotations
from pathlib import Path
from typing import Union

def is_heic(path: Union[str, Path]) -> bool:
    p = Path(path)
    return p.suffix.lower() in {".heic", ".heif"}


def heic_to_jpg(heic_path: Union[str, Path], out_path: Union[str, Path, None] = None) -> Path:
    """
    Convert HEIC/HEIF image to JPG.
    Requires: pillow-heif, Pillow
    """
    heic_path = Path(heic_path)
    if not heic_path.exists():
        raise FileNotFoundError(f"HEIC file not found: {heic_path}")

    if out_path is None:
        out_path = heic_path.with_suffix(".jpg")
    out_path = Path(out_path)

    # Lazy import so baseline users don't need pillow-heif installed
    try:
        import pillow_heif  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:
        raise ImportError(
            "HEIC support requires 'pillow-heif' and 'Pillow'. "
            "Install with: pip install pillow-heif Pillow"
        ) from e

    pillow_heif.register_heif_opener()
    img = Image.open(heic_path).convert("RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="JPEG", quality=95)

    return out_path


def load_image_auto(path: Union[str, Path]) -> Path:
    """
    If input is HEIC/HEIF -> convert to JPG and return new path.
    Else return original path.
    """
    p = Path(path)
    if is_heic(p):
        return heic_to_jpg(p)
    return p


def maybe_convert_heic(image_path: str) -> str:
    """Convert .heic/.heif to .jpg if needed. Returns path to usable image."""
    p = Path(image_path)
    suf = p.suffix.lower()
    if suf not in [".heic", ".heif"]:
        return str(p)

    try:
        import pillow_heif  # type: ignore
        from PIL import Image
    except Exception as e:
        raise RuntimeError("HEIC input requires `pip install pillow-heif`") from e

    pillow_heif.register_heif_opener()
    out_path = p.with_suffix(".jpg")
    img = Image.open(p).convert("RGB")
    img.save(out_path, format="JPEG", quality=95)
    return str(out_path)
