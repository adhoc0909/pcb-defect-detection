from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET

@dataclass(frozen=True)
class VocObject:
    cls: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

@dataclass(frozen=True)
class VocAnnotation:
    filename: str
    width: int
    height: int
    objects: Tuple[VocObject, ...]

class VocParser:
    """Fast VOC XML parser with memoization."""
    def __init__(self):
        self._cache: Dict[str, VocAnnotation] = {}

    def parse(self, xml_path: Path) -> VocAnnotation:
        key = str(xml_path)
        if key in self._cache:
            return self._cache[key]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename = root.findtext("filename") or ""
        size = root.find("size")
        if size is None:
            raise ValueError(f"Missing <size> in {xml_path}")

        width = int(float(size.findtext("width")))
        height = int(float(size.findtext("height")))

        objs: List[VocObject] = []
        for obj in root.findall("object"):
            cls = (obj.findtext("name") or "").strip()
            if not cls:
                continue
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            xmin = float(bnd.findtext("xmin"))
            ymin = float(bnd.findtext("ymin"))
            xmax = float(bnd.findtext("xmax"))
            ymax = float(bnd.findtext("ymax"))
            objs.append(VocObject(cls, xmin, ymin, xmax, ymax))

        ann = VocAnnotation(filename, width, height, tuple(objs))
        self._cache[key] = ann
        return ann
