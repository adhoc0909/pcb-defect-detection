from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd

def results_to_dataframe(results) -> pd.DataFrame:
    """Convert Ultralytics validation Results to a per-class table if available.

    Ultralytics versions differ; we try multiple known attributes.
    """
    r = results
    # Try common fields: r.box or r.metrics
    # In many versions, r.results_dict contains overall metrics only.
    # The textual table shown in console comes from validator printing. We can't always recover it perfectly.
    rows: List[Dict[str, Any]] = []

    # Prefer a stored per-class list if present
    for attr in ["ap_class_index", "ap50", "ap", "p", "r"]:
        if hasattr(r, attr):
            break

    # If class-wise AP exists (ap50/ap), we can compile a table.
    if hasattr(r, "ap") and hasattr(r, "ap50") and hasattr(r, "names"):
        names = r.names if isinstance(r.names, dict) else {i:n for i,n in enumerate(r.names)}
        ap = getattr(r, "ap", None)
        ap50 = getattr(r, "ap50", None)
        p = getattr(r, "p", None)
        rec = getattr(r, "r", None)

        if ap is not None and ap50 is not None:
            for i, cls_name in names.items():
                row = {"class": cls_name}
                try:
                    row["P"] = float(p[i]) if p is not None else None
                    row["R"] = float(rec[i]) if rec is not None else None
                    row["mAP50"] = float(ap50[i])
                    row["mAP50-95"] = float(ap[i])
                except Exception:
                    continue
                rows.append(row)

    return pd.DataFrame(rows)
