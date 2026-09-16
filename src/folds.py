"""Leak-aware cross-validation folds for SlotKnee-S (spec §5).

WHY GROUPED FOLDS
-----------------
Two kinds of rows must never straddle a train/validation split:

* **Identical reports.**  train.csv (4,407 rows) holds 54 report texts that appear more
  than once (204 rows; the largest template -- a Turkish "everything normal" -- 37 times).
  A model can memorise the template's label vector from the training half and "predict"
  the validation half.
* **The same scanner.**  Site-specific intensity and geometry are the cheapest shortcut a
  CNN/ViT learns; holding out whole scanners gives an honest estimate of generalisation to
  the hidden test set and stops the OOF score from flattering scanner-specific features.

The scanner fingerprint is read from ONE DICOM header per study, ``stop_before_pixels``
and ``specific_tags``, so it costs ~0.4 ms/study (649 studies in 0.3 s here).

Tag note: ``ImagingFrequency`` is the centre frequency measured at each exam, and the same
magnet drifts by tens of Hz between exams (63.685259 / 63.685226 / 63.685214 MHz are one
Avanto).  Taken verbatim it makes 580 distinct fingerprints out of 649 studies -- i.e. no
grouping at all -- so it is rounded to ``freq_decimals`` (default 3 = kHz) before joining.

Public API::

    scanner_fingerprint(study_dir, freq_decimals=3) -> str
    report_group(report_text, row_id=None) -> str
    build_groups(df, data_dir, image_dir="train_images") -> pd.Series
    assign_grouped_folds(df, n_folds=5, seed=42, group_col="group",
                         strat_col=None, keep_col="fold") -> np.ndarray
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
import warnings
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ID_COL = "StudyInstanceUID"
FINGERPRINT_TAGS: List[str] = [
    "Manufacturer", "ManufacturerModelName", "SoftwareVersions",
    "MagneticFieldStrength", "ImagingFrequency", "ReceiveCoilName",
]
UNKNOWN = "unknown"

_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Scanner fingerprint
# ---------------------------------------------------------------------------

def _first_dicom(study_dir: str) -> Optional[str]:
    """Deterministic choice of one file: first series (sorted), first file (sorted)."""
    if not os.path.isdir(study_dir):
        return None
    entries = sorted(e for e in os.listdir(study_dir) if not e.startswith("."))
    # Study dirs hold series dirs; tolerate a flat dir of files too.
    for name in entries:
        p = os.path.join(study_dir, name)
        if os.path.isdir(p):
            files = sorted(f for f in os.listdir(p) if not f.startswith("."))
            dcm = [f for f in files if f.lower().endswith(".dcm")] or files
            if dcm:
                return os.path.join(p, dcm[0])
    files = [e for e in entries if os.path.isfile(os.path.join(study_dir, e))]
    dcm = [f for f in files if f.lower().endswith(".dcm")] or files
    return os.path.join(study_dir, dcm[0]) if dcm else None


def _fmt(tag: str, value, freq_decimals: Optional[int]) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "MultiValue":
        parts = [_fmt(tag, v, freq_decimals) for v in value]
        return "/".join(p for p in parts if p)
    if tag in ("ImagingFrequency", "MagneticFieldStrength"):
        try:
            f = float(value)
        except (TypeError, ValueError):
            return _WS.sub(" ", str(value)).strip()
        if tag == "MagneticFieldStrength":
            return "%g" % f                                   # '1.5', '3'
        if freq_decimals is None:
            return _WS.sub(" ", str(value)).strip()           # verbatim
        return ("%." + str(int(freq_decimals)) + "f") % f
    return _WS.sub(" ", str(value)).strip()


@lru_cache(maxsize=None)
def scanner_fingerprint(study_dir: str, freq_decimals: Optional[int] = 3) -> str:
    """``Manufacturer|Model|Software|FieldStrength|Frequency|Coil`` from ONE header.

    Header-only read (``stop_before_pixels``, ``specific_tags``) of the first file of
    the first series.  Tags that are absent are skipped; ``"unknown"`` when none is
    present, the directory does not exist, or the file is not readable as DICOM.
    ``freq_decimals`` rounds ImagingFrequency (MHz); ``None`` keeps it verbatim.
    Results are memoised per (path, freq_decimals).
    """
    path = _first_dicom(str(study_dir))
    if path is None:
        return UNKNOWN
    try:
        import pydicom
        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=FINGERPRINT_TAGS)
    except Exception:
        return UNKNOWN
    parts = []
    for tag in FINGERPRINT_TAGS:
        try:
            s = _fmt(tag, ds.get(tag, None), freq_decimals)
        except Exception:
            s = ""
        if s:
            parts.append(s)
    return "|".join(parts) if parts else UNKNOWN


# ---------------------------------------------------------------------------
# Report hash
# ---------------------------------------------------------------------------

def normalise_report(text) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    return _WS.sub(" ", str(text).lower()).strip()


def report_group(report_text, row_id=None) -> str:
    """sha1 of the lower-cased, whitespace-collapsed report.

    An empty report cannot be "the same report" as any other, so it gets a per-row
    unique id: ``empty:<row_id>`` when ``row_id`` is given, else a random one.
    """
    norm = normalise_report(report_text)
    if not norm:
        return f"empty:{row_id}" if row_id is not None else f"empty:{uuid.uuid4().hex}"
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------

def _attach_reports(df: pd.DataFrame, data_dir: str, text_col: str) -> pd.Series:
    """Report text per row; pulled from <data_dir>/train.csv when df lacks it."""
    if text_col in df.columns:
        return df[text_col]
    path = os.path.join(str(data_dir), "train.csv")
    if ID_COL in df.columns and os.path.exists(path):
        rep = pd.read_csv(path, engine="python", usecols=[ID_COL, text_col])
        rep[ID_COL] = rep[ID_COL].astype(str).str.strip()
        m = rep.drop_duplicates(ID_COL).set_index(ID_COL)[text_col]
        return df[ID_COL].astype(str).str.strip().map(m)
    return pd.Series([""] * len(df), index=df.index, dtype=object)


def build_groups(df: pd.DataFrame, data_dir: str, image_dir: str = "train_images",
                 text_col: str = "Report", freq_decimals: Optional[int] = 3) -> pd.Series:
    """Group id per row, combining scanner fingerprint and report hash.

    Rule: rows that share a report text share a group (``r:<sha1>``) regardless of
    scanner; otherwise the group is the scanner fingerprint (``s:<fingerprint>``);
    a study with no readable DICOM under ``<data_dir>/<image_dir>/<uid>`` falls back
    to its own report hash (so the 3,758 off-disk rows do not collapse into one
    "unknown" group).  If ``df`` has no ``Report`` column the texts are looked up in
    ``<data_dir>/train.csv``.  Returns a Series named ``group`` aligned with ``df``.
    """
    ids = df[ID_COL].astype(str).str.strip() if ID_COL in df.columns \
        else pd.Series([str(i) for i in df.index], index=df.index)
    texts = _attach_reports(df, data_dir, text_col)
    hashes = [report_group(t, rid) for t, rid in zip(texts.tolist(), ids.tolist())]
    counts = Counter(hashes)
    root = os.path.join(str(data_dir), image_dir)
    groups = []
    for uid, h in zip(ids.tolist(), hashes):
        if counts[h] > 1:
            groups.append("r:" + h)
            continue
        fp = scanner_fingerprint(os.path.join(root, uid), freq_decimals) \
            if os.path.isdir(os.path.join(root, uid)) else UNKNOWN
        groups.append("s:" + fp if fp != UNKNOWN else "r:" + h)
    return pd.Series(groups, index=df.index, name="group")


# ---------------------------------------------------------------------------
# Folds
# ---------------------------------------------------------------------------

class FoldIntegrityError(ValueError):
    """The grouping or the kept folds are BROKEN -- as opposed to the dataset merely
    being too small to split.

    Callers must never quietly substitute round-robin folds for this: ungrouped folds let
    scanner identity straddle the split and inflate CV by ~0.053, so the run would look
    BETTER than the truth.  A dataset that is simply too small raises plain ValueError
    (or sklearn does), which a caller may legitimately absorb.
    """


def positive_bucket(df: pd.DataFrame) -> np.ndarray:
    """0 / 1-2 / 3-4 / 5+ positives (y > 0.5) per row -> 0..3."""
    ycols = [c for c in df.columns if str(c).startswith("y_")]
    if not ycols:
        from src.llm_labels import LABELS
        ycols = [c for c in LABELS if c in df.columns]
    if not ycols:
        return np.zeros(len(df), dtype=int)
    y = df[ycols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = (np.nan_to_num(y, nan=0.0) > 0.5).sum(axis=1)
    return np.digitize(n, [1, 3, 5]).astype(int)          # 0 | 1-2 | 3-4 | 5+


def assign_grouped_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42,
                         group_col: str = "group", strat_col: Optional[str] = None,
                         keep_col: Optional[str] = "fold") -> np.ndarray:
    """Deterministic StratifiedGroupKFold fold id per row (int array, len(df)).

    * Groups come from ``df[group_col]`` (build them with ``build_groups``).
    * Stratification: ``df[strat_col]`` if given, else ``positive_bucket`` over the
      ``y_`` columns.
    * Rows whose ``df[keep_col]`` is not NaN (the 58 gold rows) keep that fold.  Their
      group-mates inherit it so a group never straddles folds; everything else is
      split by ``StratifiedGroupKFold(shuffle=True, random_state=seed)``.
    """
    n = len(df)
    if n == 0:
        return np.zeros(0, dtype=int)
    if group_col not in df.columns:
        raise FoldIntegrityError(f"assign_grouped_folds: no '{group_col}' column; call build_groups first")
    groups = df[group_col].astype(str).to_numpy()
    strat = (pd.factorize(df[strat_col].astype(str))[0] if strat_col is not None
             else positive_bucket(df))

    fold = np.full(n, -1, dtype=int)
    if keep_col is not None and keep_col in df.columns:
        kept = pd.to_numeric(df[keep_col], errors="coerce").to_numpy(dtype=float)
        has = np.isfinite(kept)
        if has.any():
            k = kept[has].astype(int)
            if k.min() < 0 or k.max() >= n_folds:
                raise FoldIntegrityError(f"kept folds {sorted(set(k))} fall outside 0..{n_folds - 1}")
            fold[has] = k
            # Group-mates of a kept row inherit its fold (majority if they disagree).
            by_group: Dict[str, Counter] = {}
            for g, f in zip(groups[has], k):
                by_group.setdefault(g, Counter())[f] += 1
            for i in np.flatnonzero(~has):
                c = by_group.get(groups[i])
                if c:
                    fold[i] = c.most_common(1)[0][0]

    todo = np.flatnonzero(fold < 0)
    if todo.size == 0:
        return fold
    g_sub = groups[todo]
    n_groups = len(set(g_sub))
    if n_groups < n_folds:
        raise ValueError(f"only {n_groups} groups left to split into {n_folds} folds")
    y_sub = strat[todo]
    from sklearn.model_selection import StratifiedGroupKFold
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    X = np.zeros((todo.size, 1))
    with warnings.catch_warnings():
        # The "5+ positives" bucket can hold fewer rows than folds; sklearn still
        # splits, it just warns.  Expected, not actionable.
        warnings.filterwarnings("ignore", message="The least populated class")
        for k, (_, test_idx) in enumerate(splitter.split(X, y_sub, g_sub)):
            fold[todo[test_idx]] = k
    assert (fold >= 0).all()
    return fold


__all__ = [
    "FINGERPRINT_TAGS", "UNKNOWN", "scanner_fingerprint", "normalise_report",
    "report_group", "build_groups", "positive_bucket", "assign_grouped_folds",
]
