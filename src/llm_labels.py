"""LLM report labels -> per-study soft targets and cell weights (SlotKnee-S, §5).

WHY THIS MODULE EXISTS
----------------------
The labels are the ceiling, not the encoder (docs/research_pivot_efficient_model.md §2).
Against the 58 gold studies the regex parser in ``src/labels.py`` reaches 0.814 macro AUC,
a single LLM reader 0.887 and a blend of two independent LLM readers 0.893.  The public
CC0 LLM label files are Kaggle datasets (``stevenleehans/rsna-knee-llm-report-labels``,
Pilkwang Kim's ``rsna-knee-llm-labels``, the ``lixin73`` GPT labels) and are NOT on this
machine, so everything here has to work with zero, one or several of them present and
fall back to the regex parser otherwise -- by *calling* ``src.labels``, never by
re-implementing it.

THE CONTRACT (what the trainer consumes)
----------------------------------------
One row per study::

    StudyInstanceUID, y_<label> x12, w_<label> x12, is_gold[, fold]

``y_`` is a soft target in [0, 1]; ``w_`` is a per-cell loss weight.  An LLM cell equal to
0.5 means "the report does not address this finding", and ``cell_weights`` maps it to a
weight of exactly 0 so it costs the loss nothing (25.4 % of cells).  Gold rows override
the LLM values with hard 0/1 and carry ``gold_weight`` on every cell.

Recommended order for a caller::

    llm = load_llm_labels([...csv...])          # or regex_fallback(train_csv, data_dir)
    llm = fill_silent_synovitis(llm)            # only meaningful for LLM-style 0.5 cells
    tgt = build_targets(llm, gold_df, gold_weight=8.0)

``regex_fallback`` already returns the ``y_/w_`` schema (the parser's calibrated soft
target and its own confidence), and ``build_targets`` accepts either schema.

THE PUBLIC FILES, MEASURED (data_subset/labels_external/, 4,407 studies)
------------------------------------------------------------------------
* ``stevenleehans/llm_labels_full.csv``  raw reader A; 25.4 % of cells are exactly 0.5.
* ``stevenleehans/llm_labels_v2.csv``    = full with silent Synovitis filled from
  Effusion (what ``fill_silent_synovitis`` does); 19.2 % of cells at 0.5.
* ``stevenleehans/llm_labels_v4_blend.csv`` = mean(v2, a second reader that writes hard
  0/1).  Silence therefore lands at 0.25 ("A silent, B absent", 19.1 % of cells) or
  0.75 -- never at 0.5 -- so ``cell_weights`` gives those cells weight 0.5, not 0.
  Best file against the 58 gold studies (macro AUC 0.893).
* ``pilkwang/report_labels_v2.csv``  p in {0.08 NO, 0.28 UNK, 0.68/0.82/0.94 YES} with a
  ``<Label>__conf`` column {0.85, 0.05, 0.95}; 27.3 % UNK.  Load with ``with_conf=True``
  so UNK cells weigh 0.05 instead of 2|0.28-0.5| = 0.44.
"""

from __future__ import annotations

import os
import re
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Spelling matches the competition CSV headers exactly; do not "tidy" these.
LABELS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

ID_COL = "StudyInstanceUID"
SILENT = 0.5          # "the report does not address this finding"

# ---------------------------------------------------------------------------
# Column-name matching
# ---------------------------------------------------------------------------

# Extra spellings seen in the wild, keyed by their folded form (see _fold_name).
# The folded LABELS themselves ("acl", "medialmeniscus", "bakers", "pfoa", ...) are
# always accepted; this table only adds synonyms.
_ALIASES: Dict[str, str] = {
    "baker": "Baker's",
    "bakerscyst": "Baker's",
    "bakercyst": "Baker's",
    "medialosteoarthritis": "Medial OA",
    "lateralosteoarthritis": "Lateral OA",
    "patellofemoraloa": "PF OA",
    "patellofemoralosteoarthritis": "PF OA",
    "pfosteoarthritis": "PF OA",
    "jointeffusion": "Effusion",
    "bonecontusion": "Contusion",
}
_ID_KEYS = {"studyinstanceuid", "studyuid", "studyid", "study", "uid", "id"}
# Optional prefixes/suffixes a probability column may carry ("p_ACL", "ACL_prob").
_PREFIXES = ("prob", "pred", "llm", "label", "p", "y")
_SUFFIXES = ("prob", "pred", "p")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _fold_name(name: str) -> str:
    """Case/space/apostrophe/underscore-insensitive key for a column name."""
    return _NON_ALNUM.sub("", str(name).lower())


def _label_keys() -> Dict[str, str]:
    keys = {_fold_name(lab): lab for lab in LABELS}
    keys.update(_ALIASES)
    return keys


def match_label_columns(columns: Iterable[str]) -> Dict[str, str]:
    """Map ``{label: column}`` for every LABELS entry found in ``columns``.

    Matching folds case, spaces, underscores, hyphens and apostrophes, accepts the
    aliases above, and strips one optional probability prefix (``p_``, ``prob_``,
    ``pred_``, ``llm_``, ``label_``, ``y_``) or suffix (``_prob``, ``_pred``, ``_p``).
    An exact (folded) match always wins over a prefixed/suffixed one.
    """
    keys = _label_keys()
    exact: Dict[str, str] = {}
    loose: Dict[str, str] = {}
    for col in columns:
        k = _fold_name(col)
        if k in keys:
            exact.setdefault(keys[k], col)
            continue
        for pre in _PREFIXES:
            if k.startswith(pre) and k[len(pre):] in keys:
                loose.setdefault(keys[k[len(pre):]], col)
                break
        else:
            for suf in _SUFFIXES:
                if k.endswith(suf) and k[:-len(suf)] in keys:
                    loose.setdefault(keys[k[:-len(suf)]], col)
                    break
    out = dict(loose)
    out.update(exact)
    return {lab: out[lab] for lab in LABELS if lab in out}


def conf_col(label: str) -> str:
    return label + "__conf"


CONF_COLS: List[str] = [conf_col(lab) for lab in LABELS]


def match_conf_columns(columns: Iterable[str], label_cols: Dict[str, str]) -> Dict[str, str]:
    """``{label: confidence column}`` for files that ship one (``<Label>__conf``).

    Pilkwang's ``report_labels_v2.csv`` and this repo's ``src.labels`` both write a
    per-cell confidence next to the probability; when present it is a better loss
    weight than ``2|p-0.5|`` (their UNK cells sit at p=0.28, conf=0.05).
    """
    by_key = {}
    for col in columns:
        by_key.setdefault(_fold_name(col), col)
    out = {}
    for lab, col in label_cols.items():
        for cand in (_fold_name(col) + "conf", _fold_name(lab) + "conf",
                     _fold_name(col) + "confidence", _fold_name(lab) + "confidence"):
            if cand in by_key and by_key[cand] != col:
                out[lab] = by_key[cand]
                break
    return out


def match_id_column(columns: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    if ID_COL in cols:
        return ID_COL
    for col in cols:
        if _fold_name(col) in _ID_KEYS:
            return col
    return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _coerce_probs(values: pd.Series) -> pd.Series:
    """Numeric probabilities in [0, 1]; anything unreadable becomes SILENT (0.5).

    Tolerances: a ``-1`` sentinel ("not addressed" in some label files) becomes 0.5
    rather than being clipped to a confident 0; a column that is clearly in percent
    (max > 1.5, min >= 0, max <= 100) is divided by 100.
    """
    v = pd.to_numeric(values, errors="coerce").astype(float)
    v = v.mask(np.isclose(v, -1.0), np.nan)
    finite = v[np.isfinite(v)]
    if len(finite) and finite.max() > 1.5 and finite.min() >= 0 and finite.max() <= 100:
        v = v / 100.0
    return v.clip(0.0, 1.0)


def _read_one(path_or_df: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """One label file -> frame indexed by StudyInstanceUID with LABELS columns (NaN = absent)."""
    if isinstance(path_or_df, pd.DataFrame):
        raw = path_or_df
        name = "<DataFrame>"
    else:
        raw = pd.read_csv(path_or_df)
        name = os.path.basename(str(path_or_df))
    id_col = match_id_column(raw.columns)
    if id_col is None:
        raise ValueError(f"{name}: no StudyInstanceUID-like column in {list(raw.columns)[:12]}")
    cols = match_label_columns(raw.columns)
    if not cols:
        raise ValueError(f"{name}: none of the {len(LABELS)} labels found in "
                         f"{list(raw.columns)[:20]}")
    confs = match_conf_columns(raw.columns, cols)
    out = pd.DataFrame(index=raw.index)
    out[ID_COL] = raw[id_col].astype(str).str.strip()
    for lab in LABELS:
        out[lab] = _coerce_probs(raw[cols[lab]]) if lab in cols else np.nan
    for lab in LABELS:
        if lab in confs:
            c = pd.to_numeric(raw[confs[lab]], errors="coerce").astype(float).clip(0.0, 1.0)
        else:
            # No explicit confidence: the distance from "not addressed" stands in.
            c = pd.Series(cell_weights(out[lab].to_numpy()), index=raw.index)
        out[conf_col(lab)] = c.where(np.isfinite(out[lab]), np.nan)
    # A study listed twice in one file is one opinion, averaged.
    out = out.groupby(ID_COL, sort=False)[LABELS + CONF_COLS].mean()
    return out


def _blend(stack: np.ndarray, blend: str) -> np.ndarray:
    """NaN-skipping blend over the file axis of ``stack`` [F, N, L]."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN cells -> NaN, handled by caller
        if blend == "mean":
            return np.nanmean(stack, axis=0)
        if blend == "median":
            return np.nanmedian(stack, axis=0)
        if blend == "max":
            return np.nanmax(stack, axis=0)
        if blend == "min":
            return np.nanmin(stack, axis=0)
        if blend == "first":
            vals = np.full(stack.shape[1:], np.nan)
            for f in range(stack.shape[0]):
                fill = np.isnan(vals) & ~np.isnan(stack[f])
                vals[fill] = stack[f][fill]
            return vals
    raise ValueError(f"unknown blend={blend!r}; use mean|median|max|min|first")


def load_llm_labels(paths: Sequence[Union[str, pd.DataFrame]], blend: str = "mean",
                    with_conf: bool = False) -> pd.DataFrame:
    """Load one or more LLM label CSVs and blend them per cell.

    Returns ``StudyInstanceUID`` + the 12 LABELS columns, every value in [0, 1].
    Column names are matched tolerantly (see ``match_label_columns``); extra columns
    such as ``<Label>__conf`` / ``<Label>__verdict`` are ignored for the probabilities.
    A label a file does not carry contributes nothing to the blend for that file; a
    cell no file carries is 0.5 ("not addressed").  ``blend`` is one of ``mean``
    (default), ``median``, ``max``, ``min`` or ``first`` (first file with a value wins).

    ``with_conf=True`` appends 12 ``<Label>__conf`` columns: the file's own
    confidence where it ships one, else ``2|p-0.5|``, blended the same way.
    ``build_targets`` uses them as the cell weights when present.
    """
    if isinstance(paths, (str, os.PathLike, pd.DataFrame)):
        paths = [paths]
    paths = list(paths)
    if not paths:
        raise ValueError("load_llm_labels: no label files given")
    frames = [_read_one(p) for p in paths]
    index = frames[0].index
    for f in frames[1:]:
        index = index.append(f.index[~f.index.isin(index)])
    blend = str(blend).lower()
    stack = np.stack([f.reindex(index)[LABELS].to_numpy(dtype=float) for f in frames])  # [F, N, 12]
    vals = _blend(stack, blend)
    vals = np.where(np.isfinite(vals), vals, SILENT).clip(0.0, 1.0)

    out = pd.DataFrame(vals, columns=LABELS)
    out.insert(0, ID_COL, index.astype(str).to_numpy())
    if with_conf:
        cstack = np.stack([f.reindex(index)[CONF_COLS].to_numpy(dtype=float) for f in frames])
        conf = _blend(cstack, blend)
        conf = np.where(np.isfinite(conf), conf, 0.0).clip(0.0, 1.0)
        for j, lab in enumerate(LABELS):
            out[conf_col(lab)] = conf[:, j]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Weights and fills
# ---------------------------------------------------------------------------

def cell_weights(p) -> np.ndarray:
    """2*|p - 0.5|: a decisive 0/1 weighs 1, a silent 0.5 weighs exactly 0."""
    p = np.asarray(p, dtype=np.float64)
    return np.clip(2.0 * np.abs(p - SILENT), 0.0, 1.0)


def fill_silent_synovitis(df: pd.DataFrame) -> pd.DataFrame:
    """Where Synovitis is silent (0.5) but Effusion is not, borrow half of Effusion's signal.

    Silence means *absent* for Baker's (3 % positive when silent) but *unknown* for
    Synovitis (34 %); the LLM v2 labels fill it from Effusion, which this reproduces:
    ``Synovitis = 0.5 + 0.5 * (Effusion - 0.5)``.  Returns a copy.
    """
    out = df.copy()
    if "Synovitis" not in out.columns or "Effusion" not in out.columns:
        return out
    syn = pd.to_numeric(out["Synovitis"], errors="coerce").astype(float)
    eff = pd.to_numeric(out["Effusion"], errors="coerce").astype(float)
    silent = np.isclose(syn, SILENT, atol=1e-6) & ~np.isclose(eff, SILENT, atol=1e-6) & np.isfinite(eff)
    out.loc[silent, "Synovitis"] = (SILENT + 0.5 * (eff[silent] - SILENT)).clip(0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

def y_col(label: str) -> str:
    return "y_" + label


def w_col(label: str) -> str:
    return "w_" + label


Y_COLS: List[str] = [y_col(lab) for lab in LABELS]
W_COLS: List[str] = [w_col(lab) for lab in LABELS]


def _as_yw(df: pd.DataFrame, min_weight: float) -> pd.DataFrame:
    """Accept either the prob schema (LABELS columns) or the y_/w_ schema."""
    id_col = match_id_column(df.columns)
    if id_col is None:
        raise ValueError("build_targets: frame has no StudyInstanceUID column")
    out = pd.DataFrame({ID_COL: df[id_col].astype(str).str.strip().to_numpy()})
    if all(c in df.columns for c in Y_COLS):
        y = df[Y_COLS].to_numpy(dtype=float)
        y = np.where(np.isfinite(y), y, SILENT).clip(0.0, 1.0)
        if all(c in df.columns for c in W_COLS):
            w = np.nan_to_num(df[W_COLS].to_numpy(dtype=float), nan=0.0).clip(0.0, 1.0)
        else:
            w = cell_weights(y)
    else:
        cols = match_label_columns(df.columns)
        missing = [lab for lab in LABELS if lab not in cols]
        if missing:
            raise ValueError(f"build_targets: label columns missing: {missing}")
        y = np.stack([_coerce_probs(df[cols[lab]]).to_numpy() for lab in LABELS], axis=1)
        y = np.where(np.isfinite(y), y, SILENT)
        confs = match_conf_columns(df.columns, cols)
        if len(confs) == len(LABELS):
            w = np.stack([pd.to_numeric(df[confs[lab]], errors="coerce").to_numpy(dtype=float)
                          for lab in LABELS], axis=1)
            w = np.nan_to_num(w, nan=0.0).clip(0.0, 1.0)
        else:
            w = cell_weights(y)
    if min_weight > 0:
        w = np.where(w >= min_weight, w, 0.0)
    for j, lab in enumerate(LABELS):
        out[y_col(lab)] = y[:, j]
    for j, lab in enumerate(LABELS):
        out[w_col(lab)] = w[:, j]
    return out


def build_targets(llm_df: Optional[pd.DataFrame], gold_df: Optional[pd.DataFrame] = None,
                  gold_weight: float = 8.0, min_weight: float = 0.0) -> pd.DataFrame:
    """One row per study: ``StudyInstanceUID, y_<label>, w_<label>, is_gold[, fold]``.

    ``llm_df`` is the output of ``load_llm_labels`` (prob columns) or of
    ``regex_fallback`` (``y_/w_`` columns).  Weights for prob input are
    ``cell_weights``; cells with weight below ``min_weight`` are zeroed (a gate, not a
    floor).  Gold rows override every LLM value with a hard 0/1 and weight
    ``gold_weight`` on every cell; gold studies absent from ``llm_df`` are appended.
    If ``gold_df`` has a ``fold`` column it is carried through (NaN for non-gold rows)
    so ``folds.assign_grouped_folds`` can keep it.
    """
    if llm_df is None or len(llm_df) == 0:
        base = pd.DataFrame({ID_COL: pd.Series(dtype=str)})
        for c in Y_COLS:
            base[c] = pd.Series(dtype=float)
        for c in W_COLS:
            base[c] = pd.Series(dtype=float)
    else:
        base = _as_yw(llm_df, min_weight)
    base = base.drop_duplicates(ID_COL, keep="first").reset_index(drop=True)
    base["is_gold"] = False

    has_fold = gold_df is not None and "fold" in gold_df.columns
    if has_fold:
        base["fold"] = np.nan

    if gold_df is not None and len(gold_df):
        gid = match_id_column(gold_df.columns)
        gcols = match_label_columns(gold_df.columns)
        missing = [lab for lab in LABELS if lab not in gcols]
        if gid is None or missing:
            raise ValueError(f"build_targets: gold frame lacks StudyInstanceUID or labels {missing}")
        g = pd.DataFrame({ID_COL: gold_df[gid].astype(str).str.strip().to_numpy()})
        gy = np.stack([pd.to_numeric(gold_df[gcols[lab]], errors="coerce").to_numpy(dtype=float)
                       for lab in LABELS], axis=1)
        known = np.isfinite(gy)
        hard = (np.nan_to_num(gy, nan=0.0) >= 0.5).astype(float)
        for j, lab in enumerate(LABELS):
            g[y_col(lab)] = hard[:, j]
        for j, lab in enumerate(LABELS):
            # An unlabelled gold cell (NaN) is not a hard label: weight 0.
            g[w_col(lab)] = np.where(known[:, j], float(gold_weight), 0.0)
        g["is_gold"] = True
        if has_fold:
            g["fold"] = pd.to_numeric(gold_df["fold"], errors="coerce").to_numpy(dtype=float)
        g = g.drop_duplicates(ID_COL, keep="first")

        gold_ids = set(g[ID_COL])
        base = pd.concat([base[~base[ID_COL].isin(gold_ids)], g[base.columns]],
                         ignore_index=True)

    base["is_gold"] = base["is_gold"].astype(bool)
    for c in Y_COLS + W_COLS:
        base[c] = base[c].astype(np.float32)
    return base.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Regex fallback (calls src.labels; never re-implements it)
# ---------------------------------------------------------------------------

def _on_disk_studies(data_dir: str, image_dir: str) -> Optional[set]:
    for candidate in (image_dir, "train_series"):
        d = os.path.join(data_dir, candidate)
        if os.path.isdir(d):
            return {s for s in os.listdir(d) if not s.startswith(".")}
    return None


def regex_fallback(train_csv: Union[str, pd.DataFrame], data_dir: str,
                   images_only: bool = True, image_dir: str = "train_images",
                   gold_df: Optional[pd.DataFrame] = None,
                   calibrate_on_folds: Optional[Sequence[int]] = None,
                   text_col: str = "Report", limit: int = 0) -> pd.DataFrame:
    """Regex soft targets in the ``y_/w_`` schema, via ``src.labels.label_dataframe``.

    ``y_`` is the parser's calibrated soft target (an unmentioned finding gets the
    per-label prior, never 0) and ``w_`` its confidence (0 for unmentioned), which is
    exactly what ``train.derived_cell_weights`` computes with ``labels_weight=1`` and
    no confidence gate.  ``images_only`` keeps the studies that have pixels under
    ``<data_dir>/<image_dir>`` (the 649 local ones).  ``calibrate_on_folds`` refits the
    parser's priors/state table on those gold folds only (needs ``gold_df``), mirroring
    ``train.build_derived_from_reports``.
    """
    import src.labels as L

    if isinstance(train_csv, pd.DataFrame):
        frame = train_csv
    else:
        path = train_csv if os.path.isabs(train_csv) or os.path.exists(train_csv) \
            else os.path.join(data_dir, train_csv)
        frame = pd.read_csv(path, engine="python")
    if ID_COL not in frame.columns or text_col not in frame.columns:
        raise ValueError(f"regex_fallback: need '{ID_COL}' and '{text_col}' columns")
    frame = frame[[ID_COL, text_col]].copy()
    frame[ID_COL] = frame[ID_COL].astype(str).str.strip()
    frame = frame[frame[text_col].fillna("").astype(str).str.strip() != ""]

    if images_only:
        on_disk = _on_disk_studies(data_dir, image_dir)
        if on_disk is not None:
            frame = frame[frame[ID_COL].isin(on_disk)]
    if limit and limit > 0:
        frame = frame.head(int(limit))
    frame = frame.drop_duplicates(ID_COL, keep="first").reset_index(drop=True)

    priors = calib = None
    if calibrate_on_folds is not None and gold_df is not None:
        keep = [int(x) for x in calibrate_on_folds]
        priors = L.calibrate_priors(gold_df, labels=LABELS, folds=keep)
        calib = L.calibrate_states(gold_df, labels=LABELS, folds=keep, priors=priors)

    derived = L.label_dataframe(frame, text_col=text_col, labels=LABELS,
                                priors=priors, calibration=calib)
    out = pd.DataFrame({ID_COL: frame[ID_COL].to_numpy()})
    for lab in LABELS:
        out[y_col(lab)] = pd.to_numeric(derived[lab], errors="coerce").fillna(SILENT).clip(0, 1).to_numpy(dtype=np.float32)
    for lab in LABELS:
        out[w_col(lab)] = pd.to_numeric(derived[lab + "__conf"], errors="coerce").fillna(0.0).clip(0, 1).to_numpy(dtype=np.float32)
    out["is_gold"] = False
    if "report_lang" in derived.columns:
        out["report_lang"] = derived["report_lang"].to_numpy()
    return out


__all__ = [
    "LABELS", "ID_COL", "SILENT", "Y_COLS", "W_COLS", "CONF_COLS", "y_col", "w_col", "conf_col",
    "match_label_columns", "match_conf_columns", "match_id_column", "load_llm_labels", "cell_weights",
    "fill_silent_synovitis", "build_targets", "regex_fallback",
]
