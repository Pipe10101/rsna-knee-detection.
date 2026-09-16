"""Sequence-slot study tensors for SlotKnee-S (docs/slotknee_spec.md §3).

A study becomes a fixed ``[6, G, T, P, P]`` uint8 array plus a ``[6]`` presence
mask.  The six slots are (plane × contrast) sequence families; inside a slot
``G`` anchors are spread over the geometry-ordered stack and ``T`` physically
adjacent slices are taken around each anchor (T=3 → an RGB-like triplet).

Pipeline per study
------------------
1. ``index_study``      header-only pass (``stop_before_pixels`` + ``specific_tags``):
                        one survey header per series (plane, contrast, spacing,
                        laterality, centre-x), slot assignment from the series CSV
                        (header fallback), then ``kaggle_data.order_series`` on the
                        chosen series only.
2. ``select_slices``    G anchors via ``np.linspace`` over the trimmed stack, T
                        adjacent slices ``[a-1, a, a+1]`` (clamped) per anchor.
3. ``build_study_tensor``
                        pixel-decode ONLY the selected files, centre-crop a fixed
                        ``crop_mm`` window from ``PixelSpacing``, ``cv2.INTER_AREA``
                        resize to ``P×P``, per-slot 1–99 percentile → uint8,
                        laterality normalisation (right knee → mirrored).

Everything here is Python-3.9 compatible and never touches the network.

Deviation from the spec text (§3.1), with evidence: in ``train_series.csv``
``Fluid_Sensitive == Fat_Suppression`` on 24,371 / 24,371 rows, i.e. the CSV's
"fluid-sensitive" means *fat-suppressed*.  The header fallback therefore maps
fat-sat / STIR / SPIR / SPAIR / TIRM tokens to fluid-sensitive and everything
else (including non-fat-sat T2 / PD) to the ``*_T1`` group — 98.0 % agreement
with the CSV on the 3,626 local series (plane via ``derive_plane``: 100 %)
versus 82 % for a literal "T2/PD → fluid-sensitive" rule.  The TR/TE weighting (T1 / PD / T2) is still derived and
stored on each ``SeriesInfo`` as ``weighting`` for inspection.

Review-driven deviations (docs/slotknee_review.md §3):
A1  slot series selection prefers routine 2-D stacks — slice count 12-60
    first, then SliceThickness >= 2 mm when the header carries it, then most
    slices, tie -> smallest UID — instead of raw "most slices", which picked
    160-slice 3-D acquisitions (VIBE/SPACE) over the 2-D TSE a radiologist
    reads.
A2  every resize goes through ``resize_to_square`` (integer-factor INTER_AREA
    shrink, then INTER_LINEAR; single-step INTER_AREA for sub-2x downscales),
    so cache building and inference stay bit-identical by construction.
A3  the 1-99 percentile is estimated on every 16th pixel.
§2.4  all header reads (survey AND geometry ordering) parse the first 8 KB of
    the file and fall back to a full-file read per file on any exception or
    missing tag; the ordering mirrors ``kaggle_data.order_series`` exactly
    (asserted in tests).
Cache files carry version "slots-v2" (pixel values differ from v1).

Zoom slots (optional, default off; the 6-slot layout is unchanged): each name
in ``zoom_slots`` (a base slot, e.g. "SAG_FS") appends a slot ``<BASE>_Z`` that
re-uses the base slot's files and anchors but crops ``zoom_mm`` (100 mm ->
0.45 mm/px at P=224, Nyquist-safe for 1-mm tears) from the SAME decoded array
— no second decode — with its own 1–99 percentile and the base slot's mask.
The index records ``slot_names``; ``SlotCache.S`` / ``.slot_names`` expose the
layout (old caches read as S=6).  Size at P=224, G=T=3: 6 slots 11.9 GB, 8
slots 4,407×8×9×224² = 15.9 GB (inside Kaggle's 20 GB output cap).
``anchor_shift`` (default 0) moves every anchor by that many slices, clamped —
inference-time TTA, never used for the training cache.

Zoom SPEC (Phase B, ACL/notch view): ``zoom_spec`` generalises the
``zoom_mm``/``zoom_slots``/``zoom_center`` trio to per-entry sizes and
centres: ``"SAG_FS:100:joint,SAG_FS:80:joint"`` appends ``SAG_FS_Z100J`` and
``SAG_FS_Z80J`` (name = ``<BASE>_Z<mm>``, ``J`` = joint-centred).  Legacy trio
entries keep their historical ``<BASE>_Z`` names and come first, so existing
caches and call sites stay bit-identical; spec entries follow in the given
order.  All crops of a base slot still come from ONE decode of each slice,
and the joint is located ONCE per base slot and shared by its entries.  The
ACL campaign layout is exactly the two SAG entries above -> S = 8 (no COR
entry is implied; list one explicitly if wanted).

Joint-centred zoom (optional, default off): ``zoom_center="joint"`` runs
``locate_joint`` on the central slices of each SAG/COR zoom slot (decoded
anyway) and centres that slot's ``zoom_mm`` crops on the estimated
tibiofemoral joint instead of the image centre; low confidence -> image
centre (counted).  The base 140 mm crops never move.  The cache index records
``zoom_center`` (``SlotCache.zoom_center``; old caches read as "image").
"""

import bisect
import collections
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pydicom

from src.kaggle_data import derive_plane

__all__ = [
    "SLOT_NAMES", "SLOT_NAMES_EXT", "slot_names_ext", "parse_zoom_slots",
    "parse_zoom_spec", "zoom_slot_name", "normalize_zoom", "ZoomEntry", "PLANES", "SAG_SLOTS", "COR_SLOTS", "AX_SLOTS", "N_SLOTS", "CACHE_VERSION",
    "SeriesInfo", "StudyIndex", "index_study", "select_slices", "anchor_indices",
    "build_study_tensor", "apply_laterality", "resolve_side", "fluid_from_header",
    "crop_window", "slot_of", "series_lookup",
    "ZOOM_CENTERS", "JointEstimate", "locate_joint", "combine_joint_estimates",
    "joint_polarity", "locate_joint_for_slot", "joint_slice_paths",
    "series_preference_key", "preferred_series", "resize_to_square",
    "SlotCache", "cache_paths", "shard_filename", "shard_layout",
]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SLOT_NAMES = ["SAG_FS", "COR_FS", "AX_FS", "SAG_T1", "COR_T1", "AX_T1"]
PLANES = ("Sagittal", "Coronal", "Axial")
SAG_SLOTS = (0, 3)
COR_SLOTS = (1, 4)
AX_SLOTS = (2, 5)
N_SLOTS = len(SLOT_NAMES)
CACHE_VERSION = "slots-v2"


def parse_zoom_slots(zoom_slots) -> Tuple[str, ...]:
    """Normalise None / "SAG_FS,COR_FS" / sequence -> tuple of base slot names."""
    if zoom_slots is None:
        return ()
    if isinstance(zoom_slots, str):
        return tuple(v.strip() for v in zoom_slots.split(",") if v.strip())
    return tuple(str(v).strip() for v in zoom_slots)


def slot_names_ext(zoom_slots=(), zoom_spec=None) -> List[str]:
    """The base 6 slot names plus one zoom slot per entry.

    Legacy ``zoom_slots`` entries are named ``<BASE>_Z``; ``zoom_spec``
    entries ("BASE:MM[:CENTER]") are named via ``zoom_slot_name``.
    """
    names = list(SLOT_NAMES)
    seen = set()
    for b in parse_zoom_slots(zoom_slots):
        if b not in SLOT_NAMES:
            raise ValueError("unknown zoom base slot %r (expected one of %s)" % (b, SLOT_NAMES))
        if b in seen:
            raise ValueError("duplicate zoom slot %r" % b)
        seen.add(b)
        names.append(b + "_Z")
    for z in normalize_zoom(zoom_spec=zoom_spec):
        if z.name in names[N_SLOTS:]:
            raise ValueError("duplicate zoom slot %r" % z.name)
        names.append(z.name)
    return names


SLOT_NAMES_EXT = slot_names_ext


def parse_zoom_spec(spec) -> Tuple[Tuple[str, float, str], ...]:
    """Normalise a zoom spec into ``((base, mm, center), ...)``.

    Accepts None, a string like ``"SAG_FS:80:joint,SAG_FS:100"`` (center
    defaults to "image"), a sequence of such strings, or a sequence of
    ``(base, mm[, center])`` tuples.  Full validation (known base/centre,
    positive mm, duplicate names) happens in ``normalize_zoom``.
    """
    if spec is None:
        return ()
    items = ([v.strip() for v in spec.split(",") if v.strip()]
             if isinstance(spec, str) else list(spec))
    out = []
    for it in items:
        if isinstance(it, str):
            parts = [q.strip() for q in it.split(":")]
            if len(parts) not in (2, 3) or not parts[0] or not parts[1]:
                raise ValueError("bad zoom spec entry %r (want BASE:MM[:CENTER])" % it)
            base, mm = parts[0], parts[1]
            center = parts[2] if len(parts) == 3 and parts[2] else "image"
        else:
            seq = tuple(it)
            if len(seq) not in (2, 3):
                raise ValueError("bad zoom spec entry %r (want (base, mm[, center]))" % (it,))
            base, mm = seq[0], seq[1]
            center = seq[2] if len(seq) == 3 else "image"
        try:
            mm_f = float(mm)
        except (TypeError, ValueError):
            raise ValueError("bad zoom mm %r in spec entry %r" % (mm, it))
        out.append((str(base), mm_f, str(center)))
    return tuple(out)


def zoom_slot_name(base: str, mm: float, center: str = "image") -> str:
    """("SAG_FS", 80, "joint") -> "SAG_FS_Z80J"; ("COR_FS", 80, "medial") -> "COR_FS_Z80M"."""
    return "%s_Z%g%s" % (base, float(mm), {"joint": "J", "medial": "M"}.get(center, ""))


_MEDIAL_SLOT_RE = re.compile(r".+_Z\d+(?:\.\d+)?M$")


def is_medial_slot(name: str) -> bool:
    """True for a "medial"-centred zoom slot (``zoom_slot_name`` suffix ``M``, e.g. ``COR_FS_Z80M``).

    Such a crop is anchored on the MEDIAL compartment, so its laterality mirror is not the
    lateral compartment: every mirror pass (``--flip-swap`` in training, ``--flip-tta`` at
    inference, ``rescore_oof.py --flip-tta``) must DROP it (zero pixels, mask 0) instead of
    flipping it, otherwise medial evidence is read under the swapped (lateral) labels.
    """
    return bool(_MEDIAL_SLOT_RE.match(str(name)))


ZoomEntry = collections.namedtuple("ZoomEntry", "base mm center name")


def normalize_zoom(zoom_mm=None, zoom_slots=(), zoom_center="image", zoom_spec=None
                   ) -> Tuple[ZoomEntry, ...]:
    """The one zoom vocabulary: legacy trio + per-entry spec -> ZoomEntry list.

    Legacy entries (``zoom_slots`` x ``zoom_mm``, ``zoom_center`` for all)
    come first with their historical ``<BASE>_Z`` names — existing caches and
    call sites stay bit-identical; ``zoom_spec`` entries follow, named by
    ``zoom_slot_name``.  Raises ValueError on unknown bases/centres,
    non-positive mm, or duplicate slot names.
    """
    entries: List[ZoomEntry] = []
    legacy = parse_zoom_slots(zoom_slots)
    if legacy:
        if zoom_mm is None or float(zoom_mm) <= 0:
            raise ValueError("zoom_slots given without a positive zoom_mm")
        for b in legacy:
            entries.append(ZoomEntry(b, float(zoom_mm), str(zoom_center), b + "_Z"))
    for base, mm, center in parse_zoom_spec(zoom_spec):
        entries.append(ZoomEntry(base, mm, center, zoom_slot_name(base, mm, center)))
    seen = set()
    for e in entries:
        if e.base not in SLOT_NAMES:
            raise ValueError("unknown zoom base slot %r (expected one of %s)" % (e.base, SLOT_NAMES))
        if e.center not in ZOOM_CENTERS:
            raise ValueError("zoom center must be one of %s (got %r)" % (ZOOM_CENTERS, e.center))
        if e.center == "medial" and e.base not in MEDIAL_BASES:
            raise ValueError("medial zoom needs a coronal base %s (got %r)" % (MEDIAL_BASES, e.base))
        if not (e.mm > 0 and math.isfinite(e.mm)):
            raise ValueError("zoom mm must be positive (got %r for %s)" % (e.mm, e.base))
        if e.name in seen:
            raise ValueError("duplicate zoom slot %r" % e.name)
        seen.add(e.name)
    return tuple(entries)

# Header tags parsed in the survey pass (one file per series).  Everything the
# index needs and nothing else; stop_before_pixels keeps pixel data off disk.
_SURVEY_TAGS = [
    "SeriesInstanceUID", "InstanceNumber", "ImagePositionPatient",
    "ImageOrientationPatient", "Rows", "Columns", "PixelSpacing",
    "Laterality", "ImageLaterality", "ScanOptions", "SeriesDescription",
    "RepetitionTime", "EchoTime", "SliceThickness",
]

# Tokens (after splitting on non-alphanumerics) that mean fat suppression.
# 'fse' (fast spin echo) and 'we' (water excitation, a T1 cartilage sequence
# the CSV files under Fluid_Sensitive=0) are deliberately NOT in this set.
_FATSAT_TOKENS = frozenset({
    "fs", "fsat", "fatsat", "stir", "spir", "spair", "tirm", "fatsup", "fatsuppressed",
})

# Patient-x of the image centre closer than this to the midline is ambiguous.
_SIDE_MIN_MM = 20.0


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def slot_of(plane: Optional[str], fluid: Optional[int]) -> Optional[int]:
    """Slot index for (plane, fluid-sensitive flag), or None if either is unknown."""
    if plane is None or fluid is None:
        return None
    try:
        p = PLANES.index(plane)
    except ValueError:
        return None
    return p + (0 if int(fluid) == 1 else 3)


def _norm_plane(value) -> Optional[str]:
    if value is None:
        return None
    try:
        if isinstance(value, float) and math.isnan(value):
            return None
    except TypeError:
        pass
    s = str(value).strip().lower()
    if s.startswith("sag"):
        return "Sagittal"
    if s.startswith("cor"):
        return "Coronal"
    if s.startswith("ax") or s.startswith("tra"):
        return "Axial"
    return None


def _to_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _text(value) -> str:
    """Stringify a DICOM value (MultiValue / str / None) for token matching."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or value.__class__.__name__ == "MultiValue":
        return " ".join(str(v) for v in value)
    return str(value)


def _tokens(text: str) -> set:
    return set(t for t in re.split(r"[^a-z0-9]+", text.lower()) if t)


_FAST_HEADER_BYTES = 8192
_GEOM_TAGS_FAST = ["InstanceNumber", "ImagePositionPatient", "ImageOrientationPatient"]


def _read_header(path, tags, needed=()):
    """(dataset or None, used_fallback) — 8 KB partial parse, full-read fallback.

    Standard Part-10 headers fit in the first kilobytes, so parsing only the
    first ``_FAST_HEADER_BYTES`` avoids touching the rest of the file (the win
    is on cold/network filesystems; measured 0 tag misses on the 4,028-file
    local corpus).  Never trusted blindly: on ANY exception, or when a tag in
    ``needed`` is absent from the partial parse, the whole file is read the
    normal way and the event is counted by the caller via ``used_fallback``.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(_FAST_HEADER_BYTES)
        ds = pydicom.dcmread(io.BytesIO(head), stop_before_pixels=True, specific_tags=tags)
        if all(ds.get(t) is not None for t in needed):
            return ds, False
    except Exception:
        pass
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, specific_tags=tags), True
    except Exception:
        return None, True


def _read_geometry_fast(path):
    """(instance, pos_along_normal, plane, used_fallback).

    Same maths as ``kaggle_data._read_geometry`` (kept in lock-step by
    tests/test_slots.py::test_order_series_fast_matches_kaggle_data), fed by
    ``_read_header`` instead of a full-file read.
    """
    ds, fb = _read_header(path, _GEOM_TAGS_FAST,
                          needed=("ImagePositionPatient", "ImageOrientationPatient"))
    if ds is None:
        return None, None, None, fb
    iop = getattr(ds, "ImageOrientationPatient", None)
    ipp = getattr(ds, "ImagePositionPatient", None)
    inst = getattr(ds, "InstanceNumber", None)
    pos = None
    if iop is not None and ipp is not None and len(iop) == 6 and len(ipp) == 3:
        v = np.asarray(iop, dtype=float)
        n = np.cross(v[:3], v[3:])
        pos = float(np.dot(np.asarray(ipp, dtype=float), n))
    try:
        inst = int(inst)
    except (TypeError, ValueError):
        inst = None
    return inst, pos, derive_plane(iop), fb


def _order_series_fast(files):
    """(ordered_paths, plane, n_header_fallbacks).

    Byte-identical ordering to ``kaggle_data.order_series`` — same keys
    (position along the normal, then InstanceNumber, then filename) and the
    same all-or-nothing rule — via the 8 KB partial header path.
    """
    keyed, plane, n_fb = [], None, 0
    for p in files:
        try:
            inst, pos, pl, fb = _read_geometry_fast(p)
        except Exception:
            inst, pos, pl, fb = None, None, None, True
        n_fb += int(fb)
        if pl and plane is None:
            plane = pl
        keyed.append((pos, inst, p))
    if all(k[0] is not None for k in keyed):
        keyed.sort(key=lambda k: (k[0], k[2]))
    elif all(k[1] is not None for k in keyed):
        keyed.sort(key=lambda k: (k[1], k[2]))
    else:
        keyed.sort(key=lambda k: k[2])
    return [k[2] for k in keyed], plane, n_fb


def fluid_from_header(scan_options, series_description, tr=None, te=None) -> Tuple[int, str]:
    """(fluid_sensitive 0/1, weighting 't1'|'pd'|'t2') from header text and TR/TE.

    Fluid-sensitive == fat-suppressed (see the module docstring).  The
    weighting is informational only.
    """
    toks = _tokens(_text(scan_options) + " " + _text(series_description))
    fluid = 1 if toks & _FATSAT_TOKENS else 0
    tr_f, te_f = _to_float(tr), _to_float(te)
    if toks & {"t1", "t1w"}:
        weighting = "t1"
    elif toks & {"t2", "t2w"}:
        weighting = "t2"
    elif toks & {"pd", "pdw", "dp"}:
        weighting = "pd"
    elif tr_f is not None and te_f is not None and tr_f > 800:
        weighting = "t2" if te_f > 60 else "pd"
    else:
        weighting = "t1"
    return fluid, weighting


def series_lookup(series_df, study_uid: Optional[str] = None) -> Dict[str, Tuple[Optional[str], Optional[int]]]:
    """series_uid -> (plane, fluid) from a series CSV frame (or a ready dict)."""
    if series_df is None:
        return {}
    if isinstance(series_df, dict):
        out = {}
        for k, v in series_df.items():
            if isinstance(v, dict):
                out[str(k)] = (_norm_plane(v.get("plane", v.get("Anatomical_Plane"))),
                               _to_int(v.get("fluid", v.get("Fluid_Sensitive"))))
            else:
                out[str(k)] = (_norm_plane(v[0]), _to_int(v[1]))
        return out
    df = series_df
    if study_uid is not None and "StudyInstanceUID" in df.columns:
        df = df[df["StudyInstanceUID"].astype(str) == str(study_uid)]
    out = {}
    has_plane = "Anatomical_Plane" in df.columns
    has_fluid = "Fluid_Sensitive" in df.columns
    for r in df.itertuples(index=False):
        plane = _norm_plane(getattr(r, "Anatomical_Plane")) if has_plane else None
        fluid = _to_int(getattr(r, "Fluid_Sensitive")) if has_fluid else None
        out[str(getattr(r, "SeriesInstanceUID"))] = (plane, fluid)
    return out


def _list_series(study_dir: str) -> List[Tuple[str, List[str]]]:
    """[(series_uid, [files])] from <study>/<series>/*.dcm (flat dirs tolerated)."""
    out, flat = [], []
    try:
        entries = sorted(os.scandir(study_dir), key=lambda e: e.name)
    except OSError:
        return out
    for e in entries:
        if e.is_dir():
            files = sorted(os.path.join(e.path, f.name) for f in os.scandir(e.path)
                           if f.is_file() and f.name.lower().endswith(".dcm"))
            if files:
                out.append((e.name, files))
        elif e.is_file() and e.name.lower().endswith(".dcm"):
            flat.append(e.path)
    if flat and not out:
        out.append((os.path.basename(os.path.normpath(study_dir)), sorted(flat)))
    return out


def _centre_x(ipp, iop, rows, cols, spacing) -> Optional[float]:
    """Patient-x (LPS) of the image centre: IPP + r·Δc·Nc/2 + d·Δr·Nr/2."""
    if ipp is None or iop is None or rows is None or cols is None or spacing is None:
        return None
    try:
        p = np.asarray(ipp, dtype=float)
        v = np.asarray(iop, dtype=float)
        if p.shape != (3,) or v.shape != (6,):
            return None
        centre = p + v[:3] * spacing[1] * (cols / 2.0) + v[3:] * spacing[0] * (rows / 2.0)
        x = float(centre[0])
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def resolve_side(tags: Sequence[Optional[str]], centre_xs: Sequence[Optional[float]],
                 min_mm: float = _SIDE_MIN_MM) -> Tuple[Optional[str], str, Optional[float]]:
    """(side, source, median_x).  Tag majority first, then patient-x sign."""
    n_l = sum(1 for t in tags if t == "L")
    n_r = sum(1 for t in tags if t == "R")
    xs = [float(x) for x in centre_xs if x is not None and math.isfinite(float(x))]
    med = float(np.median(xs)) if xs else None
    if n_l != n_r and (n_l or n_r):
        return ("L" if n_l > n_r else "R"), "tag", med
    if med is None or abs(med) < min_mm:
        return None, "unresolved", med
    return ("R" if med < 0 else "L"), "geometry", med


def crop_window(rows: int, cols: int, spacing: Optional[Tuple[float, float]],
                crop_mm: float, offset_mm: Optional[Tuple[float, float]] = None
                ) -> Tuple[int, int, int, int, bool]:
    """(r0, r1, c0, c1, fallback) for a centred ``crop_mm`` window.

    px = round(mm / spacing) per axis.  If the window exceeds the image along an
    axis the whole extent of that axis is used and ``fallback`` is True.
    ``offset_mm`` = (dy, dx) moves the window centre by that many mm (+ = down /
    right) and shifts it back inside the image; None (the default) is the
    original image-centred window, bit for bit.
    """
    if spacing is None or len(spacing) != 2:
        return 0, rows, 0, cols, True
    try:
        sr, sc = float(spacing[0]), float(spacing[1])
    except (TypeError, ValueError):
        return 0, rows, 0, cols, True
    if not (math.isfinite(sr) and math.isfinite(sc)) or sr <= 0 or sc <= 0:
        return 0, rows, 0, cols, True
    h = int(round(crop_mm / sr))
    w = int(round(crop_mm / sc))
    fallback = h > rows or w > cols
    h = max(1, min(h, rows))
    w = max(1, min(w, cols))
    r0 = (rows - h) // 2
    c0 = (cols - w) // 2
    if offset_mm is not None:
        dy, dx = float(offset_mm[0]), float(offset_mm[1])
        if math.isfinite(dy) and math.isfinite(dx):
            r0 = max(0, min(r0 + int(round(dy / sr)), rows - h))
            c0 = max(0, min(c0 + int(round(dx / sc)), cols - w))
    return r0, r0 + h, c0, c0 + w, fallback


# --------------------------------------------------------------------------
# Joint-centre locator (opt-in: build_study_tensor(..., zoom_center="joint"))
# --------------------------------------------------------------------------
#
# The zoom crops default to the IMAGE centre, which is not reliably the joint
# centre (knees sit off-centre in the field of view, and the weakest findings
# -- ACL, lateral meniscus, MCL -- live within a few cm of the joint line).
# ``locate_joint`` estimates the tibiofemoral joint line on ONE decoded
# sagittal / coronal slice, without labels, in about a millisecond:
#   1. the ``crop_mm`` window is shrunk to ~1.1 mm/px, blurred (sigma 1 px)
#      and scaled to its 1-99 percentiles;
#   2. leg mask = tissue (above a low fixed level, opened) -> orthogonal hull
#      (row-wise and column-wise spans) so marrow and fat inside the leg count
#      as leg; a three-class Otsu inside the leg splits dark / muscle / bright
#      and the bone mask is the dark class for fat-suppressed slices
#      (``polarity="dark"``: suppressed marrow) or the bright class for T1
#      (``polarity="bright"``: fatty marrow);
#   3. the bone fraction per row over the leg's central half of columns shows
#      two masses (femur, tibia) interrupted by the cartilage / fluid band:
#      the joint row is the most prominent valley inside the middle ``band``
#      of the window (prominence = min(peak above, peak below) - valley,
#      peaks taken after a 10 mm grey opening so coil fall-off bands at the
#      FOV edge cannot pose as bone), prominence weighted by a soft centre
#      prior (sigma 35 mm); among the candidate valleys (score >= 0.4 x the
#      best) the LOWEST is the joint -- the tibia below it is solid bone,
#      whereas the notch, fat pad and suprapatellar pouch carve extra valleys
#      above it; confidence = prominence / lower peak, < ``min_conf`` ->
#      fallback; on FS slices the row then snaps to the peak of the bright
#      (cartilage / fluid) fraction within 12 mm, which survives weak fat
#      suppression where marrow and muscle look alike;
#   4. the joint column is the centroid of bone inside the eroded leg within
#      +-``JOINT_X_HALF_MM`` of that row (fallback: the leg centroid).
# ``combine_joint_estimates`` merges the slot's ``joint_slice_paths`` (the
# centre anchor's T adjacent slices plus the nearest anchors' centre slices,
# all decoded anyway), anchored on the middle slice -- the outer coronal
# anchors cut the patella and the posterior condyles -- with a majority vote
# when that slice is not confident; ``locate_joint_for_slot`` wires it into
# ``build_study_tensor``.  Everything is clamped to the middle ``band`` of
# the image and to +-JOINT_MAX_OFFSET_MM, and falls back to the image centre
# (``fallback=True``, counted) when no slice is confident.  Axial slots never
# use the locator.

ZOOM_CENTERS = ("image", "joint", "medial")
# "medial" (coronal bases only): the window centre is offset from the LOCATED joint point by
# MEDIAL_CENTRE_MM = (distal, medial) mm, i.e. rows -20..+60 mm and cols -72 medial..+8 lateral
# of the joint for an 80 mm window.  On locator fallback the window is offset from the IMAGE
# centre by MEDIAL_FALLBACK_MM: the joint sits at median +13.1 mm below the image centre
# (measure_medial.py, 157 studies), so +33 mm lands the window at about -7..+73 mm around the
# true joint instead of degenerating to an image-centred crop.
MEDIAL_CENTRE_MM = (20.0, 32.0)
MEDIAL_FALLBACK_MM = (33.0, 32.0)
MEDIAL_BASES = ("COR_FS", "COR_T1")
JOINT_WORK_PX = 128            # locator working resolution (140 mm -> ~1.1 mm/px)
JOINT_BAND = 0.6               # plausible band: the middle 60 % of the image
JOINT_MAX_OFFSET_MM = 40.0     # and never further than this from the image centre
JOINT_MIN_CONF = 0.3           # below this the slice's estimate is discarded
JOINT_X_HALF_MM = 15.0         # rows around the joint line used for the column estimate
_JOINT_SIGMA_MM = 2.5          # smoothing of the row profile
_JOINT_LEG_LEVEL = 24          # 1-99 %-scaled level (of 255) above which a pixel is tissue
_JOINT_MIN_PEAK = 0.15         # bone fraction both masses must reach
_JOINT_MIN_MASS_MM = 10.0      # a bone mass thinner than this is not a peak
_JOINT_PRIOR_MM = 35.0         # soft centre prior (sigma) when choosing between valleys
_JOINT_CAND_FRAC = 0.4         # valleys scoring >= this x the best are candidates; lowest wins
_JOINT_MERGE_MM = 16.0         # candidates this close above the lowest are the same joint space
_JOINT_AGREE_MM = 15.0         # slices within this of the anchor / median agree
_JOINT_MAX_OTHER_ANCHORS = 4   # other anchors (nearest the centre) whose centre slice is used
_JOINT_SNAP_MM = 12.0          # FS: snap the valley to the bright cartilage/fluid band this close
_JOINT_MIN_BRIGHT = 0.08       # ... if that band covers >= this fraction of the central columns


@dataclass
class JointEstimate:
    """Joint centre of one slice (or the combination over a slot's slices).

    ``row`` / ``col`` are full-image pixel coordinates (pixel centres), ``dy_mm``
    / ``dx_mm`` the offsets from the image centre (+ = down / right) in mm.
    ``fallback`` means the image centre is returned (``reason`` tells why).
    """
    row: float
    col: float
    dy_mm: float = 0.0
    dx_mm: float = 0.0
    conf: float = 0.0
    fallback: bool = True
    reason: str = ""
    ms: float = 0.0
    n_slices: int = 1

    @property
    def offset_mm(self) -> Optional[Tuple[float, float]]:
        """(dy, dx) for ``crop_window`` -- None when falling back."""
        return None if self.fallback else (float(self.dy_mm), float(self.dx_mm))

    def as_dict(self) -> dict:
        return {"row": float(self.row), "col": float(self.col),
                "dy_mm": float(self.dy_mm), "dx_mm": float(self.dx_mm),
                "conf": float(self.conf), "fallback": bool(self.fallback),
                "reason": str(self.reason), "ms": float(self.ms), "n_slices": int(self.n_slices)}


def medial_offset(est: "JointEstimate", side: Optional[str]) -> Tuple[Tuple[float, float], bool]:
    """(dy, dx) crop offset in mm for a "medial" zoom entry, plus a side-fallback flag.

    RAW orientation (before apply_laterality): coronal columns run patient-right ->
    patient-left (ImageOrientationPatient row-direction +x in 198/198 local COR_FS
    series), so medial is image-LEFT (dx < 0) for a LEFT knee and image-RIGHT for a
    RIGHT knee; apply_laterality then mirrors R so every cached COR slot has medial
    on the image left.  Locator fallback: MEDIAL_FALLBACK_MM from the image centre.
    Unresolved side: distal shift only, flagged True (counted by the caller).
    """
    if est.fallback:
        dy0, dx0, (dist, med) = 0.0, 0.0, MEDIAL_FALLBACK_MM
    else:
        dy0, dx0, (dist, med) = float(est.dy_mm), float(est.dx_mm), MEDIAL_CENTRE_MM
    sign = {"L": -1.0, "R": 1.0}.get(side)
    if sign is None:
        return (dy0 + dist, dx0), True
    return (dy0 + dist, dx0 + sign * med), False


def _otsu2(values: np.ndarray) -> int:
    """Two-class Otsu split of a uint8 sample: largest t with ``v <= t`` the low class."""
    hist = np.bincount(np.asarray(values, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 127
    bins = np.arange(256, dtype=np.float64)
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * bins)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = w0 * (total - w0) * (m0 / w0 - (m0[-1] - m0) / (total - w0)) ** 2
    between[~np.isfinite(between)] = -1.0
    return int(np.argmax(between))


def _otsu3(values: np.ndarray, nbins: int = 64) -> Tuple[int, int]:
    """Three-class Otsu split of a uint8 sample -> (t_low, t_high) on the 0-255 scale.

    Classes: ``v <= t_low`` (dark), ``t_low < v <= t_high`` (mid), ``v > t_high``
    (bright).  64 histogram bins keep the exhaustive search at ~4k pairs.
    """
    q = 256 // nbins
    hist = np.bincount(np.asarray(values, dtype=np.uint8).ravel() // q, minlength=nbins).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 85, 170
    bins = np.arange(nbins, dtype=np.float64)
    cw = np.concatenate([[0.0], np.cumsum(hist)])
    cm = np.concatenate([[0.0], np.cumsum(hist * bins)])
    t1 = np.arange(0, nbins - 1)[:, None]          # dark class = bins <= t1
    t2 = np.arange(1, nbins)[None, :]              # mid class = t1 < bins <= t2
    w0, m0 = cw[t1 + 1], cm[t1 + 1]
    w1, m1 = cw[t2 + 1] - cw[t1 + 1], cm[t2 + 1] - cm[t1 + 1]
    w2, m2 = total - cw[t2 + 1], cm[-1] - cm[t2 + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        v = m0 ** 2 / w0 + m1 ** 2 / w1 + m2 ** 2 / w2
    v = np.where((t2 > t1) & np.isfinite(v), v, -1.0)
    i, j = np.unravel_index(int(np.argmax(v)), v.shape)
    return int(i) * q + q - 1, int(j + 1) * q + q - 1


def _span_hull(fg: np.ndarray) -> np.ndarray:
    """Orthogonal hull of a boolean mask: pixels inside the first..last
    foreground pixel of their row AND of their column."""
    h, w = fg.shape
    rows_any = fg.any(axis=1)
    cols_any = fg.any(axis=0)
    first_c = np.argmax(fg, axis=1)
    last_c = w - 1 - np.argmax(fg[:, ::-1], axis=1)
    first_r = np.argmax(fg, axis=0)
    last_r = h - 1 - np.argmax(fg[::-1, :], axis=0)
    ci = np.arange(w)[None, :]
    ri = np.arange(h)[:, None]
    span_r = rows_any[:, None] & (ci >= first_c[:, None]) & (ci <= last_c[:, None])
    span_c = cols_any[None, :] & (ri >= first_r[None, :]) & (ri <= last_r[None, :])
    return span_r & span_c


def _smooth1d(v: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(round(3.0 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float64) / sigma) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(v, r, mode="edge"), k, mode="valid")


def joint_polarity(slot: int) -> str:
    """Bone polarity for ``locate_joint``: fat-suppressed slots are "dark", T1 "bright"."""
    return "dark" if int(slot) % N_SLOTS < 3 else "bright"


def _joint_analysis(a: np.ndarray, spacing: Tuple[float, float], crop_mm: float, polarity: str,
                    work_px: int, band: float) -> Tuple[Optional[dict], str]:
    """Steps 1-3 of the locator -> (intermediates dict, reason).  dict is None on failure.

    Shared by ``locate_joint`` and ``scripts/locate_joint_report.py`` so the
    report draws exactly what the locator computed.
    """
    sr, sc = float(spacing[0]), float(spacing[1])
    rows, cols = a.shape
    r0, r1, c0, c1, _ = crop_window(rows, cols, (sr, sc), crop_mm)
    sub = a[r0:r1, c0:c1]
    h, w = sub.shape
    if min(h, w) < 8:
        return None, "tiny image"
    # 1. working resolution (square; mm_r / mm_c absorb the aspect) + blur + 1-99 % scaling
    wh = ww = int(work_px)
    small = resize_to_square(sub, wh)                 # integer-factor fast path, float32
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    lo, hi = np.percentile(small, (1.0, 99.0))
    if not (hi > lo):
        return None, "flat image"
    u8 = np.clip((small - lo) * (255.0 / (hi - lo)), 0.0, 255.0).astype(np.uint8)
    mm_r, mm_c = sr * h / float(wh), sc * w / float(ww)          # mm per working pixel
    # 2. leg = orthogonal hull of the (opened) tissue mask; bone = dark / bright class inside it
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg = cv2.morphologyEx((u8 > _JOINT_LEG_LEVEL).astype(np.uint8), cv2.MORPH_OPEN, k3) > 0
    if fg.sum() < 0.05 * fg.size:
        return None, "no foreground"
    leg = _span_hull(fg)
    inside = u8[leg]
    t_low, t_high = _otsu3(inside)
    # Bright tails (fluid, vessels) can pull the 3-class low cut up into the
    # muscle; the 2-class cut can sit between muscle and fluid when marrow and
    # muscle overlap.  The lower of the two is the dark-marrow cut in both cases.
    t_low = min(t_low, _otsu2(inside))
    bone = leg & ((u8 <= t_low) if polarity == "dark" else (u8 >= t_high))
    # 3. bone fraction per row over the leg's central half of columns, smoothed
    col_mass = np.cumsum(leg.sum(axis=0), dtype=np.float64)
    lo_c = int(np.searchsorted(col_mass, 0.25 * col_mass[-1]))
    hi_c = max(int(np.searchsorted(col_mass, 0.75 * col_mass[-1])), lo_c + 1)
    leg_rows = leg[:, lo_c:hi_c + 1].sum(axis=1).astype(np.float64)
    bone_rows = bone[:, lo_c:hi_c + 1].sum(axis=1).astype(np.float64)
    valid = leg_rows >= 0.25 * (hi_c + 1 - lo_c)
    prof = np.where(valid, bone_rows / np.maximum(leg_rows, 1.0), 0.0)
    prof = _smooth1d(prof, _JOINT_SIGMA_MM / mm_r)
    # FS only: the bright class is cartilage / fluid, i.e. the joint band itself
    # -- a polarity-free cue that survives weak fat suppression (marrow ~ muscle).
    bright = None
    if polarity == "dark":
        bright_rows = (leg & (u8 >= t_high))[:, lo_c:hi_c + 1].sum(axis=1).astype(np.float64)
        bright = _smooth1d(np.where(valid, bright_rows / np.maximum(leg_rows, 1.0), 0.0), _JOINT_SIGMA_MM / mm_r)
    # A bone mass must be at least _JOINT_MIN_MASS_MM tall to count as a peak:
    # a grey opening of the profile erases thinner ridges (coil fall-off bands
    # at the FOV edge, cortical lines) but leaves every valley untouched.
    k_open = max(3, 2 * int(round(0.5 * _JOINT_MIN_MASS_MM / mm_r)) + 1)
    col = prof.astype(np.float32).reshape(-1, 1)
    kern = np.ones((k_open, 1), dtype=np.uint8)
    prof_open = cv2.dilate(cv2.erode(col, kern), kern).ravel().astype(np.float64)
    # the most prominent valley inside the plausible band
    left_max = np.maximum.accumulate(prof_open)
    right_max = np.maximum.accumulate(prof_open[::-1])[::-1]
    lower_peak = np.minimum(left_max, right_max)
    depth = np.maximum(lower_peak - prof, 0.0)
    lo_r = int(round((0.5 - band / 2.0) * (wh - 1)))
    hi_r = int(round((0.5 + band / 2.0) * (wh - 1)))
    # Soft prior: the knee sits near the FOV centre, a suprapatellar effusion
    # 35-45 mm above the joint must not out-score the joint line.
    dist_mm = (np.arange(wh) - (wh - 1) / 2.0) * mm_r
    score = depth * np.exp(-0.5 * (dist_mm / _JOINT_PRIOR_MM) ** 2)
    # Candidates = local maxima of the score inside the band that reach
    # _JOINT_CAND_FRAC of the best; the LOWEST candidate is the joint: the
    # tibia below the joint line is solid bone for 40+ mm, whereas above it
    # the notch, fat pad and suprapatellar pouch all carve extra valleys.
    band_score = score[lo_r:hi_r + 1]
    best = float(band_score.max())
    r_best = lo_r + int(np.argmax(band_score))
    if best > 0:
        inner = band_score[1:-1]
        is_max = (inner >= band_score[:-2]) & (inner >= band_score[2:]) & (inner >= _JOINT_CAND_FRAC * best)
        cands = np.flatnonzero(is_max) + 1
        if cands.size:
            # The femoral and tibial cartilage layers can carve two valleys
            # ~10 mm apart with the menisci between: merge candidates within
            # _JOINT_MERGE_MM above the lowest one (score-weighted mean).
            low = int(cands.max())
            near = cands[(low - cands) * mm_r <= _JOINT_MERGE_MM]
            wts = band_score[near]
            r_best = lo_r + int(round(float((near * wts).sum() / wts.sum())))
    rows_full = r0 + (np.arange(wh) + 0.5) * (h / float(wh)) - 0.5
    return {"u8": u8, "fg": fg, "leg": leg, "bone": bone, "t_low": t_low, "t_high": t_high,
            "cols": (lo_c, hi_c), "prof": prof, "prof_open": prof_open, "depth": depth,
            "score": score, "lower_peak": lower_peak, "bright": bright,
            "r_best": r_best, "band_rows": (lo_r, hi_r), "rows_full": rows_full,
            "mm_r": mm_r, "mm_c": mm_c, "origin": (r0, c0), "sub_shape": (h, w),
            "work_shape": (wh, ww)}, "ok"


def locate_joint(a: np.ndarray, spacing: Optional[Tuple[float, float]], crop_mm: float = 140.0,
                 polarity: str = "dark", work_px: int = JOINT_WORK_PX, band: float = JOINT_BAND,
                 min_conf: float = JOINT_MIN_CONF) -> JointEstimate:
    """Tibiofemoral joint centre of one raw sagittal / coronal slice (see above).

    Never raises on image content: every failure mode returns the image centre
    with ``fallback=True`` and a ``reason``.  ~1 ms on a 512-640 px slice.
    """
    t0 = time.perf_counter()
    if polarity not in ("dark", "bright"):
        raise ValueError("polarity must be 'dark' or 'bright' (got %r)" % (polarity,))
    a = np.asarray(a)
    if a.ndim != 2 or a.size == 0:
        return JointEstimate(0.0, 0.0, reason="not a 2-D image")
    rows, cols = a.shape
    est = JointEstimate((rows - 1) / 2.0, (cols - 1) / 2.0)

    def done(reason: str = "") -> JointEstimate:
        est.reason = reason
        est.ms = (time.perf_counter() - t0) * 1000.0
        return est

    if spacing is None or len(spacing) != 2:
        return done("no spacing")
    sr, sc = _to_float(spacing[0]), _to_float(spacing[1])
    if sr is None or sc is None or sr <= 0 or sc <= 0:
        return done("no spacing")
    an, reason = _joint_analysis(a, (sr, sc), crop_mm, polarity, work_px, band)
    if an is None:
        return done(reason)
    prof, depth, r_best = an["prof"], an["depth"], an["r_best"]
    wh, ww = an["work_shape"]
    peak = float(an["lower_peak"][r_best])
    if peak < _JOINT_MIN_PEAK or depth[r_best] <= 0:
        return done("no bone masses")
    conf = float(depth[r_best] / peak)
    est.conf = conf
    if conf < min_conf:
        return done("low confidence")
    # FS: snap to the bright cartilage / fluid band within _JOINT_SNAP_MM
    bright = an["bright"]
    curve, r_pick = prof, r_best
    if bright is not None:
        half_snap = max(1, int(round(_JOINT_SNAP_MM / an["mm_r"])))
        wa, wb = max(0, r_best - half_snap), min(wh, r_best + half_snap + 1)
        rb = wa + int(np.argmax(bright[wa:wb]))
        if bright[rb] >= _JOINT_MIN_BRIGHT:
            curve, r_pick = -bright, rb
    # sub-pixel refinement (parabola through the three points around the pick)
    r_ref = float(r_pick)
    if 0 < r_pick < wh - 1:
        y0, y1, y2 = curve[r_pick - 1], curve[r_pick], curve[r_pick + 1]
        den = y0 - 2.0 * y1 + y2
        if den > 1e-9:
            r_ref = r_pick + 0.5 * (y0 - y2) / den
    # 4. column: centroid of bone inside the eroded leg within +-15 mm of the joint row
    leg, bone, mm_r, mm_c = an["leg"], an["bone"], an["mm_r"], an["mm_c"]
    k_er = max(3, 2 * int(round(5.0 / mm_c)) + 1)
    leg_in = cv2.erode(leg.astype(np.uint8), np.ones((k_er, k_er), dtype=np.uint8)) > 0
    half = max(1, int(round(JOINT_X_HALF_MM / mm_r)))
    ra, rb = max(0, r_best - half), min(wh, r_best + half + 1)
    sel = bone[ra:rb] & leg_in[ra:rb]
    n_sel = int(sel.sum())
    if n_sel >= 10:
        c_ref = float((sel.sum(axis=0) * np.arange(ww)).sum() / n_sel)
    else:
        c_ref = float((leg.sum(axis=0) * np.arange(ww)).sum() / max(1.0, float(leg.sum())))
    # back to full-image pixel centres, then mm offsets from the image centre, clamped to the band
    (r0, c0), (h, w) = an["origin"], an["sub_shape"]
    row_full = r0 + (r_ref + 0.5) * (h / float(wh)) - 0.5
    col_full = c0 + (c_ref + 0.5) * (w / float(ww)) - 0.5
    dy = (row_full - (rows - 1) / 2.0) * sr
    dx = (col_full - (cols - 1) / 2.0) * sc
    lim_y = min(band / 2.0 * rows * sr, JOINT_MAX_OFFSET_MM)
    lim_x = min(band / 2.0 * cols * sc, JOINT_MAX_OFFSET_MM)
    dy = float(np.clip(dy, -lim_y, lim_y))
    dx = float(np.clip(dx, -lim_x, lim_x))
    est.row = (rows - 1) / 2.0 + dy / sr
    est.col = (cols - 1) / 2.0 + dx / sc
    est.dy_mm, est.dx_mm = dy, dx
    est.fallback = False
    return done("ok")


def combine_joint_estimates(ests: Sequence[JointEstimate], min_conf: float = JOINT_MIN_CONF,
                            centre: Optional[int] = None) -> JointEstimate:
    """One estimate per series from its slice estimates (same geometry).

    Confident = ``conf >= min_conf`` and not a fallback.  The centre slice
    (index ``centre``, default the middle of the list) is the anchor: the
    outer anchors of a coronal stack cut the patella / posterior condyles and
    sagittal ones the condyle rims, the middle slice is the one through the
    joint.  If the centre slice is confident, the result is the median over
    it and every confident slice within ``_JOINT_AGREE_MM`` of it; otherwise
    the largest cluster of confident slices agreeing within ``_JOINT_AGREE_MM``
    must be a majority of all slices (1 of 1-2, 2 of 3-4, ...).  Anything
    else -> the image centre with ``fallback=True``; an empty list -> a zero
    fallback.
    """
    ests = list(ests)
    if not ests:
        return JointEstimate(0.0, 0.0, reason="no slices", n_slices=0)
    if centre is None:
        centre = len(ests) // 2
    good = [e for e in ests if not e.fallback and e.conf >= min_conf]
    ms = float(sum(e.ms for e in ests))
    base = ests[0]
    anchor = ests[centre] if 0 <= centre < len(ests) else None
    if anchor is not None and anchor in good:
        agree = [e for e in good if abs(e.dy_mm - anchor.dy_mm) <= _JOINT_AGREE_MM]
        how = "centre"
    else:
        agree = []
        for seed in sorted(good, key=lambda e: -e.conf):      # largest cluster, ties -> most confident seed
            cl = [e for e in good if abs(e.dy_mm - seed.dy_mm) <= _JOINT_AGREE_MM]
            if len(cl) > len(agree):
                agree = cl
        need = (len(ests) + 1) // 2
        if len(agree) < need:
            return JointEstimate(base.row, base.col, conf=float(max(e.conf for e in ests)),
                                 fallback=True, reason="%d/%d slices agree (%s)" % (len(agree), len(ests), "; ".join(
                                     sorted(set(e.reason for e in ests)))), ms=ms, n_slices=len(ests))
        how = "majority"
    med = lambda k: float(np.median([getattr(e, k) for e in agree]))   # noqa: E731
    return JointEstimate(med("row"), med("col"), med("dy_mm"), med("dx_mm"), med("conf"),
                         fallback=False, reason="ok %s (%d/%d slices)" % (how, len(agree), len(ests)),
                         ms=ms, n_slices=len(ests))


def joint_slice_paths(groups: Sequence[Sequence[str]], max_other: int = _JOINT_MAX_OTHER_ANCHORS
                      ) -> Tuple[List[str], int]:
    """(paths the locator looks at, index of the anchor slice) for one slot's groups.

    All ``T`` slices of the centre anchor (adjacent, all through the joint)
    plus the centre slice of the ``max_other`` anchors nearest the centre --
    every one of them is decoded for the tensor anyway.  The anchor is the
    centre anchor's middle slice.
    """
    G = len(groups)
    if G == 0:
        return [], 0
    gc = G // 2
    order = sorted(range(G), key=lambda g: (abs(g - gc), g))[:1 + max(0, int(max_other))]
    paths: List[str] = []
    for g in order:
        grp = list(groups[g])
        take = grp if g == gc else [grp[len(grp) // 2]]
        for pth in take:
            if pth not in paths:
                paths.append(pth)
    gcentre = list(groups[gc])
    return paths, paths.index(gcentre[len(gcentre) // 2])


def locate_joint_for_slot(slot: int, raw_slices: Sequence[Tuple[np.ndarray, Optional[Tuple[float, float]]]],
                          crop_mm: float = 140.0, min_conf: float = JOINT_MIN_CONF,
                          centre: Optional[int] = None) -> JointEstimate:
    """Joint centre of one slot from its decoded (array, spacing) slices.

    ``centre`` indexes the anchor slice; it is honoured for coronal slots
    only (the middle coronal slice is the one through the joint, the middle
    sagittal slice is the intercondylar notch, so sagittal slots take the
    majority vote).  Axial slots (``AX_*``) always fall back (no joint line).
    """
    if slot % N_SLOTS in AX_SLOTS:
        first = raw_slices[0][0] if raw_slices else None
        if first is None:
            return JointEstimate(0.0, 0.0, reason="axial", n_slices=len(raw_slices))
        return JointEstimate((first.shape[0] - 1) / 2.0, (first.shape[1] - 1) / 2.0,
                             reason="axial", n_slices=len(raw_slices))
    pol = joint_polarity(slot)
    ests = [locate_joint(a, sp, crop_mm=crop_mm, polarity=pol, min_conf=min_conf)
            for a, sp in raw_slices if a is not None]
    if centre is not None and len(ests) != len(raw_slices):
        centre = None                      # a slice failed to decode: indices shifted
    if slot % N_SLOTS not in COR_SLOTS:
        centre = None                      # sagittal: the middle slice is the notch, vote instead
    return combine_joint_estimates(ests, min_conf=min_conf, centre=centre)


# --------------------------------------------------------------------------
# Index
# --------------------------------------------------------------------------

@dataclass
class SeriesInfo:
    uid: str
    files: List[str]
    plane: Optional[str] = None
    fluid: Optional[int] = None
    source: str = "none"            # 'csv' | 'header' | 'none'
    weighting: Optional[str] = None
    spacing: Optional[Tuple[float, float]] = None
    thickness: Optional[float] = None
    header_fallbacks: int = 0
    shape: Optional[Tuple[int, int]] = None
    laterality: Optional[str] = None
    centre_x: Optional[float] = None
    description: str = ""
    slot: Optional[int] = None

    @property
    def n(self) -> int:
        return len(self.files)


@dataclass
class StudyIndex:
    study_dir: str
    uid: str
    slots: Dict[int, List[str]] = field(default_factory=dict)      # slot -> ordered paths
    slot_series: Dict[int, str] = field(default_factory=dict)      # slot -> series uid
    side: Optional[str] = None                                     # 'L' | 'R' | None
    side_source: str = "unresolved"                                # 'tag' | 'geometry' | 'unresolved'
    centre_x: Optional[float] = None
    spacing: Dict[str, Optional[Tuple[float, float]]] = field(default_factory=dict)
    shape: Dict[str, Optional[Tuple[int, int]]] = field(default_factory=dict)
    series: List[SeriesInfo] = field(default_factory=list)
    n_files: int = 0
    n_header_fallback: int = 0
    ms: float = 0.0

    @property
    def missing_slots(self) -> List[int]:
        return [s for s in range(N_SLOTS) if s not in self.slots]

    def slot_spacing(self, slot: int) -> Optional[Tuple[float, float]]:
        uid = self.slot_series.get(slot)
        return self.spacing.get(uid) if uid is not None else None


def _survey_series(uid: str, files: List[str], csv_entry) -> SeriesInfo:
    """One header read (8 KB partial via _read_header) -> series facts."""
    info = SeriesInfo(uid=uid, files=files)
    path = files[len(files) // 2]
    ds, fb = _read_header(path, _SURVEY_TAGS,
                          needed=("ImagePositionPatient", "ImageOrientationPatient",
                                  "PixelSpacing", "Rows"))
    info.header_fallbacks += int(fb)
    rows = cols = None
    spacing = None
    iop = ipp = None
    lat = None
    if ds is not None:
        rows = _to_int(ds.get("Rows"))
        cols = _to_int(ds.get("Columns"))
        ps = ds.get("PixelSpacing")
        if ps is not None and len(ps) == 2:
            sr, sc = _to_float(ps[0]), _to_float(ps[1])
            if sr is not None and sc is not None and sr > 0 and sc > 0:
                spacing = (sr, sc)
        iop = ds.get("ImageOrientationPatient")
        ipp = ds.get("ImagePositionPatient")
        for tag in ("Laterality", "ImageLaterality"):
            v = str(ds.get(tag) or "").strip().upper()
            if v in ("L", "R"):
                lat = v
                break
        info.description = _text(ds.get("SeriesDescription"))
        thick = _to_float(ds.get("SliceThickness"))
        info.thickness = thick if thick is not None and thick > 0 else None
        info.fluid, info.weighting = fluid_from_header(
            ds.get("ScanOptions"), ds.get("SeriesDescription"),
            ds.get("RepetitionTime"), ds.get("EchoTime"))
        info.plane = derive_plane(iop) if iop is not None else None
        info.source = "header"
    info.spacing = spacing
    info.shape = (rows, cols) if rows is not None and cols is not None else None
    info.laterality = lat
    info.centre_x = _centre_x(ipp, iop, rows, cols, spacing)
    # CSV overrides the header wherever it has a value.
    if csv_entry is not None:
        plane, fluid = csv_entry
        if plane is not None:
            info.plane = plane
        if fluid is not None:
            info.fluid = fluid
        if plane is not None and fluid is not None:
            info.source = "csv"
        elif plane is not None or fluid is not None:
            info.source = "csv+header"
    return info


def index_study(study_dir: str, series_df=None) -> StudyIndex:
    """Header-only index of one study (no pixel data is read).

    ``series_df`` may be the full series CSV frame, a frame already restricted
    to this study, a dict ``{series_uid: (plane, fluid)}`` or None (header
    fallback for every series).
    """
    t0 = time.perf_counter()
    study_dir = str(study_dir)
    uid = os.path.basename(os.path.normpath(study_dir))
    lookup = series_lookup(series_df, uid)
    index = StudyIndex(study_dir=study_dir, uid=uid)

    listed = _list_series(study_dir)
    index.n_files = sum(len(f) for _, f in listed)
    for suid, files in listed:
        info = _survey_series(suid, files, lookup.get(suid))
        info.slot = slot_of(info.plane, info.fluid)
        index.n_header_fallback += info.header_fallbacks
        index.series.append(info)
        index.spacing[suid] = info.spacing
        index.shape[suid] = info.shape

    # Slot assignment: tiered preference (see series_preference_key), then the
    # chosen series (only) is geometry-ordered through the 8 KB header path.
    candidates: Dict[int, List[SeriesInfo]] = {}
    for info in index.series:
        if info.slot is not None:
            candidates.setdefault(info.slot, []).append(info)
    for s, cands in candidates.items():
        info = preferred_series(cands)
        paths, geo_plane, n_fb = _order_series_fast(info.files)
        index.n_header_fallback += n_fb
        if info.plane is None and geo_plane is not None:
            info.plane = geo_plane
        index.slots[s] = paths
        index.slot_series[s] = info.uid

    side, source, med = resolve_side([i.laterality for i in index.series],
                                     [i.centre_x for i in index.series])
    index.side, index.side_source, index.centre_x = side, source, med
    index.ms = (time.perf_counter() - t0) * 1000.0
    return index


def _uid_key(uid: str):
    """Sort key so that 'smallest UID' is numeric where possible."""
    try:
        return (0, tuple(int(p) for p in uid.split(".")))
    except ValueError:
        return (1, uid)


# Per-slot series preference (docs/slotknee_review.md §3 A1).  Raw "most
# slices wins" picked 3-D thin-slice acquisitions (VIBE/SPACE, 100+ slices of
# <= 1 mm) over the routine 2-D TSE a radiologist reads.  Tiers, in order:
# slice count within PREFER_N_RANGE; SliceThickness >= PREFER_MIN_THICKNESS_MM
# when the header carries it (unknown thickness is NOT penalised); most
# slices; smallest UID.  Deterministic; when no candidate reaches a tier the
# comparison falls through, so out-of-range-only slots still fill.
PREFER_N_RANGE = (12, 60)
PREFER_MIN_THICKNESS_MM = 2.0


def series_preference_key(info: "SeriesInfo") -> Tuple[int, int, int]:
    in_range = 1 if PREFER_N_RANGE[0] <= info.n <= PREFER_N_RANGE[1] else 0
    thick_ok = 0 if (info.thickness is not None and info.thickness < PREFER_MIN_THICKNESS_MM) else 1
    return (in_range, thick_ok, info.n)


def preferred_series(candidates: Sequence["SeriesInfo"]) -> Optional["SeriesInfo"]:
    """The slot's series under the tiered preference; None for no candidates."""
    best = None
    for info in candidates:
        if best is None:
            take = True
        else:
            ka, kb = series_preference_key(info), series_preference_key(best)
            take = ka > kb or (ka == kb and _uid_key(info.uid) < _uid_key(best.uid))
        if take:
            best = info
    return best


# --------------------------------------------------------------------------
# Slice selection
# --------------------------------------------------------------------------

def anchor_indices(n: int, G: int = 3, T: int = 3, trim_frac: float = 0.15,
                   anchor_shift: int = 0) -> np.ndarray:
    """[G, T] slice indices into a stack of ``n`` geometry-ordered slices.

    Anchors are ``np.linspace`` over ``[trim·(n-1), (1-trim)·(n-1)]`` (G=1 →
    the centre), shifted by ``anchor_shift`` slices and clamped; each anchor
    takes ``T`` adjacent indices ``a + (t - T//2)`` clamped to ``[0, n-1]``.
    """
    if n <= 0:
        raise ValueError("empty stack")
    if G < 1 or T < 1:
        raise ValueError("G and T must be >= 1")
    lo = trim_frac * (n - 1)
    hi = (1.0 - trim_frac) * (n - 1)
    if G == 1:
        anchors = np.array([(lo + hi) / 2.0])
    else:
        anchors = np.linspace(lo, hi, G)
    anchors = np.rint(anchors).astype(int)
    if anchor_shift:
        anchors = np.clip(anchors + int(anchor_shift), 0, n - 1)
    offsets = np.arange(T) - (T // 2)
    idx = anchors[:, None] + offsets[None, :]
    return np.clip(idx, 0, n - 1)


def select_slices(index: StudyIndex, G: int = 3, T: int = 3,
                  trim_frac: float = 0.15, anchor_shift: int = 0) -> Dict[int, List[List[str]]]:
    """slot -> G groups of T geometry-adjacent paths (only these get decoded)."""
    out: Dict[int, List[List[str]]] = {}
    for slot, paths in index.slots.items():
        n = len(paths)
        if n == 0:
            continue
        idx = anchor_indices(n, G, T, trim_frac, anchor_shift)
        out[slot] = [[paths[int(j)] for j in row] for row in idx]
    return out


# --------------------------------------------------------------------------
# Decode / crop / normalise
# --------------------------------------------------------------------------

def resize_to_square(a: np.ndarray, P: int) -> np.ndarray:
    """Resize a 2-D crop to float32 [P, P] — the ONE resize used everywhere.

    cv2.INTER_AREA at a non-integer ratio is the most expensive CPU op in the
    pipeline (docs/slotknee_review.md §2.4: ~1.04 ms at 640->224 vs 0.015 ms
    at an exact 2x), so: shrink by the largest integer factor k first
    (trimming up to k-1 edge pixels per axis, centred, so cv2 hits its
    exact-integer fast path), then finish with INTER_LINEAR on the k^2-times
    smaller image.  A sub-2x downscale (e.g. 336->224) has no integer factor
    and plain INTER_LINEAR would alias, so it keeps single-step INTER_AREA;
    upscales use INTER_LINEAR (INTER_AREA degenerates to nearest-neighbour
    when enlarging).  The resize runs in the input dtype (uint16 is faster
    than float32) and converts once at the end.  Cache building and inference
    both call this function, so their pixels are bit-identical by
    construction.
    """
    a = np.ascontiguousarray(a)
    if a.dtype not in (np.uint8, np.uint16, np.int16, np.float32):
        a = a.astype(np.float32)
    h, w = a.shape
    if (h, w) != (P, P):
        k = min(h // P, w // P)
        if k >= 2:
            h2, w2 = h - h % k, w - w % k
            if (h2, w2) != (h, w):
                r0, c0 = (h - h2) // 2, (w - w2) // 2
                a = np.ascontiguousarray(a[r0:r0 + h2, c0:c0 + w2])
            a = cv2.resize(a, (w2 // k, h2 // k), interpolation=cv2.INTER_AREA)
            if a.shape != (P, P):
                a = cv2.resize(a, (P, P), interpolation=cv2.INTER_LINEAR)
        elif h > P or w > P:
            a = cv2.resize(a, (P, P), interpolation=cv2.INTER_AREA)
        else:
            a = cv2.resize(a, (P, P), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(a, dtype=np.float32)


def _syntax_name(ds) -> str:
    try:
        return str(ds.file_meta.TransferSyntaxUID.name)
    except Exception:
        return "unknown"


_DECODER = "pydicom"


def set_decoder(name: str) -> str:
    """Choose the pixel decoder: 'pydicom' (default) or 'dicomsdl' (several× faster).

    Decoding dominates inference wall-clock (submit kernel: 3607 ms/study decode vs
    1176 ms model), so the decoder is the biggest efficiency lever we have.  Returns the
    decoder actually selected: asking for one that is not installed silently keeps
    pydicom, because a missing wheel must never turn into a failed submission.
    """
    global _DECODER
    if name == "dicomsdl":
        try:
            import dicomsdl  # noqa: F401
        except Exception:
            print("decoder: dicomsdl not importable; staying on pydicom", flush=True)
            name = "pydicom"
    _DECODER = name
    return _DECODER


def _decode_raw_dicomsdl(path: str):
    """dicomsdl fast path -> (array, spacing, syntax). Raises on anything unexpected.

    ``storedvalue=True`` returns the raw stored values, which is what pydicom's
    ``pixel_array`` gives — verified pixel-identical on the local corpus, so the two
    paths are interchangeable and the crops downstream are bit-for-bit the same.
    """
    import dicomsdl
    d = dicomsdl.open(path)
    a = d.pixelData(storedvalue=True)
    info = d.getPixelDataInfo()
    syntax = str(getattr(d, "TransferSyntaxUID", "") or "unknown")
    try:                                   # prefer the human-readable pydicom name
        from pydicom.uid import UID
        syntax = str(UID(syntax).name)
    except Exception:
        pass
    if a is None:
        raise ValueError("dicomsdl returned no pixel data")
    a = np.asarray(a)
    if str(info.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        a = a.max() - a
    spacing = None
    ps = getattr(d, "PixelSpacing", None)
    if ps is not None and len(ps) == 2:
        r, c = _to_float(ps[0]), _to_float(ps[1])
        spacing = None if (r is None or c is None) else (r, c)
    return a, spacing, syntax


def _decode_raw(path: str) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], str]:
    """One file -> (2-D pixel array or None, (row, col) spacing or None, transfer syntax).

    The decode half of ``_decode_crops`` (same handlers, same 3-D / MONOCHROME1
    handling); ``_crops_from_raw`` is the other half.  Split so the joint
    locator can look at the raw slice before it is cropped.
    """
    if _DECODER == "dicomsdl":
        try:
            a, spacing, syntax = _decode_raw_dicomsdl(path)
            if a.ndim == 3:                # same multi-frame / RGB handling as below
                if a.shape[-1] in (3, 4) and a.shape[0] != a.shape[-1]:
                    a = a[..., :3].mean(axis=-1)
                else:
                    a = a[a.shape[0] // 2]
            if a.ndim == 2 and a.size:
                return a, spacing, syntax
        except Exception:
            pass                           # any surprise -> the proven pydicom path
    syntax = "unreadable"
    try:
        ds = pydicom.dcmread(path)
        syntax = _syntax_name(ds)
        a = ds.pixel_array
    except Exception:
        if syntax == "unreadable":
            try:
                syntax = _syntax_name(pydicom.dcmread(path, stop_before_pixels=True,
                                                      specific_tags=["SOPInstanceUID"]))
            except Exception:
                pass
        return None, None, syntax
    try:
        if a.ndim == 3:
            if a.shape[-1] in (3, 4) and a.shape[0] != a.shape[-1]:
                a = a[..., :3].mean(axis=-1)
            else:
                a = a[a.shape[0] // 2]
        if a.ndim != 2 or a.size == 0:
            return None, None, syntax
        if str(ds.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            a = a.max() - a
        spacing = None
        ps = ds.get("PixelSpacing")
        if ps is not None and len(ps) == 2:
            spacing = (_to_float(ps[0]), _to_float(ps[1]))
            if spacing[0] is None or spacing[1] is None:
                spacing = None
        return a, spacing, syntax
    except Exception:
        return None, None, syntax


def _crops_from_raw(a: np.ndarray, spacing, P: int, crop_mms: Sequence[float],
                    offsets_mm=None) -> Tuple[Optional[List[np.ndarray]], List[bool]]:
    """Raw slice -> ([float32 [P, P] crop per crop_mm], [fallback per crop]).

    ``offsets_mm[j]`` (dy, dx) re-centres crop ``j`` (None = image centre).
    """
    try:
        rows, cols = a.shape
        crops, fallbacks = [], []
        for j, mm in enumerate(crop_mms):
            off = offsets_mm[j] if offsets_mm is not None else None
            r0, r1, c0, c1, fb = crop_window(rows, cols, spacing, mm, off)
            crops.append(resize_to_square(a[r0:r1, c0:c1], P))
            fallbacks.append(fb)
        return crops, fallbacks
    except Exception:
        return None, [False] * len(crop_mms)


def _decode_crops(path: str, P: int, crop_mms: Sequence[float], offsets_mm=None
                  ) -> Tuple[Optional[List[np.ndarray]], List[bool], str]:
    """One file, decoded ONCE -> ([float32 [P, P] crop per crop_mm], [fallback per crop], syntax).

    ``pixel_array`` goes through pydicom's pixel handlers, so JPEG-Lossless /
    JPEG2000 decode via pylibjpeg + openjpeg when installed.  Any failure
    returns ``(None, False, syntax)`` so the caller can zero-fill and count it
    per syntax instead of losing the study.  ``offsets_mm`` (per crop, None =
    image centre) re-centres individual crops — used by the joint-centred zoom.
    """
    a, spacing, syntax = _decode_raw(path)
    if a is None:
        return None, [False] * len(crop_mms), syntax
    crops, fallbacks = _crops_from_raw(a, spacing, P, crop_mms, offsets_mm)
    return crops, fallbacks, syntax


def _decode_crop(path: str, P: int, crop_mm: float) -> Tuple[Optional[np.ndarray], bool, str]:
    """Single-crop convenience wrapper around ``_decode_crops``."""
    crops, fbs, syntax = _decode_crops(path, P, [crop_mm])
    return (crops[0] if crops else None), bool(fbs[0]), syntax


def _to_uint8(stack: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Per-slot 1st–99th percentile over the decoded slices -> uint8."""
    out = np.zeros(stack.shape, dtype=np.uint8)
    if not ok.any():
        return out
    vals = stack[ok]
    # Downsample by 16x for the percentile calculation to drastically speed up sorting
    lo, hi = np.percentile(vals.ravel()[::16], (1.0, 99.0))
    if not (hi > lo):
        return out
    scaled = (stack - np.float32(lo)) * np.float32(255.0 / (hi - lo))
    np.clip(scaled, 0.0, 255.0, out=scaled)
    out[:] = np.rint(scaled).astype(np.uint8)
    out[~ok] = 0
    return out


def apply_laterality(x: np.ndarray, side: Optional[str], slot_names=None) -> np.ndarray:
    """Right knee -> mirror so every study looks like a left knee.

    COR/AX slots (incl. their ``_Z`` zoom slots): flip columns.  SAG slots:
    reverse anchor (G) order.  Left / unknown -> unchanged.  ``slot_names``
    defaults to the base 6; pass ``slot_names_ext(zoom_slots)`` for wider x.
    Always returns a new contiguous array.
    """
    out = np.array(x, copy=True)
    if side != "R":
        return out
    names = list(slot_names) if slot_names is not None else list(SLOT_NAMES)
    if len(names) != x.shape[0]:
        raise ValueError("x has %d slots but %d slot names were given" % (x.shape[0], len(names)))
    for s, name in enumerate(names):
        plane = name.split("_")[0]
        if plane in ("COR", "AX"):
            out[s] = x[s][..., ::-1]
        elif plane == "SAG":
            out[s] = x[s][::-1]
        else:
            raise ValueError("cannot infer plane from slot name %r" % name)
    return out


def build_study_tensor(study_dir: str, series_df=None, P: int = 224, crop_mm: float = 140.0,
                       G: int = 3, T: int = 3, laterality: bool = True,
                       trim_frac: float = 0.15, index: Optional[StudyIndex] = None,
                       zoom_mm: Optional[float] = None, zoom_slots=(),
                       anchor_shift: int = 0, zoom_center: str = "image",
                       zoom_spec=None
                       ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """(x uint8 [S, G, T, P, P], mask uint8 [S], info) with S = 6 + len(zoom_slots).

    Only the ``6·G·T`` selected files are pixel-decoded, each exactly once; at
    most one slot's float32 crops (one ``G·T·P·P·4``-byte stack per crop size)
    plus one raw slice are alive at a time.  A slice that fails to decode is
    zero-filled and counted; a slot is marked present only if at least one of
    its slices decoded.

    Zoom slots: for each base name in ``zoom_slots`` a slot ``<BASE>_Z`` is
    appended (order preserved) holding the SAME files and anchors cropped to
    ``zoom_mm`` around the image centre from the same decoded array, resampled
    to P, normalised on its own 1–99 percentiles; mask copied from the base
    slot.  ``anchor_shift`` shifts every anchor by that many slices (clamped).

    ``zoom_center="joint"`` (default "image") centres each SAG/COR zoom slot's
    crops on the tibiofemoral joint estimated by ``locate_joint_for_slot`` from
    the slot's ``joint_slice_paths`` (decoded first, still exactly once); a
    low-confidence estimate falls back to the image centre and is counted in
    ``info["n_joint_fallback"]``; ``info["joint"]`` holds the per-slot
    estimates.  The base crops and every default are unchanged.

    ``zoom_spec`` ("BASE:MM[:CENTER]" entries, see ``normalize_zoom``) appends
    further zoom slots with per-entry mm and centre — e.g.
    ``"SAG_FS:100:joint,SAG_FS:80:joint"`` -> ``SAG_FS_Z100J`` + ``SAG_FS_Z80J``
    (the ACL/notch view, S = 8).  Spec entries follow the legacy trio's slots;
    all of a base slot's crops share its single decode and single joint fix.

    ``center="medial"`` (coronal bases only) is a joint-anchored window whose centre is
    MEDIAL_CENTRE_MM = (20 mm distal, 32 mm medial) from the located joint point
    (``medial_offset``; MEDIAL_FALLBACK_MM from the image centre on locator fallback;
    distal shift only when the side is unresolved, counted in
    ``info["n_medial_side_fallback"]``).  Named ``<BASE>_Z<mm>M``, e.g. ``COR_FS_Z80M``.
    """
    if P % 14 != 0:
        raise ValueError("P must be a multiple of 14 (got %d)" % P)
    if zoom_center not in ZOOM_CENTERS:
        raise ValueError("zoom_center must be one of %s (got %r)" % (ZOOM_CENTERS, zoom_center))
    zoom_slots = parse_zoom_slots(zoom_slots)
    zooms = normalize_zoom(zoom_mm, zoom_slots, zoom_center, zoom_spec)
    names = list(SLOT_NAMES) + [z.name for z in zooms]
    zoom_of: Dict[int, List[Tuple[int, ZoomEntry]]] = {}
    for j, z in enumerate(zooms):
        zoom_of.setdefault(SLOT_NAMES.index(z.base), []).append((N_SLOTS + j, z))
    S = len(names)
    t0 = time.perf_counter()
    if index is None:
        index = index_study(study_dir, series_df)
    t_index = time.perf_counter()
    selection = select_slices(index, G=G, T=T, trim_frac=trim_frac, anchor_shift=anchor_shift)

    x = np.zeros((S, G, T, P, P), dtype=np.uint8)
    mask = np.zeros(S, dtype=np.uint8)
    n_decoded = n_fail = n_fallback = n_zoom_fallback = 0
    n_joint_fallback = 0
    n_medial_side_fallback = 0
    joint_info: Dict[str, dict] = {}
    syntax_hist: Dict[str, int] = {}
    fail_by_syntax: Dict[str, int] = {}
    for slot in range(N_SLOTS):
        groups = selection.get(slot)
        if not groups:
            continue
        zentries = zoom_of.get(slot, [])
        mms = [float(crop_mm)] + [z.mm for _, z in zentries]
        offsets = None
        raw: Dict[str, Tuple[Optional[np.ndarray], Optional[Tuple[float, float]], str]] = {}
        if any(z.center in ("joint", "medial") for _, z in zentries):
            # Decode the locator's slices first (all in the selection, so still
            # exactly one decode each), locate the joint ONCE for the base slot,
            # then cut every zoom entry from the same raw arrays.
            jpaths, jcentre = joint_slice_paths(groups)
            for jp in jpaths:
                raw[jp] = _decode_raw(jp)
            est = locate_joint_for_slot(slot, [(raw[jp][0], raw[jp][1]) for jp in jpaths],
                                        crop_mm=float(crop_mm), centre=jcentre)
            jd = est.as_dict()
            n_joint_fallback += int(est.fallback)
            moff, mfb = medial_offset(est, index.side)      # RAW orientation; side known before the flip
            if any(z.center == "medial" for _, z in zentries):
                jd["medial_offset"] = [float(moff[0]), float(moff[1])]
                jd["medial_side_fallback"] = bool(mfb)
                n_medial_side_fallback += int(mfb)
            joint_info[SLOT_NAMES[slot]] = jd
            offsets = [None] + [est.offset_mm if z.center == "joint"
                                else (moff if z.center == "medial" else None)
                                for _, z in zentries]
        stacks = [np.zeros((G, T, P, P), dtype=np.float32) for _ in mms]
        ok = np.zeros((G, T), dtype=bool)
        cache: Dict[str, Optional[List[np.ndarray]]] = {}
        for g, grp in enumerate(groups):
            for t, path in enumerate(grp):
                if path not in cache:
                    if path in raw:
                        a_raw, sp_raw, syntax = raw.pop(path)
                        if a_raw is None:
                            crops, fbs = None, [False] * len(mms)
                        else:
                            crops, fbs = _crops_from_raw(a_raw, sp_raw, P, mms, offsets)
                        del a_raw
                    else:
                        crops, fbs, syntax = _decode_crops(path, P, mms, offsets)
                    cache[path] = crops
                    if crops is None:
                        n_fail += 1
                        fail_by_syntax[syntax] = fail_by_syntax.get(syntax, 0) + 1
                    else:
                        n_decoded += 1
                        n_fallback += int(fbs[0])
                        n_zoom_fallback += sum(int(v) for v in fbs[1:])
                        syntax_hist[syntax] = syntax_hist.get(syntax, 0) + 1
                crops = cache[path]
                if crops is not None:
                    for j, c in enumerate(crops):
                        stacks[j][g, t] = c
                    ok[g, t] = True
        del cache, raw
        present = 1 if ok.any() else 0
        x[slot] = _to_uint8(stacks[0], ok)
        mask[slot] = present
        for j, (zi, _) in enumerate(zentries):
            x[zi] = _to_uint8(stacks[1 + j], ok)
            mask[zi] = present
        del stacks

    side = index.side
    flipped = False
    if laterality and side == "R":
        x = apply_laterality(x, "R", names)
        flipped = True
    t_end = time.perf_counter()
    info = {
        "uid": index.uid,
        "side": side,
        "side_source": index.side_source,
        "centre_x": index.centre_x,
        "flipped": flipped,
        "slot_names": names,
        "zoom_mm": float(zoom_mm) if zoom_mm is not None else None,
        "zoom_slots": list(zoom_slots),
        "zoom_center": str(zoom_center),
        "zoom_spec": [list(t) for t in parse_zoom_spec(zoom_spec)],
        "joint": joint_info,
        "n_joint_fallback": n_joint_fallback,
        "n_medial_side_fallback": n_medial_side_fallback,
        "anchor_shift": int(anchor_shift),
        "n_files": index.n_files,
        "n_series": len(index.series),
        "n_decoded": n_decoded,
        "n_decode_fail": n_fail,
        "n_header_fallback": index.n_header_fallback,
        "decode_fail_by_syntax": fail_by_syntax,
        "syntax_hist": syntax_hist,
        "n_crop_fallback": n_fallback,
        "n_zoom_crop_fallback": n_zoom_fallback,
        "missing_slots": index.missing_slots,
        "slot_series": dict(index.slot_series),
        "ms": (t_end - t0) * 1000.0,
        "ms_index": (t_index - t0) * 1000.0,
        "ms_decode": (t_end - t_index) * 1000.0,
    }
    return x, mask, info


# --------------------------------------------------------------------------
# Cache reader
# --------------------------------------------------------------------------

def cache_paths(cache_dir: str, split: str = "train") -> Dict[str, str]:
    return {
        "x": os.path.join(cache_dir, "%s_x.u8" % split),
        "mask": os.path.join(cache_dir, "%s_mask.u8" % split),
        "index": os.path.join(cache_dir, "%s_index.json" % split),
    }


def shard_filename(split: str, k: int) -> str:
    """Shard 0 keeps the spec name ``<split>_x.u8``; later shards add ``.001`` etc."""
    return "%s_x.u8" % split if k == 0 else "%s_x.%03d.u8" % (split, k)


def shard_layout(n_rows: int, shard_size: int, split: str = "train") -> List[dict]:
    """[{'x': filename, 'start': row0, 'n': rows}] covering ``n_rows`` rows."""
    if shard_size is None or shard_size <= 0:
        shard_size = max(1, n_rows)
    out = []
    start = 0
    k = 0
    while start < n_rows:
        n = min(shard_size, n_rows - start)
        out.append({"x": shard_filename(split, k), "start": start, "n": n})
        start += n
        k += 1
    return out


class SlotCache:
    """Memory-mapped reader for ``scripts/build_slot_cache.py`` output.

    ``cache[i]`` -> ``(x [S, G, T, P, P] uint8 memmap view, mask [S] uint8)``,
    ``S = len(cache.slot_names)`` (6 for caches without zoom slots);
    nothing is copied until the caller does so.  Multi-shard caches (index key
    ``shards``) are read transparently; a legacy single-file cache without the
    key falls back to ``<split>_x.u8``.
    """

    def __init__(self, dir: str, split: str = "train"):
        self.dir, self.split = str(dir), split
        paths = cache_paths(self.dir, split)
        with open(paths["index"]) as f:
            self.index = json.load(f)
        self.uids: List[str] = list(self.index["studies"])
        self.N = len(self.uids)
        self.P = int(self.index["P"])
        self.G = int(self.index["G"])
        self.T = int(self.index["T"])
        self.crop_mm = float(self.index.get("crop_mm", 140.0))
        self.trim_frac = float(self.index.get("trim_frac", 0.15))     # anchor selection band (0.35 = central block)
        self.version = self.index.get("version", CACHE_VERSION)
        self.done = np.asarray(self.index.get("done", [1] * self.N), dtype=bool)
        self.slot_names: List[str] = list(self.index.get("slot_names") or SLOT_NAMES)
        self.S = len(self.slot_names)                      # old caches: 6
        self.zoom_mm = self.index.get("zoom_mm")
        self.zoom_slots: List[str] = list(self.index.get("zoom_slots") or [])
        self.zoom_center: str = str(self.index.get("zoom_center") or "image")   # old caches: "image"
        self.zoom_spec: List[tuple] = [tuple(v) for v in (self.index.get("zoom_spec") or [])]
        self.row_shape = (self.S, self.G, self.T, self.P, self.P)
        self.shape = (self.N,) + self.row_shape
        self.shards: List[dict] = list(self.index.get("shards") or shard_layout(self.N, 0, split))
        self._starts = [int(s["start"]) for s in self.shards]
        self._x: List[np.ndarray] = []
        for s in self.shards:
            n = int(s["n"])
            if n <= 0:
                self._x.append(np.zeros((0,) + self.row_shape, dtype=np.uint8))
                continue
            self._x.append(np.memmap(os.path.join(self.dir, s["x"]), dtype=np.uint8,
                                     mode="r", shape=(n,) + self.row_shape))
        if self.N > 0:
            self.mask = np.memmap(paths["mask"], dtype=np.uint8, mode="r", shape=(self.N, self.S))
        else:
            self.mask = np.zeros((0, self.S), dtype=np.uint8)
        self._row = {u: i for i, u in enumerate(self.uids)}

    def __len__(self) -> int:
        return self.N

    def _locate(self, i: int) -> Tuple[int, int]:
        if i < 0:
            i += self.N
        if not 0 <= i < self.N:
            raise IndexError("row %d out of range for %d studies" % (i, self.N))
        k = bisect.bisect_right(self._starts, i) - 1
        return k, i - self._starts[k]

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        k, r = self._locate(int(i))
        return self._x[k][r], self.mask[i]

    @property
    def x(self) -> np.ndarray:
        """Whole-cache view; only valid (and zero-copy) for single-shard caches."""
        if len(self._x) == 1:
            return self._x[0]
        raise AttributeError("multi-shard cache: index rows with cache[i] instead of .x")

    def row_of(self, uid: str) -> int:
        return self._row[uid]

    def get(self, uid: str) -> Tuple[np.ndarray, np.ndarray]:
        return self[self.row_of(uid)]

    @property
    def done_rows(self) -> np.ndarray:
        return np.flatnonzero(self.done)

    @property
    def stats(self) -> dict:
        """Stats of the last run that built rows (falls back to the last run)."""
        return dict(self.index.get("build_stats") or self.index.get("stats", {}))

    def side(self, i: int) -> Optional[str]:
        sides = self.index.get("side")
        return sides[i] if sides is not None and i < len(sides) else None
