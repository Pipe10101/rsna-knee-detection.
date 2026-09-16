import os, json, hashlib, random, shutil, time, warnings
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import pydicom
import cv2
import multiprocessing
from multiprocessing import Pool, cpu_count
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RandomBiasField(A.ImageOnlyTransform):
    """MRI intensity inhomogeneity (coil sensitivity falloff).

    field = exp(sum_ij c_ij * x^i * y^j), normalised to mean 1, applied
    multiplicatively.  Standard formulation (Sudre et al. / TorchIO).

    Applied to the whole (H, W, C) stack in ONE call, so every 2.5D slice gets
    the SAME field.  A per-slice field would leave the channels mutually
    inconsistent -- the exact failure `aug_slice_consistent` exists to prevent.
    """

    def __init__(self, coeff_range=(-0.35, 0.35), order=3, p=0.25):
        super().__init__(p=p)
        self.coeff_range = coeff_range
        self.order = int(order)

    def get_params(self):
        n = sum(1 for i in range(self.order + 1)
                for j in range(self.order + 1 - i))
        return {"coeffs": np.random.uniform(self.coeff_range[0],
                                            self.coeff_range[1], size=n)}

    def apply(self, img, coeffs=None, **params):
        h, w = img.shape[:2]
        # Build at low res and upsample: the field is smooth by construction, so
        # this is exact enough and keeps the cost off the training critical path.
        hs, ws = min(h, 64), min(w, 64)
        y, x = np.meshgrid(np.linspace(-1, 1, hs), np.linspace(-1, 1, ws),
                           indexing="ij")
        acc, k = np.zeros((hs, ws), np.float32), 0
        for i in range(self.order + 1):
            for j in range(self.order + 1 - i):
                acc += float(coeffs[k]) * (x ** i) * (y ** j)
                k += 1
        field = np.exp(acc).astype(np.float32)
        field /= float(field.mean())          # preserve overall brightness
        if (hs, ws) != (h, w):
            field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
        if img.ndim == 3:
            field = field[..., None]          # broadcast across the slice stack
        out = img.astype(np.float32) * field
        if img.dtype == np.uint8:
            return np.clip(out, 0, 255).astype(np.uint8)
        return out

    def get_transform_init_args_names(self):
        return ("coeff_range", "order")

# NOTE ON cv2.setNumThreads(0): the usual DataLoader advice is to disable
# OpenCV's internal thread pool so N workers do not spawn N*K threads. It was
# tried here and MEASURED, and it does not pay: 3000 640x640->224x224 resizes
# per process gave a 0.92x / 0.95x / 1.02x / 1.00x speedup at 1 / 2 / 4 / 8
# concurrent worker processes — i.e. neutral at best and ~8% SLOWER in the
# single-process case that train.py falls back to when it forces
# num_workers=0. So this module deliberately does NOT mutate OpenCV's global
# thread state; the pipeline is cache-bound, not resize-bound (a warm study
# costs ~1.0 ms, of which the resize is ~0.05 ms).

# Fallback targets — overridden at runtime by auto-detecting CSV columns in train.py
KNEE_TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]


# ---------------------------------------------------------------------------
# Laterality bookkeeping
#
# A left-right mirror of a CORONAL or AXIAL knee maps medial <-> lateral. Four
# of the twelve targets name a side (Medial/Lateral Meniscus, Medial/Lateral
# OA), and MCL (Medial Collateral Ligament) names a side with no counterpart
# label in this task. Mirroring the pixels without permuting those labels
# teaches the net that a medial finding and a lateral finding are the same
# thing, which is exactly the signal the metric rewards on those columns.
# ---------------------------------------------------------------------------

def _norm_name(name: str) -> str:
    """Canonicalise a target name: 'Medial_Meniscus' == 'medial meniscus'."""
    return " ".join(str(name).replace("_", " ").replace("-", " ").lower().split())


# Side-specific targets that have NO mirror partner, so a mirrored study cannot
# be relabelled correctly. Presence of any of these blocks the flip.
_UNPAIRABLE_SIDED = {"mcl", "medial collateral ligament",
                     "lcl", "lateral collateral ligament"}


def build_lateral_swap(target_names):
    """Build the label permutation induced by a medial<->lateral mirror.

    Returns (perm, pairs, blockers) where
      perm     : list of ints such that swapped[i] = labels[perm[i]], or None
                 if the mirror cannot be represented as a label permutation.
      pairs    : list of (medial_name, lateral_name) that were matched.
      blockers : target names that are side-specific but unpairable.
    """
    names = list(target_names)
    canon = [_norm_name(n) for n in names]
    index = {c: i for i, c in enumerate(canon)}

    perm = list(range(len(names)))
    pairs, matched = [], set()
    for i, c in enumerate(canon):
        if not c.startswith("medial "):
            continue
        partner = "lateral " + c[len("medial "):]
        j = index.get(partner)
        if j is None:
            continue
        perm[i], perm[j] = j, i
        pairs.append((names[i], names[j]))
        matched.update({i, j})

    blockers = []
    for i, c in enumerate(canon):
        if i in matched:
            continue
        if c in _UNPAIRABLE_SIDED:
            blockers.append(names[i])
        elif c.startswith("medial ") or c.startswith("lateral "):
            blockers.append(names[i])   # sided but its partner column is absent

    if not pairs:
        perm = None
    return perm, pairs, blockers


def _cfg(cfg, name, default):
    """Read an augmentation knob, tolerating a Config that predates it."""
    value = getattr(cfg, name, default)
    return default if value is None else value


def build_transforms(cfg, is_train: bool, in_channels: int = None):
    """Build the albumentations pipeline for a stack of `in_channels` slices.

    Verified against albumentations 2.0.8 argument names. Note that
    albumentations 2.x only *warns* about removed kwargs (max_holes=, var_limit=)
    and then silently falls back to its own defaults, so every parameter below
    is spelled with the current name and asserted in tests/test_augmentation.py.

    NOTE: the medial<->lateral mirror is deliberately NOT in this Compose. It is
    applied in RSNADataset.__getitem__ so the label permutation can be applied
    in the same breath — see build_lateral_swap().
    """
    size = int(_cfg(cfg, "image_size", 224))
    c = int(in_channels if in_channels is not None else _cfg(cfg, "in_channels", 3))
    mean = float(_cfg(cfg, "norm_mean", 0.485))
    std = float(_cfg(cfg, "norm_std", 0.229))
    tail = [A.Normalize(mean=[mean] * c, std=[std] * c), ToTensorV2()]

    if not is_train or not bool(_cfg(cfg, "aug_enabled", True)):
        return A.Compose([A.Resize(size, size)] + tail)

    ops = [A.RandomResizedCrop(size=(size, size), scale=(float(_cfg(cfg, "aug_crop_scale_min", 0.85)), 1.0), p=1.0)]

    # Superior<->inferior mirror. Kept configurable but defaulted to 0.0: a knee
    # is always acquired feet-to-head, so an upside-down knee is a distribution
    # the model will never meet at test time and burns capacity to fit.
    vflip_p = float(_cfg(cfg, "aug_vflip_p", 0.0))
    if vflip_p > 0:
        ops.append(A.VerticalFlip(p=vflip_p))

    # Positioning jitter: leg rotation/offset in the coil varies per patient, so
    # small shift/scale/rotate is a genuine re-acquisition of the same knee.
    # border_mode=CONSTANT + fill=0 pads with black (air), never with mirrored
    # anatomy that would fake a second condyle at the edge.
    affine_p = float(_cfg(cfg, "aug_affine_p", 0.7))
    if affine_p > 0:
        shift = float(_cfg(cfg, "aug_shift_limit", 0.0625))
        scale = float(_cfg(cfg, "aug_scale_limit", 0.10))
        rot = float(_cfg(cfg, "aug_rotate_limit", 12.0))
        ops.append(A.Affine(
            translate_percent={"x": (-shift, shift), "y": (-shift, shift)},
            scale=(1.0 - scale, 1.0 + scale),
            rotate=(-rot, rot),
            border_mode=cv2.BORDER_CONSTANT, fill=0,
            p=affine_p,
        ))

    # Non-rigid deformation: models soft-tissue/positioning variability between
    # patients. Parameters measured on a displacement field at 224px (knee FOV
    # ~150 mm => 0.67 mm/px): elastic alpha=40/sigma=6 gives p99 ~1.7 px (~1.1 mm)
    # and grid distort_limit=0.05 gives p99 ~3.4 px (~2.3 mm). Both stay well
    # under meniscus thickness (~4-6 mm) and ACL diameter (~10 mm), so they
    # cannot manufacture a discontinuity that mimics a tear. OneOf, not both:
    # stacking two warps compounds the displacement and doubles the CPU cost.
    deform_p = float(_cfg(cfg, "aug_deform_p", 0.40))
    if deform_p > 0:
        ops.append(A.OneOf([
            A.ElasticTransform(
                alpha=float(_cfg(cfg, "aug_elastic_alpha", 40.0)),
                sigma=float(_cfg(cfg, "aug_elastic_sigma", 6.0)),
                approximate=True, same_dxdy=True,
                border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
            A.GridDistortion(
                num_steps=int(_cfg(cfg, "aug_grid_num_steps", 4)),
                distort_limit=float(_cfg(cfg, "aug_grid_distort_limit", 0.05)),
                normalized=True,
                border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
        ], p=deform_p))

    # Occlusion: simulates signal dropout, coil shading and metal/motion voids.
    # Holes are 3-6% of the side (~7-13 px, ~5-9 mm) so a hole cannot erase a
    # whole ACL/meniscus/patella and silently make the study's label wrong —
    # note albumentations 2.x DEFAULTS are 10-20% (22-45 px, 15-30 mm), which is
    # what the old max_height=32 spelling silently fell back to.
    dropout_p = float(_cfg(cfg, "aug_dropout_p", 0.25))
    if dropout_p > 0:
        lo = float(_cfg(cfg, "aug_dropout_hole_frac_min", 0.03))
        hi = float(_cfg(cfg, "aug_dropout_hole_frac_max", 0.06))
        ops.append(A.CoarseDropout(
            num_holes_range=(1, int(_cfg(cfg, "aug_dropout_max_holes", 3))),
            hole_height_range=(lo, hi), hole_width_range=(lo, hi),
            fill=0, p=dropout_p))
            
    # Structured Dropout (GridDropout): obliterates contiguous features to force
    # the transformer to rely on global anatomical landmarks rather than local textures.
    grid_drop_p = float(_cfg(cfg, "aug_grid_dropout_p", 0.20))
    if grid_drop_p > 0:
        ops.append(A.GridDropout(ratio=0.1, random_offset=True, p=grid_drop_p))
        
    # Patient Motion Blur: explicitly simulates patient movement during the 5-10
    # minute MRI sequence acquisition (a very common artifact).
    motion_blur_p = float(_cfg(cfg, "aug_motion_blur_p", 0.25))
    if motion_blur_p > 0:
        ops.append(A.MotionBlur(blur_limit=(3, 7), p=motion_blur_p))

    # Coil-sensitivity inhomogeneity. Physically the most justified intensity
    # augmentation for MRI after brightness/contrast, and the one that
    # RandomBrightnessContrast structurally cannot produce.
    bias_p = float(_cfg(cfg, "aug_bias_field_p", 0.25))
    if bias_p > 0:
        ops.append(RandomBiasField(
            coeff_range=(-float(_cfg(cfg, "aug_bias_field_coeff", 0.35)),
                         float(_cfg(cfg, "aug_bias_field_coeff", 0.35))),
            order=int(_cfg(cfg, "aug_bias_field_order", 3)),
            p=bias_p))

    # Intensity: MRI has no absolute unit — window/level, receiver gain and
    # vendor scaling differ per scanner, so brightness/contrast jitter is the
    # single most physically-justified augmentation here.
    bc_p = float(_cfg(cfg, "aug_brightness_contrast_p", 0.5))
    if bc_p > 0:
        ops.append(A.RandomBrightnessContrast(
            brightness_limit=float(_cfg(cfg, "aug_brightness_limit", 0.20)),
            contrast_limit=float(_cfg(cfg, "aug_contrast_limit", 0.20)),
            p=bc_p))

    # CLAHE: Contrast Limited Adaptive Histogram Equalization.
    # Enhances local contrast that global min-max normalization hides --
    # critical for MRI where pathological tissue (e.g. torn ACL) has similar
    # intensity to surrounding healthy tissue.  clip_limit=2 is conservative.
    #
    # CHANNEL LIMIT: albumentations 2.0.8 CLAHE accepts ONLY 1- or 3-channel
    # images and raises TypeError otherwise. It is the single channel-limited
    # op in this Compose (all twelve others were tested at c=1,2,3,4,6,9,12).
    # in_channels=3 is the default so nothing changes today, but raising
    # in_channels to get real medial/lateral sagittal coverage would otherwise
    # crash the first training batch. Skip it instead, and say so once.
    clahe_p = float(_cfg(cfg, "aug_clahe_p", 0.30))
    if clahe_p > 0:
        if c in (1, 3):
            ops.append(A.CLAHE(
                clip_limit=float(_cfg(cfg, "aug_clahe_clip", 2.0)),
                tile_grid_size=(8, 8), p=clahe_p))
        else:
            warnings.warn(
                f"aug_clahe_p={clahe_p} but in_channels={c}: albumentations "
                "CLAHE supports only 1- or 3-channel images, so it is disabled "
                "for this stack. Set aug_clahe_p=0 to silence this.",
                RuntimeWarning)

    # Sharpness / blur: different MRI reconstruction kernels (sharp vs smooth)
    # produce images with very different edge profiles. Training on both teaches
    # the model to be kernel-agnostic.
    sharpen_p = float(_cfg(cfg, "aug_sharpen_p", 0.20))
    if sharpen_p > 0:
        ops.append(A.OneOf([
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=sharpen_p))

    # Gamma: fat-suppressed vs non-suppressed sequences differ non-linearly, not
    # just by a gain factor, so a mild gamma covers what brightness cannot.
    gamma_p = float(_cfg(cfg, "aug_gamma_p", 0.20))
    if gamma_p > 0:
        ops.append(A.RandomGamma(gamma_limit=(85, 115), p=gamma_p))

    # Thermal/Rician noise: real and SNR-dependent (thin slices and high-res
    # sequences are noisier). std_range is a FRACTION of full scale in
    # albumentations 2.x: 0.01-0.05 == 2.5-13 grey levels. The old var_limit=
    # spelling was ignored and fell back to std_range=(0.2, 0.44) == 51-112 grey
    # levels, which saturates the image.
    noise_p = float(_cfg(cfg, "aug_noise_p", 0.20))
    if noise_p > 0:
        ops.append(A.GaussNoise(
            std_range=(float(_cfg(cfg, "aug_noise_std_min", 0.01)),
                       float(_cfg(cfg, "aug_noise_std_max", 0.05))),
            per_channel=True, p=noise_p))

    return A.Compose(ops + tail)


# ===========================================================================
# Slice geometry: DICOM order, not filename order
#
# WHY THIS EXISTS.  The previous loader did:
#
#     all_dcm_paths = [every *.dcm under the STUDY directory]
#     all_dcm_paths.sort()                       # <-- lexicographic, by SOP UID
#     margin = int(len(all_dcm_paths) * 0.15)
#     sel = linspace(margin, len - 1 - margin, in_channels)
#
# Every one of those four lines is wrong, and they compound:
#
#  1. FILENAME ORDER IS NOT ANATOMICAL ORDER.  Filenames here are SOP Instance
#     UIDs — random digit strings.  Measured on 212 real series (40 studies):
#     Spearman rho between the filename-sort rank and the true slice position
#     (ImagePositionPatient projected on the slice normal) is |rho| = 0.15
#     (median 0.13, max 0.58); ZERO of 212 series reached |rho| > 0.9.  By
#     contrast InstanceNumber is perfectly geometric: |rho| = 1.00 in 212/212.
#     So "drop the outer 15%" was dropping a hash-random 30% of the study, and
#     `linspace` was sampling three arbitrary slices, not a spread.
#
#  2. THE STUDY IS NOT A SERIES.  os.walk pooled ~156 files from ~5 series
#     across three different acquisition planes into ONE list.  Result over 120
#     studies: the three model channels came from >1 distinct series 100% of the
#     time and from >1 distinct PLANE 95.8% of the time, and *which* plane
#     landed in which channel was decided by UID hash — 24 different plane
#     orderings observed.  A 2.5D channel stack is supposed to be neighbouring
#     parallel slices; this was a random triple of unregistered views.
#
#  3. THE OUTER 15% IS NOT "MOSTLY AIR/SKIN".  Measured with Otsu tissue masks
#     on 178 geometrically-sorted series (34 studies): of the slices the 15%
#     rule discards, 89.7% (Sagittal) and 93.9% (Axial) carry at least half the
#     series' peak tissue area — indistinguishable from the slices it keeps
#     (90.6% / 86.5%).  On Sagittal the trim removes 13.2 mm from each end of a
#     ~101 mm stack: those are the MEDIAL and LATERAL compartments, i.e. the
#     evidence for Medial Meniscus, Lateral Meniscus, Medial OA and Lateral OA.
#     On Coronal the posterior end (popliteal fossa — Baker's cyst) is fully
#     tissue-bearing too; only the anterior end is genuinely empty.
#     Hence slice_trim_frac defaults to 0.0 and sampling uses bin centres,
#     which never lands on the outermost slice anyway.
# ===========================================================================

_PLANES = ("Sagittal", "Coronal", "Axial")

# Only these tags are parsed for the ordering pass. stop_before_pixels keeps the
# pixel data off the wire entirely: 0.19 ms/file vs 0.29 ms with pixels, and it
# is the difference between a 29 ms/study metadata pass and a 144 ms/study one.
_GEOM_TAGS = ["InstanceNumber", "ImagePositionPatient",
              "ImageOrientationPatient", "SeriesInstanceUID"]

_CACHE_VERSION = "v2geo"     # bump => old (filename-ordered) caches are ignored


def derive_plane(iop):
    """Anatomical plane from ImageOrientationPatient, or None.

    The slice normal is the cross product of the row and column direction
    cosines; whichever patient axis it points along names the plane
    (x=L/R -> Sagittal, y=A/P -> Coronal, z=S/I -> Axial).
    Validated against data_subset/train_series.csv on 447 real series:
    447/447 agreement (100%), so this works on the offline Kaggle test set
    where no series CSV plane column may be available.
    """
    if iop is None or len(iop) != 6:
        return None
    v = np.asarray(iop, dtype=float)
    n = np.cross(v[:3], v[3:])
    if not np.isfinite(n).all() or not n.any():
        return None
    return _PLANES[int(np.argmax(np.abs(n)))]


def _read_geometry(path):
    """(instance_number, position_along_normal, plane) for one file.

    Metadata only — pydicom never touches the pixel data.
    """
    ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=_GEOM_TAGS)
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
    return inst, pos, derive_plane(iop)


def _list_series_dirs(study_dir):
    """[(series_uid, [file paths])] — one entry per series subdirectory.

    Falls back to treating the study directory itself as a single series when
    the DICOMs sit directly in it (some exports are flat).
    """
    out, flat = [], []
    try:
        entries = sorted(os.scandir(study_dir), key=lambda e: e.name)
    except OSError:
        return out
    for e in entries:
        if e.is_dir():
            files = sorted(
                os.path.join(e.path, f.name)
                for f in os.scandir(e.path)
                if f.is_file() and f.name.endswith(".dcm"))
            if files:
                out.append((e.name, files))
        elif e.is_file() and e.name.endswith(".dcm"):
            flat.append(e.path)
    if flat and not out:
        out.append((os.path.basename(study_dir), sorted(flat)))
    return out


def order_series(files):
    """Sort one series' files head->toe / medial->lateral / anterior->posterior.

    Ordering authority, in order of preference:
      1. ImagePositionPatient . normal   (true geometry; present on every one of
         the 212 series sampled here)
      2. InstanceNumber                  (|rho| = 1.00 vs geometry on 212/212)
      3. filename                        (last resort; provably meaningless —
         |rho| = 0.15 vs geometry, 0/212 series above 0.9)
    Returns (ordered_paths, plane).
    """
    keyed, plane = [], None
    for p in files:
        try:
            inst, pos, pl = _read_geometry(p)
        except Exception:
            inst, pos, pl = None, None, None
        if pl and plane is None:
            plane = pl
        keyed.append((pos, inst, p))
    if all(k[0] is not None for k in keyed):
        keyed.sort(key=lambda k: (k[0], k[2]))
    elif all(k[1] is not None for k in keyed):
        keyed.sort(key=lambda k: (k[1], k[2]))
    else:
        keyed.sort(key=lambda k: k[2])
    return [k[2] for k in keyed], plane


def build_series_index(study_dir, geometric=True, order_uids=None):
    """Index of a study: [{'uid', 'plane', 'paths'}], series sorted by UID.

    `order_uids` is the laziness knob and the single largest cold-path saving
    here. The header pass is 95% of the cold cost (57 ms/study over ~182 files),
    but plane_balanced only ever samples from one primary series per plane —
    ~3 of ~6 series. Passing the set of UIDs that will actually be sampled
    orders only those and reads a single header from each of the rest (enough
    to know its plane and slice count, which is all the selector needs to make
    the identical choice). order_uids=None orders everything; order_uids=set()
    is the survey-only pass.
    """
    index = []
    for uid, files in _list_series_dirs(study_dir):
        if not geometric:
            index.append({"uid": uid, "plane": None, "paths": list(files)})
            continue
        if order_uids is None or uid in order_uids:
            paths, plane = order_series(files)
        else:
            # survey only: one header for the plane, filename order retained.
            # Nothing samples from this series, so its order is never consulted.
            plane = None
            try:
                plane = _read_geometry(files[0])[2]
            except Exception:
                pass
            paths = list(files)
        index.append({"uid": uid, "plane": plane, "paths": paths})
    index.sort(key=lambda s: s["uid"])
    return index


def plan_series_uids(index, k, cfg):
    """UIDs select_slice_paths() will draw from, decided WITHOUT any ordering.

    Only plane and slice count feed the decision, and both survive the
    survey-only pass, so the lazy path provably selects the same series as the
    eager one.
    """
    if str(_cfg(cfg, "slice_selection", "plane_balanced")) == "study_pooled":
        return {s["uid"] for s in index}
    by_plane = {}
    for s in index:
        if s["paths"]:
            by_plane.setdefault(s["plane"] or "Sagittal", []).append(s)
    for pl in by_plane:
        by_plane[pl].sort(key=lambda s: (-len(s["paths"]), s["uid"]))
    order = [p for p in _plane_priority(cfg) if p in by_plane]
    if str(_cfg(cfg, "slice_selection", "plane_balanced")) == "single_series":
        order = order[:1]
    return {by_plane[pl][0]["uid"]
            for pl, q in zip(order, _quotas(k, len(order))) if q > 0}


# --- on-disk index cache ----------------------------------------------------
# The ordering pass costs ~29 ms/study for every file in the study (~5 ms when
# restricted to one series per plane). That is cheap once, but it would be paid
# again for every resolution phase, because train.py scopes the pixel cache by
# image_size. The index does not depend on image_size or in_channels, so it
# lives one level up and is written once per study for the whole run.

def _index_dir(cache_dir):
    """Directory holding the JSON slice-index sidecars for `cache_dir`.

    One level ABOVE the resolution-scoped pixel cache, because the index does
    not depend on image_size or in_channels and is therefore shared by every
    resolution phase of a run.
    """
    root = os.path.dirname(os.path.abspath(cache_dir)) or cache_dir
    return os.path.join(root, "_slice_index")


def _index_cache_path(cache_dir, study_id, geometric):
    d = _index_dir(cache_dir)
    os.makedirs(d, exist_ok=True)
    tag = "geo" if geometric else "fname"
    return os.path.join(d, f"{study_id}_{tag}.json")


def _atomic_write(path, write_fn):
    """Write via a private temp file + os.replace.

    Without this, two DataLoader workers racing on the same study leave a
    half-written .npy that np.load either rejects or, worse, reads short.
    os.replace is atomic on POSIX. Nothing pre-existing is ever removed — the
    only file this function can unlink is the temp file it just created itself.

    write_fn receives an OPEN BINARY FILE OBJECT, not a path. That matters:
    np.save(path, ...) silently appends '.npy' when the name does not already
    end in it, so a path-based temp ('...npy.1234.ab.tmp') would be written to
    '...npy.1234.ab.tmp.npy' and the subsequent os.replace would raise
    FileNotFoundError — i.e. the cache would never be written at all. Handing
    over the file object removes that whole class of failure.
    """
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, f".{os.path.basename(path)}."
                          f"{os.getpid()}.{random.getrandbits(32):08x}.tmp")
    try:
        with open(tmp, "wb") as fh:
            write_fn(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)          # only ever the temp file this call made
        except OSError:
            pass
        return False


def load_series_index(study_dir, study_id, cache_dir=None, geometric=True,
                      use_cache=True, lazy_k=None, lazy_cfg=None):
    """build_series_index() with a JSON sidecar so the header pass is paid once.

    With lazy_k/lazy_cfg the ordering pass is restricted to the series the
    selector will actually sample (see build_series_index.order_uids): a survey
    pass reads one header per series, plan_series_uids() picks the same series
    it would have picked from a fully ordered index, and only those are sorted.
    Measured 57 ms -> 30 ms per study on the first pass; the sidecar then makes
    every later pass ~0.
    """
    cp = None
    if use_cache and cache_dir:
        cp = _index_cache_path(cache_dir, study_id, geometric)
        try:
            with open(cp) as fh:
                cached = json.load(fh)
            if cached.get("root") == study_dir:
                return cached["series"]
        except Exception:
            pass

    if geometric and lazy_k:
        survey = build_series_index(study_dir, geometric=True, order_uids=set())
        index = build_series_index(
            study_dir, geometric=True,
            order_uids=plan_series_uids(survey, int(lazy_k), lazy_cfg))
    else:
        index = build_series_index(study_dir, geometric=geometric)

    if cp is not None and index:
        _atomic_write(cp, lambda fh: fh.write(json.dumps(
            {"root": study_dir, "series": index}).encode("utf-8")))
    return index


# ===========================================================================
# Slice selection
# ===========================================================================

def _plane_priority(cfg):
    raw = str(_cfg(cfg, "slice_plane_priority", "Sagittal,Coronal,Axial"))
    order, seen = [], set()
    for tok in raw.split(","):
        t = tok.strip().capitalize()
        if t in _PLANES and t not in seen:
            order.append(t); seen.add(t)
    for p in _PLANES:
        if p not in seen:
            order.append(p)
    return order


def _sample_positions(n, k, trim_frac, mode, jitter=0.0):
    """k indices into a geometrically ordered run of n slices.

    mode='bin_center' (default): split the (trimmed) run into k equal bins and
    take each bin's centre. This spreads across the FULL anatomical range while
    never selecting the outermost slice — which measurement shows is the only
    one that is reliably air/skin (sagittal edge slice: 10% tissue fraction and
    54 mm A-P extent, vs 37-42% and 119-131 mm one slice in).

    mode='endpoints': legacy np.linspace(start, end, k), which includes both
    extreme slices.
    """
    if n <= 0 or k <= 0:
        return []
    m = int(n * float(trim_frac))
    lo, hi = m, n - 1 - m
    if hi - lo + 1 < k:                 # trim would starve the request
        lo, hi = 0, n - 1
    if mode == "endpoints":
        idx = [int(i) for i in np.linspace(lo, hi, k, dtype=int)]
    else:
        span = hi - lo + 1
        idx = [int(lo + min(span - 1, int((j + 0.5) * span / k))) for j in range(k)]
    if jitter > 0:
        # Per-slice positional jitter INSIDE the series, in geometric units, so
        # the model never memorises one fixed set of cross-sections. Only
        # reachable with cfg.cache_slices=False (a cached study is frozen).
        j = max(1, int(round(n * float(jitter))))
        idx = [int(min(n - 1, max(0, i + random.randint(-j, j)))) for i in idx]
    return idx


def _quotas(k, n_groups):
    """Split k slots over n_groups as evenly as possible, front-loaded."""
    if n_groups <= 0:
        return []
    base, rem = divmod(k, n_groups)
    return [base + (1 if i < rem else 0) for i in range(n_groups)]


def select_slice_paths(index, k, cfg, jitter=0.0):
    """Choose `k` DICOM paths from a geometrically ordered series index.

    Modes (cfg.slice_selection):
      'plane_balanced' (default) — deterministic quota per anatomical plane in
          cfg.slice_plane_priority order; within a plane the series with the
          most slices wins (the primary acquisition, not a 3-image localiser)
          and its quota is spread over the whole geometric range.  Channel c
          therefore means the SAME view for every study, which the previous
          hash-ordered pooling never guaranteed.
      'single_series' — all k slices from one series of the highest-priority
          plane present.  With k=3 on Sagittal this yields medial compartment /
          intercondylar notch / lateral compartment, i.e. direct evidence for
          Medial+Lateral Meniscus, Medial+Lateral OA and ACL.
      'study_pooled' — bit-exact legacy behaviour, kept so the old tensors can
          be reproduced and A/B'd.  Do not use for training.
    """
    mode = str(_cfg(cfg, "slice_selection", "plane_balanced"))
    trim = float(_cfg(cfg, "slice_trim_frac", 0.0))
    samp = str(_cfg(cfg, "slice_sample", "bin_center"))

    if mode == "study_pooled":
        pool = [p for s in index for p in s["paths"]]
        pool.sort()
        n = len(pool)
        if n == 0:
            return []
        if n < k:
            return pool + [pool[-1]] * (k - n)
        return [pool[i] for i in _sample_positions(n, k, trim, samp, jitter)]

    by_plane = {}
    for s in index:
        if not s["paths"]:
            continue
        by_plane.setdefault(s["plane"] or "Sagittal", []).append(s)
    if not by_plane:
        return []
    # deterministic primary series per plane: most slices, UID as tiebreak
    for pl in by_plane:
        by_plane[pl].sort(key=lambda s: (-len(s["paths"]), s["uid"]))

    order = [p for p in _plane_priority(cfg) if p in by_plane]
    if mode == "single_series":
        order = order[:1]
    # _quotas always sums to exactly k, so no slot is lost; k < len(order)
    # simply means the lowest-priority planes get 0 and are skipped.
    picked = []
    for pl, q in zip(order, _quotas(k, len(order))):
        if q <= 0:
            continue
        paths = by_plane[pl][0]["paths"]
        for i in _sample_positions(len(paths), q, trim, samp, jitter):
            picked.append(paths[i])
    if not picked:
        return []
    # A series shorter than its quota can under-deliver (_sample_positions
    # clamps to n): pad with the last slice, as the legacy path did.
    while len(picked) < k:
        picked.append(picked[-1])
    return picked[:k]


def selection_signature(cfg, in_channels):
    """Short hash of everything that changes the produced pixels.

    Goes into the cache filename so a cache built under the old (broken)
    filename-ordered policy can never be silently reused by the fixed one.
    """
    key = "|".join(str(x) for x in (
        _CACHE_VERSION,
        int(in_channels),
        int(_cfg(cfg, "image_size", 224)),
        str(_cfg(cfg, "slice_order", "geometric")),
        str(_cfg(cfg, "slice_selection", "plane_balanced")),
        float(_cfg(cfg, "slice_trim_frac", 0.0)),
        str(_cfg(cfg, "slice_sample", "bin_center")),
        str(_cfg(cfg, "slice_plane_priority", "Sagittal,Coronal,Axial")),
    ))
    return hashlib.sha1(key.encode()).hexdigest()[:10]


def cache_path_for(cache_dir, study_id, in_channels, cfg):
    return os.path.join(
        cache_dir, f"{study_id}_{in_channels}_{selection_signature(cfg, in_channels)}.npy")


# ===========================================================================
# Decode
# ===========================================================================

def decode_slice(path, size):
    """One DICOM -> min-max normalised uint8 (size, size).

    Bit-identical to the previous implementation (verified: 0 differing pixels
    over 240 real slices) but 1.53x faster, because:
      * np.nan_to_num is skipped for integer pixel data — an int array cannot
        hold NaN or +-Inf, so the call was a guaranteed no-op full-array copy
        of a 640x640 buffer;
      * min/max are taken on the native uint16 array instead of a float32 copy
        (uint16 -> float32 is exact for BitsStored <= 24, so the extrema and
        therefore every subsequent value are unchanged);
      * the subtraction and division are the same float32 ops in the same order,
        so no new rounding is introduced (a reciprocal-multiply variant was
        1 grey level off and was rejected).
    """
    ds = pydicom.dcmread(path)
    a = ds.pixel_array
    if a.dtype.kind == "f":
        a = np.nan_to_num(a.astype(np.float32, copy=False),
                          nan=0.0, posinf=0.0, neginf=0.0)
        lo, hi = float(np.min(a)), float(np.max(a))
    else:
        lo, hi = float(a.min()), float(a.max())
        a = a.astype(np.float32, copy=False)
    if hi > lo:
        f = (a - np.float32(lo)) / np.float32(hi - lo)
    else:
        f = np.zeros(a.shape, dtype=np.float32)
    f = cv2.resize(f, (size, size))
    return (f * 255).astype(np.uint8)


# ===========================================================================
# Study loading + cache
# ===========================================================================

def load_study_slices(study_id, image_dir, cfg, cache_dir=None,
                      in_channels=None, use_cache=True, allow_index_cache=True,
                      jitter=0.0):
    """(list of `in_channels` uint8 HxW slices, cache_status).

    Single implementation shared by RSNADataset and the pre-cache pool. They
    used to be two copy-pasted bodies that had already drifted apart (the
    pre-cache pool never applied the training-time slice jitter, so whichever
    ran first decided the pixels for the whole run).

    cache_status is one of 'hit' | 'miss' | 'nodir' | 'nofiles', for profiling.
    """
    C = int(in_channels if in_channels is not None else _cfg(cfg, "in_channels", 3))
    size = int(_cfg(cfg, "image_size", 224))
    blank = np.zeros((size, size), dtype=np.uint8)

    cpath = None
    if use_cache and cache_dir:
        cpath = cache_path_for(cache_dir, study_id, C, cfg)
        if os.path.exists(cpath):
            try:
                arr = np.load(cpath)
                if arr.shape == (C, size, size) and arr.dtype == np.uint8:
                    return [arr[i] for i in range(C)], "hit"
            except Exception:
                pass                     # corrupt/stale entry: fall through

    study_dir = os.path.join(image_dir, str(study_id))
    if not os.path.isdir(study_dir):
        return [blank] * C, "nodir"

    geometric = str(_cfg(cfg, "slice_order", "geometric")) != "filename"
    index = load_series_index(study_dir, str(study_id), cache_dir=cache_dir,
                              geometric=geometric,
                              use_cache=bool(allow_index_cache and
                                             _cfg(cfg, "slice_index_cache", True)),
                              lazy_k=C, lazy_cfg=cfg)
    paths = select_slice_paths(index, C, cfg, jitter=jitter)
    if not paths:
        return [blank] * C, "nofiles"

    slices = [_safe_decode(p, size, blank) for p in paths]
    stack = np.stack(slices, axis=0)

    if cpath is not None:
        write_slice_cache(cpath, stack, cfg)
    return [stack[i] for i in range(C)], "miss"


def _safe_decode(path, size, blank):
    try:
        return decode_slice(path, size)
    except Exception:
        return blank


def write_slice_cache(cpath, stack, cfg=None):
    """Persist a study's slice stack, refusing an all-zero one.

    An earlier version of this project accumulated 13,230 all-zero cache
    entries: one transient read failure per study was frozen into the cache and
    every later epoch happily served a black image as if it were a knee. A
    black stack is never legitimate output for a real study, so it is treated
    as a failure to be retried rather than a result to be memoised.
    Returns True if written.
    """
    if cfg is None or bool(_cfg(cfg, "cache_reject_zero", True)):
        if not np.any(stack):
            return False
    return _atomic_write(cpath, lambda fh: np.save(fh, stack, allow_pickle=False))


# ===========================================================================
# How many CPUs do we actually have, and where can the cache live?
#
# WHY THIS EXISTS.  Two host-level facts decide whether a Kaggle session
# survives, and the pipeline previously guessed at both:
#
#  1. os.cpu_count() / multiprocessing.cpu_count() report the HOST's core
#     count, not the container's. Inside a cgroup-limited notebook they can
#     over-report by 8x, and the pre-cache pool sized itself from that number.
#     sched_getaffinity() honours the affinity mask and the cgroup cpu.max /
#     cpu.cfs_quota_us files hold the quota that is actually enforced, so all
#     three are consulted and the smallest wins.
#
#  2. The cache is not small any more. Measured on real studies (see
#     estimate_cache_bytes, whose pixel term is exact, not fitted):
#
#       in_channels   224px      384px
#            3       0.66 GB    1.95 GB
#            6       1.33 GB    3.90 GB
#            9       1.99 GB    5.85 GB      (4,407 studies, pixels only)
#
#     plus ~0.13-0.21 GB of JSON slice-index sidecars and one full pixel cache
#     PER RESOLUTION (train.py scopes cache_dir by sz{size}_ch{channels}), so
#     the 224px+384px pipeline at in_channels=9 needs ~8 GB. Two placements
#     lose the session outright:
#       * /kaggle/working — everything there is committed as notebook OUTPUT
#         and counts against the ~19.5 GiB output limit, alongside the model
#         checkpoints. 8,814 extra files also make the commit crawl.
#       * any tmpfs/ramfs — a RAM-backed cache is charged to the same
#         13-16 GB the training process needs.
#     Neither can be assumed from here (image layouts change), so both are
#     MEASURED at runtime: shutil.disk_usage for capacity, /proc/mounts for
#     the filesystem type.
# ===========================================================================

def available_cpus():
    """Cores this process may actually use — affinity and cgroup aware.

    Returns at least 1. Prefer this to os.cpu_count() for sizing worker pools:
    on a 2-vCPU Kaggle container os.cpu_count() can still report the host's
    full core count, and a pool sized from that thrashes.
    """
    counts = []
    n = os.cpu_count()
    if n:
        counts.append(int(n))
    try:                                   # affinity mask (Linux)
        counts.append(len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        pass
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()
        if quota != "max" and int(period) > 0:
            counts.append(max(1, int(float(quota) / float(period) + 0.5)))
    except (OSError, ValueError):
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            quota = int(fh.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            period = int(fh.read().strip())
        if quota > 0 and period > 0:
            counts.append(max(1, int(quota / period + 0.5)))
    except (OSError, ValueError):
        pass
    return max(1, min(counts)) if counts else 1


# Measured on real studies: a cached stack is exactly C*S*S uint8 plus numpy's
# fixed 128-byte .npy header (verified below in estimate_cache_bytes' docstring
# example and in tests/test_kaggle_data_memory.py).
_NPY_HEADER_BYTES = 128

# JSON slice-index sidecar, bytes per study. It is ~ (files in study) x (length
# of one absolute DICOM path), so it scales with the mount's path length:
# measured 47 KiB/study on this checkout (149-char data root, 169 files/study
# mean) and ~28 KiB/study projected for Kaggle's shorter
# /kaggle/working/data/train_images root. The higher figure is the default so
# the budget errs towards refusing to start rather than filling the disk.
_INDEX_BYTES_PER_STUDY = 48 * 1024


def estimate_cache_bytes(n_studies, in_channels, image_size,
                         index_bytes_per_study=_INDEX_BYTES_PER_STUDY,
                         block_size=4096):
    """Bytes `n_studies` will occupy in one resolution-scoped cache directory.

    The pixel term is exact: np.save of a (C, S, S) uint8 array writes
    C*S*S + 128 bytes, rounded up to the filesystem block. The index term is
    a measured per-study constant (see _INDEX_BYTES_PER_STUDY); pass a value
    measured from a warm cache with measured_index_bytes() for a tighter
    figure, or 0 to price the pixels alone.

    >>> estimate_cache_bytes(1, 3, 224, index_bytes_per_study=0)
    151552
    """
    n = max(0, int(n_studies))
    px = int(in_channels) * int(image_size) * int(image_size) + _NPY_HEADER_BYTES
    if block_size > 0:
        px = -(-px // block_size) * block_size
        if index_bytes_per_study:
            index_bytes_per_study = -(-int(index_bytes_per_study)
                                      // block_size) * block_size
    return n * (px + int(index_bytes_per_study))


def measured_index_bytes(cache_dir, sample=64):
    """Mean sidecar size actually on disk, or None if none have been written.

    The sidecar length is dominated by the absolute DICOM paths it stores, so
    it is a property of the mount, not of this code. Measuring beats guessing.
    """
    d = _index_dir(cache_dir)
    try:
        names = [e.name for e in os.scandir(d) if e.is_file()
                 and e.name.endswith(".json")]
    except OSError:
        return None
    if not names:
        return None
    names = names[:max(1, int(sample))]
    total = 0
    for name in names:
        try:
            total += os.path.getsize(os.path.join(d, name))
        except OSError:
            return None
    return total / float(len(names))


def _nearest_existing(path):
    """`path` if it exists, else its closest existing ancestor.

    Free space and filesystem type are properties of the mount, and an
    unborn directory inherits its parent's mount, so this answers the same
    question without having to create anything first.
    """
    p = os.path.abspath(path)
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return p or os.sep


def _fs_type(path):
    """Filesystem type backing `path` ('tmpfs', 'ext4', ...), or '' if unknown.

    Longest-prefix match against /proc/mounts, which is the only place the
    answer is recorded on Linux. Returns '' off Linux, where the RAM-backed
    question does not arise for the paths this module considers.
    """
    try:
        real = os.path.realpath(path)
        best, best_type = "", ""
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mp, fstype = parts[1], parts[2]
                mp = mp.replace("\\040", " ")
                if (real == mp or real.startswith(mp.rstrip("/") + "/")) \
                        and len(mp) > len(best):
                    best, best_type = mp, fstype
        return best_type
    except OSError:
        return ""


# A cache on any of these is charged to the same RAM the training process
# needs, which on a 13-16 GB Kaggle box is exactly the budget under pressure.
_RAM_BACKED = {"tmpfs", "ramfs", "devtmpfs"}


def on_kaggle():
    return bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or \
        os.path.isdir("/kaggle/input")


def _writable_dir(path):
    """True if `path` exists (or can be created) and accepts a write."""
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        return False
    probe = os.path.join(path, f".rsna_probe.{os.getpid()}")
    try:
        with open(probe, "wb") as fh:
            fh.write(b"0")
        os.unlink(probe)                   # only ever the probe this call made
        return True
    except OSError:
        return False


def cache_location_report(candidates=None, need_bytes=0):
    """[{path, free, total, fstype, ram_backed, is_output, writable, fits}].

    Pure inspection — creates nothing except the write probe it removes again.
    Ordered as given; the caller decides. `need_bytes` only fills in `fits`.
    """
    if candidates is None:
        candidates = default_cache_candidates()
    out = []
    for path in candidates:
        probe = _nearest_existing(path)
        rec = {"path": path, "free": 0, "total": 0, "fstype": "",
               "ram_backed": False, "is_output": False,
               "writable": False, "fits": False}
        try:
            usage = shutil.disk_usage(probe)
            rec["free"], rec["total"] = usage.free, usage.total
        except OSError:
            out.append(rec)
            continue
        rec["fstype"] = _fs_type(probe)
        rec["ram_backed"] = rec["fstype"] in _RAM_BACKED
        rec["is_output"] = os.path.realpath(path).startswith("/kaggle/working")
        rec["writable"] = _writable_dir(path)
        rec["fits"] = (rec["writable"] and not rec["ram_backed"]
                       and rec["free"] >= int(need_bytes))
        out.append(rec)
    return out


def default_cache_candidates():
    """Cache roots to try, best first.

    /kaggle/working is deliberately LAST: it is the notebook's output
    directory, so everything written there is committed at the end of the
    session and counts against the ~19.5 GiB output limit that the model
    checkpoints also draw on.
    """
    if on_kaggle():
        return ["/kaggle/temp/rsna_cache", "/tmp/rsna_cache",
                "/kaggle/working/.rsna_cache"]
    return ["/tmp/rsna_cache"]


def resolve_cache_dir(cfg=None, need_bytes=0, candidates=None, verbose=True):
    """Pick a cache root that measurably has room, or return the configured one.

    Call this ONCE, before the resolution suffix is appended:

        cfg.cache_dir = resolve_cache_dir(cfg, need_bytes=estimate_cache_bytes(
            n_studies, cfg.in_channels, cfg.image_size))

    Honours cfg.cache_dir when it is not the shipped default (an explicit
    choice is never second-guessed) and when cfg.cache_auto_locate is False.
    """
    configured = str(_cfg(cfg, "cache_dir", "/tmp/rsna_cache")) if cfg is not None \
        else "/tmp/rsna_cache"
    if cfg is not None and not bool(_cfg(cfg, "cache_auto_locate", True)):
        return configured
    if candidates is None:
        candidates = list(default_cache_candidates())
    # An explicitly-configured directory is the first thing tried, not ignored.
    if configured not in candidates:
        candidates = [configured] + candidates

    report = cache_location_report(candidates, need_bytes=need_bytes)
    for rec in report:
        if rec["fits"] and not rec["is_output"]:
            if verbose and rec["path"] != configured:
                print(f"Cache relocated to {rec['path']} "
                      f"({rec['free']/2**30:.1f} GiB free, {rec['fstype'] or 'fs'}) "
                      f"— {configured} could not take "
                      f"{need_bytes/2**30:.2f} GiB")
            return rec["path"]
    for rec in report:                      # last resort: the output directory
        if rec["fits"]:
            if verbose:
                print(f"WARNING: cache falling back to {rec['path']}, which is "
                      "inside /kaggle/working — it will be committed as notebook "
                      "OUTPUT and counts against the ~19.5 GiB output limit "
                      "shared with the model checkpoints.")
            return rec["path"]
    if verbose:
        print(f"WARNING: no cache candidate has room for "
              f"{need_bytes/2**30:.2f} GiB; keeping {configured}. Candidates:")
        for rec in report:
            print(f"    {rec['path']:<34s} free {rec['free']/2**30:7.2f} GiB "
                  f"{rec['fstype'] or '?':<8s}"
                  f"{' RAM-BACKED' if rec['ram_backed'] else ''}"
                  f"{'' if rec['writable'] else ' NOT-WRITABLE'}")
    return configured


# ===========================================================================
# Pre-cache pool
# ===========================================================================

def _precache_worker(args):
    """Top-level so multiprocessing can pickle it.

    'unwritten' is distinct from 'miss': the slices were produced but did not
    reach the disk, which is what a full filesystem looks like from in here
    (_atomic_write swallows ENOSPC and returns False) and also what
    cache_reject_zero does to an all-black stack. Both mean this study will be
    re-decoded on every epoch, so both have to be visible.
    """
    study_id, image_dir, in_channels, cache_dir, cfg = args
    try:
        cpath = cache_path_for(cache_dir, study_id, in_channels, cfg)
        if os.path.exists(cpath):
            return "hit"
        _, status = load_study_slices(study_id, image_dir, cfg,
                                      cache_dir=cache_dir, in_channels=in_channels)
        if status == "miss" and not os.path.exists(cpath):
            return "unwritten"
        return status
    except Exception:
        return "error"


def precache_budget(n_todo, cfg, cache_dir):
    """Disk arithmetic for a pre-cache run. Pure except for stat() calls.

    Returns a dict with `need`, `free`, `total`, `fstype`, `ram_backed`,
    `is_output`, `reserve` and `shortfall` — everything the caller needs to
    decide, and everything the log needs to be actionable.
    """
    C = int(_cfg(cfg, "in_channels", 3))
    size = int(_cfg(cfg, "image_size", 224))
    idx = measured_index_bytes(cache_dir)
    need = estimate_cache_bytes(
        n_todo, C, size,
        index_bytes_per_study=(_INDEX_BYTES_PER_STUDY if idx is None
                               else int(idx)))
    reserve = int(float(_cfg(cfg, "cache_reserve_gb", 1.0)) * 2 ** 30)
    cap = float(_cfg(cfg, "cache_max_gb", 0.0))
    # statvfs needs a path that exists; the budget is usually wanted BEFORE the
    # directory is created, so ask the nearest existing ancestor — it is on the
    # same filesystem by construction.
    probe = _nearest_existing(cache_dir)
    try:
        usage = shutil.disk_usage(probe)
        free, total = usage.free, usage.total
    except OSError:
        free = total = 0
    fstype = _fs_type(probe)
    return {
        "n_todo": int(n_todo), "in_channels": C, "image_size": size,
        "index_bytes": _INDEX_BYTES_PER_STUDY if idx is None else int(idx),
        "index_measured": idx is not None,
        "need": need, "free": free, "total": total, "reserve": reserve,
        "cap_bytes": int(cap * 2 ** 30) if cap > 0 else 0,
        "fstype": fstype, "ram_backed": fstype in _RAM_BACKED,
        "is_output": os.path.realpath(cache_dir).startswith("/kaggle/working"),
        "shortfall": max(0, need + reserve - free),
    }


def precache_dataset(df, image_dir, cfg, cache_dir="/tmp/rsna_cache",
                     progress_every=0.05):
    """Warm the whole cache up front so even epoch 1 runs at cache speed.

    At 4,407 studies this is no longer a formality:

      * It is RESUMABLE and always was — a study whose .npy already exists is
        skipped with one stat() — but nothing said so, so a re-run looked like
        a repeat of the whole cost. The 'hit' count now reports it explicitly.
      * It PRICES ITSELF FIRST. estimate_cache_bytes is exact on the pixel term
        and measured on the index term, so the log says how many GiB this will
        take and how many the filesystem has, before any of it is written.
      * It WATCHES FREE SPACE while it runs and stops cleanly when the reserve
        is breached, instead of silently writing nothing. _atomic_write returns
        False on ENOSPC and load_study_slices ignores that, so a full disk used
        to be invisible: every study would report 'miss' and be re-decoded from
        DICOM on every single epoch for the rest of the session.
      * It reports PROGRESS and an ETA, because a silent multi-minute block
        before epoch 1 is indistinguishable from a hang.
    """
    os.makedirs(cache_dir, exist_ok=True)
    study_ids = df['StudyInstanceUID'].astype(str).unique().tolist()
    C = int(_cfg(cfg, "in_channels", 3))

    # --- resume: what is already on disk costs nothing to re-check ----------
    todo, already = [], 0
    for sid in study_ids:
        if os.path.exists(cache_path_for(cache_dir, sid, C, cfg)):
            already += 1
        else:
            todo.append(sid)

    budget = precache_budget(len(todo), cfg, cache_dir)
    print(f"Pre-cache: {len(study_ids)} studies "
          f"({already} already cached, {len(todo)} to build) at "
          f"{budget['in_channels']}x{budget['image_size']}x{budget['image_size']} "
          f"-> {budget['need']/2**30:.2f} GiB needed, "
          f"{budget['free']/2**30:.2f} GiB free on "
          f"{cache_dir} ({budget['fstype'] or 'fs'})")
    if budget["ram_backed"]:
        print(f"  WARNING: {cache_dir} is {budget['fstype']} — RAM-BACKED. Every "
              f"cached byte is charged to the same system memory the training "
              f"process needs. Move the cache with "
              f"cfg.cache_dir = kaggle_data.resolve_cache_dir(cfg, need_bytes=...)")
    if budget["is_output"]:
        print(f"  WARNING: {cache_dir} is inside /kaggle/working, so all "
              f"{len(study_ids)} .npy files plus their JSON sidecars will be "
              f"committed as notebook OUTPUT against the ~19.5 GiB limit.")
    if budget["shortfall"] > 0:
        print(f"  WARNING: short by {budget['shortfall']/2**30:.2f} GiB against a "
              f"{budget['reserve']/2**30:.2f} GiB reserve. Pre-caching will stop "
              f"when the reserve is reached and the rest will be decoded live.")
        if bool(_cfg(cfg, "cache_require_space", False)):
            raise SystemExit(
                f"FATAL: cache_require_space=True and {cache_dir} cannot hold "
                f"{budget['need']/2**30:.2f} GiB. Free space or set "
                f"cfg.cache_dir to a filesystem that can.")

    counts = {"hit": already}
    if not todo:
        print("Pre-cache: nothing to do (cache already complete).")
        return counts

    args = [(sid, image_dir, C, cache_dir, cfg) for sid in todo]
    # available_cpus(), not cpu_count(): inside a cgroup-limited Kaggle
    # container cpu_count() reports the HOST's cores, and a pool sized from
    # that oversubscribes a 2-vCPU box by 8x-48x. The cap keeps a 96-core dev
    # box from thrashing the page cache.
    n_workers = max(1, min(available_cpus(), 8))
    print(f"Pre-caching {len(todo)} studies using {n_workers} CPU cores "
          f"(order={_cfg(cfg,'slice_order','geometric')}, "
          f"select={_cfg(cfg,'slice_selection','plane_balanced')}, "
          f"trim={_cfg(cfg,'slice_trim_frac',0.0)})...")

    reserve = budget["reserve"]
    cap_bytes = budget["cap_bytes"]
    written_budget = budget["need"] // max(1, len(todo))
    state = {"done": 0, "stopped": None, "t0": time.time(),
             "next_report": max(1, int(len(todo) * float(progress_every)))}

    def _tick(status):
        counts[status] = counts.get(status, 0) + 1
        state["done"] += 1
        done = state["done"]
        if done >= state["next_report"] or done == len(todo):
            el = time.time() - state["t0"]
            rate = done / el if el > 0 else 0.0
            eta = (len(todo) - done) / rate if rate > 0 else 0.0
            print(f"    {done}/{len(todo)} ({100.0*done/len(todo):4.1f}%) "
                  f"{rate:6.1f} studies/s  elapsed {el/60:5.1f} min  "
                  f"ETA {eta/60:5.1f} min")
            state["next_report"] = done + max(1, int(len(todo) * float(progress_every)))
        # Free-space guard: cheap (one statvfs) and only every 64 studies, but
        # it is the difference between "the cache stopped growing" and "the
        # session is now silently decoding DICOMs on every epoch".
        if done % 64 == 0 and (reserve > 0 or cap_bytes > 0):
            try:
                free = shutil.disk_usage(cache_dir).free
            except OSError:
                return True
            if free < reserve + written_budget:
                state["stopped"] = (
                    f"free space fell to {free/2**30:.2f} GiB, below the "
                    f"{reserve/2**30:.2f} GiB reserve")
                return False
            if cap_bytes:
                grown = estimate_cache_bytes(
                    done, C, budget["image_size"],
                    index_bytes_per_study=budget["index_bytes"])
                if grown >= cap_bytes:
                    state["stopped"] = (
                        f"cache_max_gb={cap_bytes/2**30:.2f} GiB reached")
                    return False
        return True

    def serial():
        for a in args:
            if not _tick(_precache_worker(a)):
                break

    if n_workers == 1:
        serial()
    else:
        # Prefer 'fork' where it exists (Linux, i.e. Kaggle): a spawned worker
        # re-imports this module AND torch, which costs more than the whole
        # pre-cache on a 2-vCPU box. Falling back to the default context keeps
        # macOS working, and falling back to serial keeps a caller that has no
        # `if __name__ == "__main__":` guard working instead of crashing
        # (spawn re-runs the parent script, which raises before any work).
        try:
            ctx = multiprocessing.get_context("fork")
        except ValueError:
            ctx = multiprocessing.get_context()
        try:
            with ctx.Pool(n_workers) as pool:
                it = pool.imap_unordered(_precache_worker, args, chunksize=4)
                for s in it:
                    if not _tick(s):
                        pool.terminate()
                        break
        except Exception as exc:
            print(f"Pre-cache pool unavailable ({type(exc).__name__}: {exc}); "
                  "falling back to a single process.")
            counts.clear()
            counts["hit"] = already
            state.update(done=0, stopped=None, t0=time.time(),
                         next_report=max(1, int(len(todo) * float(progress_every))))
            serial()

    bad = counts.get("nodir", 0) + counts.get("nofiles", 0) + counts.get("error", 0)
    unwritten = counts.get("unwritten", 0)
    elapsed = time.time() - state["t0"]
    print(f"Pre-caching complete in {elapsed/60:.1f} min: {counts}")
    if state["stopped"]:
        print(f"  STOPPED EARLY: {state['stopped']}. "
              f"{len(todo) - state['done']} studies were left uncached and will "
              f"be decoded from DICOM on every epoch.")
    if bad:
        print(f"WARNING: {bad}/{len(study_ids)} studies produced no usable slices.")
    if unwritten:
        print(f"WARNING: {unwritten} studies decoded but did NOT reach the cache. "
              f"Either the filesystem is full (ENOSPC is swallowed by the atomic "
              f"write) or the stack was all-zero and cache_reject_zero refused it. "
              f"Each one will be re-decoded on every epoch.")
    return counts


class RSNADataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str, cfg, is_train: bool = True,
                 cache_dir: str = "/tmp/rsna_cache"):
        self.df = df
        self.image_dir = image_dir
        self.cfg = cfg
        self.is_train = is_train
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.slice_consistent = bool(_cfg(cfg, "aug_slice_consistent", True))
        # cache_slices=False bypasses the pixel cache entirely; the only reason
        # to do that is to let aug_slice_jitter_frac actually vary the slice
        # window per epoch (see _load_slices).
        self.use_cache = bool(_cfg(cfg, "cache_slices", True))
        self.cache_stats = {}          # {'hit'|'miss'|'nodir'|'nofiles': n}
        # Slice jitter and the pixel cache are mutually exclusive by
        # construction: a cached study has exactly one frozen slice set.
        self._jitter = (float(_cfg(cfg, "aug_slice_jitter_frac", 0.10))
                        if (is_train and not self.use_cache
                            and bool(_cfg(cfg, "aug_enabled", True))) else 0.0)
        self.transform = self.get_transforms()

        # --- laterality mirror: pixels and labels move together, or not at all
        self._swap_names = None
        self.lateral_swap_perm = None
        self.lateral_pairs = []
        self.lateral_blockers = []
        # Validation/test never mirrors: there is nothing to gain and the
        # laterality columns would be scored against mirrored pixels.
        _aug_on = is_train and bool(_cfg(cfg, "aug_enabled", True))
        self._hflip_p_requested = float(_cfg(cfg, "aug_hflip_p", 0.0)) if _aug_on else 0.0
        self.hflip_p = 0.0
        self._resolve_lateral_swap(list(KNEE_TARGETS))
        self._bind_frame(df)

    # -- fork-friendly view of the frame ------------------------------------
    #
    # WHY.  Every DataLoader worker is a fork of this object, and the frame is
    # no longer 58 rows: it is ~4,407, and it carries object-dtype columns
    # (StudyInstanceUID, and 'Report' when train_csv=train.csv — 6.7 MB of
    # Python str objects for the report text alone). Copy-on-write does not
    # protect against that, because CPython dirties a whole page to bump one
    # refcount, so a worker that walks the frame pays for pages, not bytes.
    # Measured on the real 4,407-row frame, four forked workers each walking
    # every row: 16.7 MB of private memory per worker with the report column,
    # 9.2 MB without.
    #
    # __getitem__ needs exactly two things from those 18 columns: one study id
    # and one row of labels. Both are pulled out here, in the parent, into
    # single contiguous numpy buffers ('S'/'U' for the ids, float32 for the
    # labels). A worker then touches ONE object per array instead of 4,407,
    # never touches the report text at all, and drops the per-item pandas cost
    # (measured 0.072-0.180 ms/sample, 8-18% of a warm-cache item).
    #
    # self.df is retained: nothing outside this class reads it today, but it
    # costs nothing as long as no forked worker touches it, and re-deriving
    # from it is what keeps the KNEE_TARGETS monkey-patch working.

    def _bind_frame(self, df):
        ids = [str(x) for x in df["StudyInstanceUID"].to_numpy()]
        try:
            # ASCII bytes: 64 B/row for a DICOM UID, vs 256 B/row for UTF-32
            # and ~120 B plus a page fault per row for Python str objects.
            self._ids = np.asarray(ids, dtype="S")
            self._ids_are_bytes = True
        except (UnicodeEncodeError, UnicodeDecodeError, SystemError, ValueError):
            self._ids = np.asarray(ids, dtype=np.str_)
            self._ids_are_bytes = False
        self._n = len(ids)
        self._frame_columns = frozenset(str(c) for c in df.columns)
        self._label_targets = None
        self._labels = None
        self._has_labels = False
        self._bind_labels(list(KNEE_TARGETS))

    def _bind_labels(self, targets):
        """(N, L) float32 label matrix for `targets`, or None when absent.

        Bit-identical to what __getitem__ used to compute per row:
        `row[targets].values.astype(np.float32)` then `np.nan_to_num(.., 0.0)`.
        nan_to_num is applied here with the SAME defaults (nan=0.0, and the
        float32 min/max for +-inf, which the per-row call also relied on).
        """
        self._label_targets = list(targets)
        self._has_labels = all(c in self._frame_columns for c in targets)
        if not self._has_labels:
            self._labels = None
            return
        mat = self.df[list(targets)].to_numpy(dtype=np.float32)
        self._labels = np.ascontiguousarray(np.nan_to_num(mat, nan=0.0))

    def _study_id(self, idx):
        sid = self._ids[idx]
        return sid.decode("ascii") if self._ids_are_bytes else str(sid)

    def _resolve_lateral_swap(self, target_names):
        """Resolve the medial<->lateral label permutation for `target_names`.

        Sets self.hflip_p, which is the ONLY thing __getitem__ consults. It stays
        at 0.0 unless a valid permutation exists, so a mirror can never reach the
        model with unmirrored labels.
        """
        self._swap_names = list(target_names)
        perm, pairs, blockers = build_lateral_swap(target_names)
        self.lateral_swap_perm = perm
        self.lateral_pairs = pairs
        self.lateral_blockers = blockers

        requested = self._hflip_p_requested
        if requested <= 0:
            self.hflip_p = 0.0
            return

        if not bool(_cfg(self.cfg, "aug_hflip_require_label_swap", True)):
            self.hflip_p = requested      # explicit opt-out; caller owns this
            return

        if perm is None:
            # No medial/lateral pair to permute -> the mirror is pure label noise.
            warnings.warn(
                "aug_hflip_p>0 but no medial/lateral target pair was found in "
                f"{target_names!r}; a mirror would corrupt laterality labels. "
                "Disabling the horizontal flip.", RuntimeWarning)
            self.hflip_p = 0.0
            return

        self.hflip_p = requested
        if blockers:
            # e.g. MCL: 'medial collateral ligament' has no LCL column to swap
            # into, so a mirrored MCL-positive knee keeps an MCL label while the
            # injured ligament now sits on the lateral side. Pairs still swap
            # correctly; this column degrades. Hence aug_hflip_p defaults to 0.
            warnings.warn(
                f"aug_hflip_p={requested} with side-specific target(s) {blockers!r} "
                "that have no mirror partner column — those labels will be wrong "
                "on mirrored samples even though the medial/lateral pairs swap "
                f"correctly ({pairs!r}).", RuntimeWarning)

    def get_transforms(self):
        in_ch = int(_cfg(self.cfg, "in_channels", 3)) if self.slice_consistent else 1
        return build_transforms(self.cfg, is_train=self.is_train, in_channels=in_ch)

    def __len__(self):
        # self._n, not len(self.df): identical by construction (both are the
        # frame's row count, and the frame is never mutated after __init__),
        # but reading it does not pull the DataFrame's index into a worker.
        return self._n

    def _load_slices(self, study_id):
        """Geometrically ordered slices for one study, as uint8 HxW arrays.

        All of the real work now lives in load_study_slices() so this class and
        the pre-cache pool can never drift apart again.

        NOTE ON aug_slice_jitter_frac.  The old code re-rolled a random slice
        window here, but only on a cache MISS — and train.py calls
        precache_dataset() before the first epoch, so the cache is 100% warm by
        the time any sample is drawn and the jitter never fired.  It was dead
        code.  It is now honest: set cfg.cache_slices=False to trade the cache
        for live per-epoch slice jitter, or leave the cache on (default) and get
        deterministic, reproducible input.
        """
        slices, status = load_study_slices(
            study_id, self.image_dir, self.cfg,
            cache_dir=self.cache_dir if self.use_cache else None,
            in_channels=int(_cfg(self.cfg, "in_channels", 3)),
            use_cache=self.use_cache, jitter=self._jitter)
        self.cache_stats[status] = self.cache_stats.get(status, 0) + 1
        return slices

    def _apply_transform(self, slices):
        """Slices (list of HxW uint8) -> (in_channels, H, W) float tensor.

        With aug_slice_consistent the whole stack goes through ONE transform
        call, so every slice gets the *same* geometric draw. Transforming each
        slice separately (the previous behaviour) re-rolls the RNG per slice and
        leaves the 2.5D channels spatially unregistered — a conv filter then
        sees three mutually shifted/flipped views of the knee in its own
        receptive field.
        """
        if self.slice_consistent:
            stack = np.stack(slices, axis=-1)              # (H, W, C)
            return self.transform(image=stack)['image']    # (C, H, W)
        return torch.cat([self.transform(image=sl)['image'] for sl in slices], dim=0)

    def __getitem__(self, idx):
        # No pandas on this path — see _bind_frame(). self._ids[idx] reads one
        # fixed-width slot out of one contiguous buffer; the old
        # self.df.iloc[idx] built a whole object-dtype Series per sample and
        # touched every block of the frame to do it.
        study_id = self._study_id(idx)

        slices = self._load_slices(study_id)

        # Module-level KNEE_TARGETS is monkey-patched by train.py after CSV
        # column auto-detection, so re-resolve the swap if the list changed.
        targets = list(KNEE_TARGETS)
        if targets != self._swap_names:
            self._resolve_lateral_swap(targets)
        if targets != self._label_targets:
            self._bind_labels(targets)

        # Medial<->lateral mirror, applied here (not inside the Compose) so the
        # label permutation is applied to the SAME sample in the same draw.
        flipped = self.hflip_p > 0 and random.random() < self.hflip_p
        if flipped:
            slices = [np.ascontiguousarray(sl[:, ::-1]) for sl in slices]

        image_tensor = self._apply_transform(slices)
        image_tensor = torch.nan_to_num(image_tensor, nan=0.0, posinf=0.0, neginf=0.0)

        # `self.is_train or all(c in row for c in targets)` — a pandas Series'
        # `in` tests its INDEX, i.e. the frame's columns, so self._has_labels
        # is the same predicate evaluated once instead of per sample.
        if self.is_train or self._has_labels:
            if self._labels is None:
                # is_train with no target columns in the frame: the old code
                # raised a KeyError here. Preserve that — a silent zero label
                # is exactly the failure mode filter_labelled() exists to stop.
                raise KeyError(
                    f"RSNADataset(is_train=True) but the frame has none of the "
                    f"target columns {targets!r}; columns are "
                    f"{sorted(self._frame_columns)!r}")
            labels = self._labels[idx]
            if flipped and self.lateral_swap_perm is not None:
                labels = labels[self.lateral_swap_perm]
            # torch.tensor (not from_numpy) — it COPIES, so the returned tensor
            # can never alias, and later be mutated into, the shared matrix.
            return image_tensor, torch.tensor(labels)
        else:
            # No labels to swap -> a mirror here is unpaired by construction.
            assert not flipped, "laterality mirror applied to an unlabelled sample"
            return image_tensor


# ═══════════════════════════════════════════════════════════════════════════
# Weak supervision from radiology reports  (opt-in, default OFF)
#
# data_subset/train.csv has 4407 rows, every one of them carrying a non-empty
# `Report`.  649 of those studies have DICOM pixels on disk; only 58 carry
# expert labels.  So 591 studies with real MRI AND a radiologist's written
# findings currently contribute nothing -- train.py drops the "Report" column.
#
# src/labels.py turns that text into calibrated soft targets.  This function is
# the ONE hook that brings them into training.  It deliberately mirrors the
# contract of train.py's attach_pseudo_rows():
#
#     combined_df, per_cell_weights = attach_report_label_rows(...)
#
# so a caller composes the two with one line and WeightedDataset consumes the
# result unchanged.
#
# THREE INVARIANTS, in order of importance:
#   1. A gold label is NEVER overwritten.  Report-derived studies arrive as NEW
#      ROWS, and any study already present in `train_df` is excluded outright,
#      so the two can never collide on a cell.
#   2. A report row NEVER reaches validation.  Its `fold` is -1, which no
#      student fold equals, exactly as attach_pseudo_rows does for pseudo rows.
#   3. It is OFF unless cfg.use_report_labels is True.  No other run
#      changes behaviour because this file grew a function.
# ═══════════════════════════════════════════════════════════════════════════

def attach_report_label_rows(train_df, weights, cfg, target_cols,
                             image_dir=None, exclude_ids=None,
                             report_df=None, calibration_folds=None):
    """Append report-derived soft-target rows to one fold's training frame.

    Parameters
    ----------
    train_df : DataFrame
        The fold's training rows (gold, and optionally pseudo rows already
        attached).  Returned unchanged at the head of the result.
    weights : (N, L) array or None
        Per-cell loss weights aligned with `train_df`.  None means all ones.
    cfg : Config
        Reads (all additive, all defaulting to the OFF behaviour):
        use_report_labels, report_csv, report_text_col, report_label_weight,
        report_min_confidence, report_labels_require_images,
        report_label_max_studies, report_label_calibrate_on_folds.
    target_cols : sequence of str
        Label columns, in the order the loss expects them.
    image_dir : str, optional
        If given, studies with no directory on disk are dropped -- a report
        without pixels is not a training sample.
    exclude_ids : iterable, optional
        Extra StudyInstanceUIDs to keep out (e.g. this fold's validation gold).
    report_df : DataFrame, optional
        Pre-loaded report table; skips the CSV read.
    calibration_folds : iterable, optional
        Gold folds to refit the label calibration on.  Leave as None to use the
        constants shipped in src/labels.py.

    Returns
    -------
    (combined_df, weights) : (DataFrame, np.ndarray[float32])
    """
    base = train_df.reset_index(drop=True).copy()
    if weights is None:
        base_w = np.ones((len(base), len(target_cols)), dtype=np.float32)
    else:
        base_w = np.asarray(weights, dtype=np.float32).reshape(len(base), -1)

    if not bool(_cfg(cfg, "use_report_labels", False)):
        return base, base_w

    try:
        from src import labels as weak
    except ImportError:                                   # flat sys.path layout
        import labels as weak                             # type: ignore

    # ---- load the report table -------------------------------------------
    text_col = str(_cfg(cfg, "report_text_col", "Report"))
    if report_df is None:
        path = os.path.join(str(_cfg(cfg, "data_dir", "data_subset")),
                            str(_cfg(cfg, "report_csv", "train.csv")))
        if not os.path.exists(path):
            warnings.warn(f"use_report_labels=True but {path} does not exist; "
                          "training on gold only.", RuntimeWarning)
            return base, base_w
        try:
            import csv as _csv
            _csv.field_size_limit(10 ** 9)   # reports run to 4.7 kB
        except Exception:
            pass
        report_df = pd.read_csv(path, engine="python")

    if text_col not in report_df.columns:
        warnings.warn(f"use_report_labels=True but the report table has no "
                      f"{text_col!r} column; training on gold only.", RuntimeWarning)
        return base, base_w

    rep = report_df[report_df[text_col].fillna("").astype(str).str.strip() != ""].copy()
    rep["StudyInstanceUID"] = rep["StudyInstanceUID"].astype(str)

    # ---- INVARIANT 1: gold always wins -----------------------------------
    # Every study already in this fold's training frame is removed, so a
    # report-derived target can never land on a study that has a real label.
    taken = set(base["StudyInstanceUID"].astype(str))
    if exclude_ids:
        taken |= {str(s) for s in exclude_ids}
    n_before = len(rep)
    rep = rep[~rep["StudyInstanceUID"].isin(taken)]
    n_gold_skipped = n_before - len(rep)

    if bool(_cfg(cfg, "report_labels_require_images", True)) and image_dir \
            and os.path.isdir(image_dir):
        on_disk = {d for d in os.listdir(image_dir) if not d.startswith(".")}
        rep = rep[rep["StudyInstanceUID"].isin(on_disk)]

    cap = int(_cfg(cfg, "report_label_max_studies", 0))
    if cap > 0 and len(rep) > cap:
        rep = rep.iloc[:cap]

    if len(rep) == 0:
        print("  report labels: no eligible studies -- gold only")
        return base, base_w

    # ---- run the rules ----------------------------------------------------
    priors = None
    calibration = None
    folds = calibration_folds if calibration_folds is not None else \
        _cfg(cfg, "report_label_calibrate_on_folds", None)
    if isinstance(folds, str):
        # Config keeps this a str so `--set report_label_calibrate_on_folds=0,1,2`
        # survives Config.from_args' orig_type(v) conversion.
        folds = [int(x) for x in folds.replace(" ", "").split(",") if x != ""]
    if folds:
        gold_path = os.path.join(str(_cfg(cfg, "data_dir", "data_subset")),
                                 str(_cfg(cfg, "train_csv", "train_gold.csv")))
        if os.path.exists(gold_path):
            gold = pd.read_csv(gold_path, engine="python")
            priors = weak.calibrate_priors(gold, folds=folds)
            calibration = weak.calibrate_states(gold, priors=priors, folds=folds)
            print(f"  report labels: calibration refit on gold folds {list(folds)}")

    labels_out = weak.label_dataframe(rep, text_col=text_col,
                                      labels=list(target_cols),
                                      priors=priors, calibration=calibration)

    w_scale = float(_cfg(cfg, "report_label_weight", 0.35))
    w_min = float(_cfg(cfg, "report_min_confidence", 0.10))

    n = len(rep)
    vals = np.zeros((n, len(target_cols)), dtype=np.float32)
    cell_w = np.zeros((n, len(target_cols)), dtype=np.float32)
    state_counts = {}
    for j, col in enumerate(target_cols):
        if col not in labels_out.columns:
            # Unknown target: no opinion, weight 0 -> the loss never sees it.
            continue
        vals[:, j] = labels_out[col].to_numpy(dtype=np.float32)
        conf = labels_out[col + "__conf"].to_numpy(dtype=np.float32)
        cell_w[:, j] = np.where(conf >= w_min, w_scale * conf, 0.0)
        st = labels_out[col + "__state"]
        state_counts[col] = (int((st == "positive").sum()),
                             int((st == "weak_positive").sum()),
                             int((st == "unmentioned").sum()))

    rows = pd.DataFrame({"StudyInstanceUID": rep["StudyInstanceUID"].values})
    for j, col in enumerate(target_cols):
        rows[col] = vals[:, j]
    rows["is_report"] = True
    rows["fold"] = -1                     # INVARIANT 2: never a validation row

    for col in base.columns:
        if col not in rows.columns:
            rows[col] = True if col == "is_report" else np.nan
    if "is_report" not in base.columns:
        base = base.copy()
        base["is_report"] = False
        rows["is_report"] = True
    rows = rows[base.columns]

    combined = pd.concat([base, rows], ignore_index=True)
    out_w = np.concatenate([base_w, cell_w], axis=0).astype(np.float32)

    live = int((cell_w > 0).sum())
    print(f"  + {len(rows)} report-labelled studies "
          f"({n_gold_skipped} skipped because they already carry gold labels); "
          f"{live}/{cell_w.size} cells above confidence {w_min}, "
          f"scale {w_scale}; gold rows keep weight 1.0")
    for col, (p, wp, un) in state_counts.items():
        print(f"      {col:<18s} positive={p:4d} weak={wp:4d} unmentioned={un:4d}")
    return combined, out_w
