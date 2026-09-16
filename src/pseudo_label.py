"""Out-of-fold pseudo-label generation for the RSNA knee pipeline.

WHY THIS FILE IS SHAPED THE WAY IT IS
=====================================

The naive version of this step -- "average all 15 checkpoints over the 4,349
unlabelled studies, dump one probability per study, retrain on everything" --
has two defects that both inflate cross-validation AUC without improving the
model.  Both are fixed here.

1. FOLD LEAKAGE.
   Teacher checkpoint ``fold_f_best.pt`` was trained on gold folds != f and
   validated on gold fold f.  If we average ALL folds' checkpoints into one
   pseudo-label per study, that pseudo-label carries information from every
   gold study, including the ones the *student* will validate on.  The student
   then trains on targets that were distilled from its own validation set:
   validation AUC climbs, true generalisation does not.  This is the classic
   pseudo-label leak and it is invisible in the logs.

   Fix: pseudo-labels are generated PER TEACHER FOLD.  For teacher fold f we
   ensemble only the checkpoints whose training set excluded gold fold f.  The
   output CSV is therefore LONG: one row per (StudyInstanceUID, teacher_fold).
   ``src/train.py`` training fold f consumes only ``teacher_fold == f`` rows,
   so nothing derived from gold fold f can ever reach fold f's training loss.

2. NO CONFIDENCE FILTERING.
   The teacher saw 58 studies.  Most of its guesses sit near the label prior,
   i.e. near chance.  Accepting all of them buries the 58 real labels under
   hundreds of near-chance targets and the student learns to imitate the
   teacher's noise (confirmation-bias amplification).

   Fix: a per-cell confidence filter.  A cell survives only if
   ``|p - 0.5| >= cfg.pseudo_conf_margin``; optionally only the top-k most
   confident positives/negatives per (fold, label) survive.  Filtered cells are
   written as NaN, and ``src/train.py`` gives NaN cells weight 0 so they
   contribute nothing to the loss.  Surviving cells keep their SOFT
   probability (never thresholded to 0/1) and are down-weighted to
   ``cfg.pseudo_weight`` relative to gold's 1.0.

   The filter is computed independently within each teacher fold, so the
   threshold itself cannot smuggle cross-fold information either.

A NOTE ON TERMINOLOGY
---------------------
docs/architecture_guide.md calls this "Knowledge Distillation".  It is not.
Knowledge distillation moves information from a STRONGER teacher into a
smaller/faster student; the soft targets help because the teacher genuinely
knows more than the hard labels convey.  Here the teacher is trained on 58
studies and is *weaker* than the task demands, so its soft outputs carry the
teacher's uncertainty, not extra knowledge.  The correct name is
self-training / pseudo-labelling, and the honest expectation is "may help a
little via extra input-space regularisation, may hurt", not "the secret to
maximizing AUC".

USAGE
-----
    python3 -m src.pseudo_label \
        --checkpoints models/effnetv2m_384/fold_0_best.pt ... \
        --backbones  tf_efficientnetv2_m.in21k_ft_in1k ... \
        --set data_dir=data_subset pseudo_conf_margin=0.30

Fold provenance is read from the checkpoint path (``fold_<N>_``).  Pass
``--checkpoint-folds`` to state it explicitly.  With
``pseudo_strict_oof=true`` (default) an un-attributable checkpoint aborts the
run rather than silently producing leaky labels.
"""

import os
import re
import json
import argparse
import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.model import RSNA25DModel
from src.kaggle_data import RSNADataset, KNEE_TARGETS, build_lateral_swap
from src.infer import tta_predict


PSEUDO_FOLD_COL = "teacher_fold"
META_SUFFIX = "_meta.json"


# ══════════════════════════════════════════════════════════════════════════
# Fold provenance
# ══════════════════════════════════════════════════════════════════════════

_FOLD_RE = re.compile(r"fold[_\-]?(\d+)", re.IGNORECASE)


def infer_checkpoint_fold(path):
    """Return the held-out fold encoded in a checkpoint path, or None.

    ``models/dinov2_large/fold_3_best.pt`` -> 3.  The LAST match wins so that a
    directory such as ``models/fold_sweep/`` cannot shadow the file name.
    """
    matches = _FOLD_RE.findall(str(path))
    if not matches:
        return None
    return int(matches[-1])


def group_checkpoints_by_fold(checkpoints, backbones=None, folds=None,
                              n_folds=5, strict=True):
    """Map held-out fold -> [(checkpoint, backbone), ...].

    ``folds[i]`` states which gold fold checkpoint ``i`` was held out from
    (i.e. the fold it validated on and therefore never trained on).  When it is
    not given the fold is parsed out of the path.

    A checkpoint whose fold cannot be determined is fatal under ``strict``: the
    only alternative is to assume it is safe for every fold, which is exactly
    the leak this module exists to prevent.
    """
    checkpoints = list(checkpoints)
    if backbones is None or len(backbones) != len(checkpoints):
        backbones = [None] * len(checkpoints)
    if folds is not None and len(folds) != len(checkpoints):
        raise ValueError(
            f"--checkpoint-folds has {len(folds)} entries for "
            f"{len(checkpoints)} checkpoints")

    grouped, unknown = {}, []
    for i, ckpt in enumerate(checkpoints):
        f = int(folds[i]) if folds is not None else infer_checkpoint_fold(ckpt)
        if f is None or not (0 <= f < n_folds):
            unknown.append(ckpt)
            continue
        grouped.setdefault(f, []).append((ckpt, backbones[i]))

    if unknown:
        msg = ("cannot determine the held-out fold for these checkpoints, so "
               "their predictions cannot be proven out-of-fold:\n  "
               + "\n  ".join(map(str, unknown))
               + "\nName them 'fold_<N>_*.pt' or pass --checkpoint-folds.")
        if strict:
            raise ValueError(msg)
        print("WARNING: " + msg + "\nDropping them (pseudo_strict_oof=false).")

    return grouped


# ══════════════════════════════════════════════════════════════════════════
# Gold fold map -- the contract between teacher and student
# ══════════════════════════════════════════════════════════════════════════

def gold_fold_map(cfg, target_cols=None):
    """Return {StudyInstanceUID: fold} for the gold studies.

    Built with exactly the same helpers ``src/train.py`` uses, so the map the
    teacher was trained under is the map the student validates under.  The map
    is written into the sidecar metadata and re-verified at training time; a
    mismatch means the fold definition drifted and every out-of-fold guarantee
    in this module is void.
    """
    from src.train import (load_dataframe, filter_labelled, drop_missing_images,
                           assign_folds)

    df, cols = load_dataframe(cfg)
    target_cols = target_cols or cols
    df = filter_labelled(df, target_cols, cfg)
    df = drop_missing_images(df, os.path.join(cfg.data_dir, "train_series"))
    df = assign_folds(df, target_cols, cfg)
    return {str(k): int(v) for k, v in
            zip(df["StudyInstanceUID"].astype(str), df["fold"])}, target_cols


# ══════════════════════════════════════════════════════════════════════════
# Confidence filtering
# ══════════════════════════════════════════════════════════════════════════

def apply_confidence_filter(probs, margin=0.30, max_per_label=0):
    """Blank out low-confidence pseudo-label cells.

    ``probs`` is an (N, L) array of probabilities from ONE teacher fold.
    Returns ``(filtered, stats)`` where ``filtered`` is a float64 copy with
    rejected cells set to NaN.  Cells keep their soft probability -- they are
    never rounded to 0/1, because the whole (modest) value of a pseudo-label
    lies in its calibration.

    Two independent gates:
      * ``margin``        -- keep only ``|p - 0.5| >= margin``.
      * ``max_per_label`` -- per label, additionally keep at most k of the most
        confident survivors on each side of 0.5.  Guards against a label where
        the teacher is confidently wrong about hundreds of studies at once.

    Both are computed strictly within this teacher fold, so no cross-fold
    statistic can leak through the threshold.
    """
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError(f"probs must be (N, L), got shape {p.shape}")
    margin = float(margin)
    out = p.copy()

    conf = np.abs(p - 0.5)
    keep = np.isfinite(p) & (conf >= margin)

    k = int(max_per_label or 0)
    if k > 0:
        capped = np.zeros_like(keep)
        for j in range(p.shape[1]):
            col_keep = keep[:, j]
            for side in (p[:, j] >= 0.5, p[:, j] < 0.5):
                idx = np.flatnonzero(col_keep & side)
                if idx.size == 0:
                    continue
                # most confident first
                order = idx[np.argsort(-conf[idx, j], kind="stable")]
                capped[order[:k], j] = True
            keep[:, j] = capped[:, j]

    out[~keep] = np.nan
    n_cells = int(p.size)
    stats = {
        "cells_total": n_cells,
        "cells_kept": int(keep.sum()),
        "cells_kept_frac": float(keep.sum() / n_cells) if n_cells else 0.0,
        "rows_with_any_kept": int((keep.any(axis=1)).sum()),
        "per_label_kept": keep.sum(axis=0).astype(int).tolist(),
        "per_label_kept_pos": ((keep) & (p >= 0.5)).sum(axis=0).astype(int).tolist(),
        "per_label_kept_neg": ((keep) & (p < 0.5)).sum(axis=0).astype(int).tolist(),
    }
    return out, stats


def build_pseudo_frame(study_ids, fold_probs, target_cols,
                       margin=0.30, max_per_label=0):
    """Assemble the long-format pseudo-label table.

    ``fold_probs`` maps teacher_fold -> (N, L) probability array aligned with
    ``study_ids``.  Returns ``(frame, stats_by_fold)``.  Rows whose every cell
    was filtered out are dropped -- an all-NaN row is a row with no supervision
    and would just cost a forward pass.
    """
    study_ids = [str(s) for s in study_ids]
    frames, stats = [], {}
    for fold in sorted(fold_probs):
        filtered, s = apply_confidence_filter(
            fold_probs[fold], margin=margin, max_per_label=max_per_label)
        if filtered.shape != (len(study_ids), len(target_cols)):
            raise ValueError(
                f"fold {fold}: predictions {filtered.shape} do not match "
                f"{len(study_ids)} studies x {len(target_cols)} targets")
        sub = pd.DataFrame(filtered, columns=list(target_cols))
        sub.insert(0, PSEUDO_FOLD_COL, int(fold))
        sub.insert(0, "StudyInstanceUID", study_ids)
        sub = sub[sub[list(target_cols)].notna().any(axis=1)]
        stats[int(fold)] = s
        frames.append(sub)

    if not frames:
        return pd.DataFrame(columns=["StudyInstanceUID", PSEUDO_FOLD_COL,
                                     *target_cols]), stats
    frame = pd.concat(frames, ignore_index=True)
    return frame, stats


def write_pseudo_bundle(out_csv, frame, meta):
    """Write the pseudo CSV plus its sidecar metadata JSON."""
    frame.to_csv(out_csv, index=False)
    meta_path = os.path.splitext(out_csv)[0] + META_SUFFIX
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)
    return meta_path


# ══════════════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════════════

def _load_model(ckpt, backbone, cfg, device):
    model = RSNA25DModel(backbone_name=backbone or cfg.backbone, pretrained=False,
                         in_channels=cfg.in_channels, num_classes=cfg.num_classes)
    state = torch.load(ckpt, map_location="cpu")
    # fold_N_best.pt is a bare state_dict; fold_N_last.pt wraps one under "model".
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_probs(models, loader, device, use_amp, amp_dtype, lateral_swap_perm, desc=""):
    """Mean-of-probabilities over ``models`` with TTA.  Returns (N, L) float64.

    Plain averaging, not rank-averaging: rank-averaging destroys calibration and
    a pseudo-label's only real content is its calibration.
    """
    per_model = []
    for i, model in enumerate(models):
        chunks = []
        for batch in tqdm(loader, desc=f"{desc} model {i + 1}/{len(models)}", leave=False):
            images = batch[0] if not isinstance(batch, torch.Tensor) else batch
            images = images.to(device)
            preds = tta_predict(model, images, device, use_amp, amp_dtype, lateral_swap_perm)
            chunks.append(preds.float().cpu().numpy())
        per_model.append(np.concatenate(chunks, axis=0))
    return np.mean(per_model, axis=0).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to teacher checkpoints.")
    parser.add_argument("--backbones", nargs="+", default=None,
                        help="Backbone name for each checkpoint (same order).")
    parser.add_argument("--checkpoint-folds", nargs="+", type=int, default=None,
                        help="Held-out gold fold of each checkpoint (same order). "
                             "Defaults to parsing 'fold_<N>' out of the path.")
    parser.add_argument("--set", nargs="+", help="Overrides for Config")
    args = parser.parse_args()

    cfg = Config.from_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp else torch.float32

    strict = bool(getattr(cfg, "pseudo_strict_oof", True))

    # ── Gold fold map (the teacher/student contract) ──────────────────────
    fold_map, target_cols = gold_fold_map(cfg)
    if not fold_map:
        print("Error: no gold studies found -- cannot establish fold provenance.")
        return 1
    print(f"Gold fold map: {len(fold_map)} studies, folds "
          f"{sorted(set(fold_map.values()))}")

    # ── Unlabelled studies ────────────────────────────────────────────────
    train_csv = os.path.join(cfg.data_dir, "train.csv")
    if not os.path.exists(train_csv):
        print(f"Error: {train_csv} not found.")
        return 1
    df = pd.read_csv(train_csv)

    cols = [c for c in target_cols if c in df.columns]
    if len(cols) != len(target_cols):
        print(f"Error: train.csv is missing target columns "
              f"{sorted(set(target_cols) - set(cols))}")
        return 1

    unlabelled = df[df[cols].isna().all(axis=1)].copy()
    # A gold study must never be pseudo-labelled: its real label always wins.
    unlabelled = unlabelled[~unlabelled["StudyInstanceUID"].astype(str).isin(fold_map)]

    image_dir = os.path.join(cfg.data_dir, "train_series")
    if os.path.isdir(image_dir):
        on_disk = {d for d in os.listdir(image_dir) if not d.startswith(".")}
        before = len(unlabelled)
        unlabelled = unlabelled[unlabelled["StudyInstanceUID"].astype(str).isin(on_disk)]
        if before != len(unlabelled):
            print(f"Skipping {before - len(unlabelled)} unlabelled studies with no "
                  f"DICOMs on disk (they would be all-black images).")
    unlabelled = unlabelled.reset_index(drop=True)

    if len(unlabelled) == 0:
        print("No usable unlabelled studies. Exiting.")
        return 0
    print(f"Generating pseudo-labels for {len(unlabelled)} unlabelled studies "
          f"x {cfg.n_folds} teacher folds...")

    # ── Group checkpoints by the fold they were held out from ─────────────
    grouped = group_checkpoints_by_fold(
        args.checkpoints, args.backbones, args.checkpoint_folds,
        n_folds=cfg.n_folds, strict=strict)
    grouped = {f: [(c, b) for c, b in v if os.path.exists(c)]
               for f, v in grouped.items()}
    grouped = {f: v for f, v in grouped.items() if v}
    if not grouped:
        print("Error: no usable checkpoints found!")
        return 1

    missing = sorted(set(fold_map.values()) - set(grouped))
    if missing:
        msg = (f"no teacher checkpoint for gold fold(s) {missing}; those "
               "training folds would get no pseudo-labels at all.")
        if strict:
            print("Error: " + msg)
            return 1
        print("WARNING: " + msg)

    for f in sorted(grouped):
        print(f"  teacher fold {f}: {len(grouped[f])} checkpoint(s) "
              f"(trained WITHOUT gold fold {f})")

    # ── Predict, one teacher fold at a time ───────────────────────────────
    cfg.num_classes = len(target_cols)
    import src.kaggle_data as kd
    kd.KNEE_TARGETS = list(target_cols)

    dataset = RSNADataset(unlabelled, image_dir, cfg, is_train=False)
    nw = min(int(getattr(cfg, "num_workers", 4)), os.cpu_count() or 1)
    if list(target_cols) != list(KNEE_TARGETS):
        nw = 0      # monkey-patched targets do not survive a spawned worker
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=max(0, nw), pin_memory=torch.cuda.is_available())

    lateral_swap_perm, _pairs, _blockers = build_lateral_swap(list(target_cols))

    fold_probs = {}
    for f in sorted(grouped):
        models = [_load_model(c, b, cfg, device) for c, b in grouped[f]]
        print(f"\nTeacher fold {f}: ensembling {len(models)} model(s) with "
              f"{cfg.tta_n}-augmentation TTA...")
        fold_probs[f] = predict_probs(models, loader, device, use_amp, amp_dtype,
                                      lateral_swap_perm, desc=f"fold {f}")
        del models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Filter + write ────────────────────────────────────────────────────
    margin = float(getattr(cfg, "pseudo_conf_margin", 0.30))
    cap = int(getattr(cfg, "pseudo_max_per_label", 0))
    frame, stats = build_pseudo_frame(
        unlabelled["StudyInstanceUID"].astype(str).tolist(), fold_probs,
        target_cols, margin=margin, max_per_label=cap)

    out_csv = os.path.join(cfg.data_dir, getattr(cfg, "pseudo_csv", "train_pseudo.csv"))
    meta = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "schema": "long/one-row-per-(study,teacher_fold)",
        "fold_column": PSEUDO_FOLD_COL,
        "n_folds": int(cfg.n_folds),
        "seed": int(cfg.seed),
        "target_cols": list(target_cols),
        "gold_folds": fold_map,
        "checkpoints_by_fold": {str(f): [c for c, _ in v] for f, v in grouped.items()},
        "backbones_by_fold": {str(f): [b for _, b in v] for f, v in grouped.items()},
        "pseudo_conf_margin": margin,
        "pseudo_max_per_label": cap,
        "pseudo_weight": float(getattr(cfg, "pseudo_weight", 0.30)),
        "n_unlabelled_studies": int(len(unlabelled)),
        "filter_stats": {str(k): v for k, v in stats.items()},
        "soft_labels": True,
    }
    meta_path = write_pseudo_bundle(out_csv, frame, meta)

    print(f"\nSaved pseudo-labels: {out_csv} ({len(frame)} rows across "
          f"{len(fold_probs)} teacher folds)")
    print(f"Sidecar metadata:    {meta_path}")
    for f in sorted(stats):
        s = stats[f]
        print(f"  fold {f}: kept {s['cells_kept']}/{s['cells_total']} cells "
              f"({100 * s['cells_kept_frac']:.1f}%) over "
              f"{s['rows_with_any_kept']} studies")
        per_lbl = ", ".join(f"{n}={k}" for n, k in zip(target_cols, s["per_label_kept"]))
        print(f"           per-label kept: {per_lbl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
