#!/usr/bin/env python3
"""Soup SlotKnee fold checkpoints into ONE inference-ready soup.pt, and report OOF.

Why this wrapper exists instead of calling ``python3 -m src.soup`` directly
(2026-08-24 audit of scripts/train_slotknee_pipeline.sh):

* ``src.soup --oof-report`` RETURNS EARLY after printing the report, so the
  original single-call pipeline never wrote soup.pt at all.
* ``src.soup._load_state`` expects a RAW state dict (or ``{"model": ...}``).
  SlotKnee checkpoints are ``{"state_dict", "hparams", "slot_layout", "args"}``;
  fed directly, ``average_states`` would call ``torch.is_floating_point`` on the
  ``hparams`` dict and crash.  src/soup.py is a separate module, so the
  adaptation lives here: extract the raw state dicts to a temp dir, feed
  ``src.soup.average_states`` / ``divergence_report`` unchanged, then re-wrap the
  mean with fold 0's ``hparams``/``slot_layout``/``args`` so
  scripts/infer_slotknee.py can rebuild the model from soup.pt alone.
* ``src.soup``'s OOF loader reads npz keys ``targets``/``ids`` which
  scripts/train_slotknee.py does not write (it writes ``y``/``w``/``uids``), so
  the pooled report here uses the same w>0 masking as scripts/ablate_slotknee.sh.

POLICY (measured 2026-08-24): the
soup is NOT the default submission.  A 5-member ViT-S T4 ensemble costs ~25 s for
the whole test set (~0.0006 efficiency units at 0.01 AUC ~ 717 s), while souping
across folds can cost real AUC; ship a soup only if a soup-vs-ensemble OOF
measurement puts it within noise.  The efficiency entry is the distilled student.
"""

import argparse
import glob
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def _fold_checkpoints(out_dir):
    paths = sorted(glob.glob(os.path.join(out_dir, "fold_*_best.pt")))
    if not paths:
        raise SystemExit(f"No fold_*_best.pt under {out_dir} -- run training first.")
    return paths


def soup(out_dir, out_path, max_spread=0.35):
    import torch
    from src.soup import average_states, divergence_report

    paths = _fold_checkpoints(out_dir)
    if len(paths) == 1:
        print("Only one fold checkpoint -- copying it through unchanged.")
        shutil.copy2(paths[0], out_path)
        return

    # SlotKnee checkpoints are {"state_dict", "hparams", "slot_layout", "args"};
    # src.soup only understands raw state dicts, so extract them first.
    tmpdir = tempfile.mkdtemp(prefix="slotknee_soup_")
    try:
        wrapper = None
        raw_paths = []
        for p in paths:
            ck = torch.load(p, map_location="cpu")
            if not isinstance(ck, dict) or "state_dict" not in ck:
                raise SystemExit(f"{p} is not a SlotKnee checkpoint (no 'state_dict').")
            if wrapper is None:  # fold 0 provides hparams / slot_layout / args
                wrapper = {k: v for k, v in ck.items() if k != "state_dict"}
            rp = os.path.join(tmpdir, os.path.basename(p))
            torch.save(ck["state_dict"], rp)
            raw_paths.append(rp)
            del ck

        print(f"Souping {len(raw_paths)} fold checkpoints:")
        spreads = divergence_report(raw_paths)
        if spreads:
            worst = max(spreads)
            print("  weight-space spread vs fold 0: "
                  + ", ".join(f"{s:.3f}" for s in spreads) + f"  (max {worst:.3f})")
            if worst > max_spread:
                raise SystemExit(
                    f"REJECTED: fold weights diverged (max relative L2 {worst:.3f} > "
                    f"{max_spread}). Their mean is not a valid model -- keep the "
                    "logit ensemble instead of souping.")
        averaged, n = average_states(raw_paths)
        wrapper["state_dict"] = averaged
        wrapper["souped_from"] = [os.path.basename(p) for p in paths]
        torch.save(wrapper, out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"Soup of {n} folds -> {out_path} ({size_mb:.0f} MB), wrapped with "
              "fold 0's hparams/slot_layout so infer_slotknee.py loads it directly.")
        print("Reminder: measured policy is ensemble > soup unless a soup-vs-ensemble "
              "OOF check says otherwise (see module docstring).")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def report(out_dir):
    """Pooled OOF over oof_fold_*.npz (train_slotknee.py keys: logits/y/w/uids)."""
    from sklearn.metrics import roc_auc_score
    from src.llm_labels import LABELS

    files = sorted(glob.glob(os.path.join(out_dir, "oof_fold_*.npz")))
    if not files:
        raise SystemExit(f"No oof_fold_*.npz under {out_dir} -- run training first.")
    lg, y, w = [], [], []
    for p in files:
        z = np.load(p, allow_pickle=True)
        lg.append(z["logits"]); y.append(z["y"]); w.append(z["w"])
    lg, y, w = np.concatenate(lg), np.concatenate(y), np.concatenate(w)

    print(f"\nOut-of-fold report -- {len(files)} folds, {len(y)} rows "
          "(clean: each row predicted by the one model that never trained on it)")
    print(f"{'label':18} {'AUC':>7} {'pos':>5} {'n':>5}")
    print("-" * 39)
    aucs = []
    for i, name in enumerate(LABELS):
        m = w[:, i] > 0
        yt = (y[m, i] > 0.5).astype(int)
        if len(np.unique(yt)) > 1:
            a = roc_auc_score(yt, lg[m, i])
            aucs.append(a)
            print(f"{name:18} {a:7.4f} {int(yt.sum()):5d} {int(m.sum()):5d}")
        else:
            print(f"{name:18} {'  n/a  '} {int(yt.sum()) if yt.size else 0:5d} {int(m.sum()):5d}")
    print("-" * 39)
    pooled = float(np.mean(aucs)) if aucs else float("nan")
    print(f"{'POOLED macro-AUC':18} {pooled:7.4f}   <- compare configurations on this")
    print("(This is the CLEAN pooled OOF of the fold models. The soup's own AUC on "
          "these rows would be leaked -- every constituent but one trained on them.)")
    return pooled


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="training output dir holding fold_*_best.pt / oof_fold_*.npz")
    ap.add_argument("--out", default=None, help="soup path (default <out_dir>/soup.pt)")
    ap.add_argument("--max-spread", type=float, default=0.35,
                    help="reject the soup if fold weights are further apart than this")
    ap.add_argument("--report", action="store_true",
                    help="print the pooled OOF report instead of souping")
    args = ap.parse_args(argv)

    if args.report:
        report(args.out_dir)
    else:
        soup(args.out_dir, args.out or os.path.join(args.out_dir, "soup.pt"),
             max_spread=args.max_spread)


if __name__ == "__main__":
    main()
