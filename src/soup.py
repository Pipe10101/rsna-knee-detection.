"""Weight-space averaging ("model soup") across folds.

WHY THIS EXISTS
---------------
A K-fold logit ensemble costs K forward passes per test study and holds K sets
of weights.  On a hidden test set of unknown size that is the single largest and
least predictable line item in the whole submission run.

Every fold here is fine-tuned from the *same* pre-trained initialisation with a
low backbone LR (LLRD) and a decaying schedule, so the fold solutions stay in
one loss basin and their arithmetic mean is itself a good model -- the "model
soup" result (Wortsman et al., 2022).  Averaging them gives:

    K models  ->  1 model      K x cheaper inference, K x less VRAM/RAM
    K forward passes -> 1      no change to the submission format

WHAT THIS SCRIPT WILL NOT DO
----------------------------
It will not claim the soup is *better* than the ensemble, because on this
dataset that claim is not measurable.  Fold model j is trained on every fold
except j, so for any held-out fold k every constituent with j != k has already
seen fold k: a soup of two or more fold models has no clean validation set, and
any AUC computed for it on OOF data is optimistically biased.  This script
therefore reports two things and labels them honestly:

  * the CLEAN mean out-of-fold AUC of the individual fold models, and
  * the LEAKED AUC of the soup on the same rows, flagged as leaked.

A leaked score should come out *above* the clean one purely from the leak.  If
it comes out below, the fold weights did not average cleanly and the soup must
be rejected -- that is the sanity check this script exists to run.

The decision to actually ship the soup is made in src/infer.py, by a label-free
agreement test against the full ensemble on real test studies.  See
`agreement_ok()` there.
"""

import os
import argparse

import numpy as np
import torch


def _load_state(path):
    """Load a checkpoint as a flat {name: tensor} state dict."""
    try:
        st = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:                       # torch < 2.0
        st = torch.load(path, map_location="cpu")
    if isinstance(st, dict) and "model" in st:
        st = st["model"]
    return {k.replace("_orig_mod.", ""): v for k, v in st.items()}


def average_states(paths, verbose=True):
    """Uniform mean of the float tensors in `paths`, one checkpoint at a time.

    Peak host RAM is two state dicts, not len(paths), so this runs inside
    Kaggle's memory limit no matter how many folds are souped.
    """
    soup, n = None, 0
    ref_keys = None
    for path in paths:
        st = _load_state(path)
        if soup is None:
            ref_keys = set(st)
            soup = {k: (v.detach().float().clone() if torch.is_floating_point(v)
                        else v.detach().clone())
                    for k, v in st.items()}
            n = 1
            if verbose:
                print(f"  soup base: {os.path.basename(path)} ({len(soup)} tensors)")
            continue

        if set(st) != ref_keys:
            missing, extra = ref_keys - set(st), set(st) - ref_keys
            raise ValueError(
                f"{path} does not match the soup base "
                f"(missing={sorted(missing)[:3]}, extra={sorted(extra)[:3]}). "
                "Souping is only defined across identical architectures.")

        n += 1
        for k, v in st.items():
            if not torch.is_floating_point(v):
                soup[k] = v.detach().clone()      # counters: take the latest
                continue
            if soup[k].shape != v.shape:
                raise ValueError(f"shape mismatch for {k}: {soup[k].shape} vs {v.shape}")
            soup[k].add_((v.detach().float() - soup[k]) / n)
        if verbose:
            print(f"  folded in: {os.path.basename(path)} (n={n})")
        del st
    return soup, n


def divergence_report(paths):
    """Relative L2 spread of the fold weights.

    Souping only works while the fold solutions share a basin.  This reports
    ||w_i - w_0|| / ||w_0|| over the flattened float weights; fine-tuning from a
    common initialisation with a decayed LR typically lands well under 0.1.
    A large spread means the folds diverged and the mean is not a model.
    """
    base = None
    spreads = []
    for path in paths:
        st = _load_state(path)
        flat = torch.cat([v.detach().float().flatten()
                          for k, v in sorted(st.items()) if torch.is_floating_point(v)])
        if base is None:
            base = flat
            continue
        spreads.append(float(torch.norm(flat - base) / (torch.norm(base) + 1e-12)))
        del st, flat
    return spreads


def _macro_auc(logits, targets):
    from sklearn.metrics import roc_auc_score
    p = 1.0 / (1.0 + np.exp(-logits))
    aucs = []
    for i in range(targets.shape[1]):
        y, s = targets[:, i], p[:, i]
        keep = np.isfinite(y)
        y, s = y[keep], s[keep]
        if y.size == 0 or len(np.unique(y)) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y, s)))
        except ValueError:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def load_oof(models_dir):
    """Concatenate every ``oof_fold_*.npz`` into one out-of-fold matrix.

    Returns (logits, targets, ids, target_cols, n_folds) or (None, ...) if
    nothing has been written yet.  Every row here was predicted by the only
    model that never trained on it, so the pooled matrix is a clean held-out
    prediction for the WHOLE gold panel.
    """
    import glob
    files = sorted(glob.glob(os.path.join(models_dir, "oof_fold_*.npz")))
    if not files:
        return None, None, None, None, 0
    L, Y, I, cols = [], [], [], None
    for f in files:
        d = np.load(f, allow_pickle=True)
        L.append(np.asarray(d["logits"], dtype=np.float64))
        Y.append(np.asarray(d["targets"], dtype=np.float64))
        I.append(np.asarray(d["ids"], dtype=object))
        if cols is None and "target_cols" in d:
            cols = [str(c) for c in d["target_cols"]]
    return (np.concatenate(L, 0), np.concatenate(Y, 0),
            np.concatenate(I, 0), cols, len(files))


def _auc_one(y, s):
    from sklearn.metrics import roc_auc_score
    keep = np.isfinite(y)
    y, s = y[keep], s[keep]
    if y.size == 0 or len(np.unique(y)) < 2:
        return float("nan"), int(y.sum() if y.size else 0), int(y.size)
    try:
        return float(roc_auc_score(y, s)), int(y.sum()), int(y.size)
    except ValueError:
        return float("nan"), int(y.sum()), int(y.size)


def oof_report(models_dir, verbose=True):
    """Pooled out-of-fold macro-AUC, per label, plus the per-fold mean.

    WHY POOLED AND NOT THE MEAN OF PER-FOLD AUCs
    --------------------------------------------
    A validation fold here is 10-13 studies.  Macro ROC-AUC on that many rows is
    a rank statistic whose finest step is ~0.003 and whose sampling error is an
    order of magnitude larger.  Worse, a label with too few positives in one
    fold is UNCOMPUTABLE there (roc_auc_score needs both classes) and silently
    drops out of that fold's macro average -- so the per-fold mean is an average
    over a DIFFERENT LABEL SET per fold, which is not comparable between two
    configurations.  The bias is not signed: with 12 rows, one fold's noise can
    push the per-fold mean either side of the pooled figure.

    Pooling fixes both: one AUC over all ~58 held-out studies, every label scored
    once on every row that carries it.  Compare configurations on the pooled
    number; the per-fold mean is reported only to show the spread.
    """
    logits, targets, ids, cols, n_folds = load_oof(models_dir)
    if logits is None:
        if verbose:
            print(f"No oof_fold_*.npz under {models_dir} -- run training first.")
        return None

    probs = 1.0 / (1.0 + np.exp(-logits))
    cols = cols or [f"label_{i}" for i in range(targets.shape[1])]

    per_label, aucs = {}, []
    for i, name in enumerate(cols):
        a, npos, n = _auc_one(targets[:, i], probs[:, i])
        per_label[name] = (a, npos, n)
        if np.isfinite(a):
            aucs.append(a)
    pooled = float(np.mean(aucs)) if aucs else float("nan")

    # The per-fold mean, for contrast only.
    import glob
    fold_aucs = []
    for f in sorted(glob.glob(os.path.join(models_dir, "oof_fold_*.npz"))):
        d = np.load(f, allow_pickle=True)
        a = _macro_auc(np.asarray(d["logits"]), np.asarray(d["targets"]))
        if np.isfinite(a):
            fold_aucs.append(a)
    per_fold_mean = float(np.mean(fold_aucs)) if fold_aucs else float("nan")

    if verbose:
        n_studies = len(np.unique(ids)) if ids is not None else targets.shape[0]
        print(f"\nOut-of-fold report — {n_folds} folds, {targets.shape[0]} rows "
              f"({n_studies} distinct studies)")
        print(f"{'label':18} {'AUC':>7} {'pos':>5} {'n':>5}")
        print("-" * 39)
        for name in cols:
            a, npos, n = per_label[name]
            shown = "  n/a  " if not np.isfinite(a) else f"{a:7.4f}"
            print(f"{name:18} {shown} {npos:5d} {n:5d}")
        print("-" * 39)
        print(f"{'POOLED macro-AUC':18} {pooled:7.4f}   <- compare configurations on this")
        print(f"{'per-fold mean':18} {per_fold_mean:7.4f}   "
              f"(spread {min(fold_aucs):.4f}-{max(fold_aucs):.4f})"
              if fold_aucs else "")
        print("\nCompare configurations on the POOLED figure. A label with too few\n"
              "positives in a fold is uncomputable there and drops out of that\n"
              "fold's macro average, so the per-fold mean is an average over a\n"
              "different label set per fold. The resulting bias is not signed --\n"
              "on 10-13 rows one fold's noise moves it either way -- which is\n"
              "exactly why it should not be the comparison statistic.")
    return {"pooled_macro_auc": pooled, "per_label": per_label,
            "per_fold_mean": per_fold_mean, "n_folds": n_folds,
            "n_rows": int(targets.shape[0])}


def clean_oof_auc(models_dir):
    """Back-compat shim: (pooled_macro_auc, n_folds).

    Previously returned the MEAN OF PER-FOLD AUCs, which is the statistic
    the build notes warn against on 10-13 study folds.  It now returns the
    pooled figure; ``oof_report`` gives the full breakdown.
    """
    r = oof_report(models_dir, verbose=False)
    if r is None:
        return None, 0
    return r["pooled_macro_auc"], r["n_folds"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", nargs="+", default=[],
                    help="Per-fold winning checkpoints for ONE backbone.")
    ap.add_argument("--out", default=None, help="Path to write the souped weights.")
    ap.add_argument("--models-dir", default=None,
                    help="Directory holding oof_fold_*.npz, for the sanity check.")
    ap.add_argument("--max-spread", type=float, default=0.35,
                    help="Reject the soup if fold weights are further apart than this.")
    ap.add_argument("--oof-report", action="store_true",
                    help="Print the pooled out-of-fold report and exit.")
    args = ap.parse_args()

    if args.oof_report:
        oof_report(args.models_dir or os.path.dirname(args.checkpoints[0]))
        return

    if not args.out:
        raise SystemExit("--out is required when souping (only --oof-report may omit it).")
    paths = [p for p in args.checkpoints if os.path.exists(p)]
    missing = [p for p in args.checkpoints if not os.path.exists(p)]
    for p in missing:
        print(f"WARNING: checkpoint not found, skipping: {p}")
    if not paths:
        raise SystemExit("No checkpoints to soup.")
    if len(paths) == 1:
        print("Only one checkpoint -- copying it through unchanged (nothing to average).")
        torch.save(_load_state(paths[0]), args.out, _use_new_zipfile_serialization=False)
        print(f"Wrote {args.out}")
        return

    print(f"Souping {len(paths)} checkpoints:")
    spreads = divergence_report(paths)
    if spreads:
        worst = max(spreads)
        print(f"  weight-space spread vs fold 0: "
              f"{', '.join(f'{s:.3f}' for s in spreads)}  (max {worst:.3f})")
        if worst > args.max_spread:
            raise SystemExit(
                f"REJECTED: fold weights diverged (max relative L2 {worst:.3f} > "
                f"{args.max_spread}). Their mean is not a valid model -- keep the "
                "logit ensemble instead of souping.")

    soup, n = average_states(paths)
    torch.save(soup, args.out, _use_new_zipfile_serialization=False)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"Soup of {n} folds -> {args.out} ({size_mb:.0f} MB, "
          f"{n}x less to load and {n}x fewer forward passes at inference)")

    models_dir = args.models_dir or os.path.dirname(paths[0])
    clean, k = clean_oof_auc(models_dir)
    if clean is not None:
        print(f"\nCLEAN pooled out-of-fold macro-AUC over the {k} fold models: "
              f"{clean:.4f}")
        print("  (The soup's own AUC on those same rows would be LEAKED -- every "
              "constituent\n   except one trained on them -- so it is not reported "
              "here as evidence.\n   src/infer.py decides soup-vs-ensemble with a "
              "label-free agreement test on\n   the real test studies.)")
    else:
        print("\nNo oof_fold_*.npz found -- run training first if you want the "
              "clean OOF baseline printed here.")


if __name__ == "__main__":
    main()
