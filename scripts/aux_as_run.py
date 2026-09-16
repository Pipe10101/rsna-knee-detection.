#!/usr/bin/env python3
"""Expose one slot's per-slot aux-head logits as a pseudo-run directory.

    python3 scripts/aux_as_run.py <run_dir> <slot_name> <out_dir>

Reads every ``oof_fold_k.npz`` in ``run_dir`` (written by scripts/train_slotknee.py with
the ``aux`` [N, S, 12] + ``slot_names`` keys) and writes ``<out_dir>/oof_fold_k.npz`` whose
``logits`` = ``aux[:, slot, :]``.  The output has exactly the keys compare_arms.py --combine
and stack_oof.py consume (uids, logits, y, w, mask, best_thresholds), so a single slot's
aux head can be scored / stacked like any other arm without touching those scripts.
"""
import glob
import os
import sys

import numpy as np


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3:
        raise SystemExit(__doc__)
    src, slot, dst = argv
    paths = sorted(glob.glob(os.path.join(src, "oof_fold_*.npz")))
    if not paths:
        raise SystemExit("no oof_fold_*.npz under %s" % src)
    os.makedirs(dst, exist_ok=True)
    for p in paths:
        z = np.load(p, allow_pickle=True)
        if "aux" not in z or "slot_names" not in z:
            raise SystemExit("%s has no per-slot aux logits (trained before 2026-09-08?)" % p)
        names = [str(s) for s in z["slot_names"]]
        if slot not in names:
            raise SystemExit("slot %r not in %s (have %s)" % (slot, p, names))
        j = names.index(slot)
        np.savez_compressed(os.path.join(dst, os.path.basename(p)), uids=z["uids"], logits=z["aux"][:, j, :],
                            y=z["y"], w=z["w"], mask=z["mask"], best_thresholds=z["best_thresholds"])
        print(p, "->", dst, "slot", slot, j)


if __name__ == "__main__":
    main()
