#!/usr/bin/env python3
"""Contact sheet + numbers for the joint-centre locator (src/slots.py, ``locate_joint``).

    python3 scripts/locate_joint_report.py --n 40 [--plane sag|cor|both] [--seed 0]
                                           [--out logs/joint_locator.png] [--data-dir data_subset]

For ``--n`` local studies (random with ``--seed``) the central fluid-sensitive
sagittal (and/or coronal) slice is decoded exactly as ``build_study_tensor``
does, ``locate_joint_for_slot`` runs on that slot's ``joint_slice_paths`` (the
centre anchor's T slices + the nearest anchors' centre slices -- all decoded
for the cache anyway) and the panel shows the anchor slice with

    grey dashed   the image-centred 100 mm zoom box (today's crop)
    red           the joint-centred 100 mm zoom box and its centre (+)
    yellow title  the locator fell back to the image centre

plus a thin bone-fraction profile on the right edge (the valley the locator
picked is the red tick).  Printed at the end: fallback fraction, median and
90th-percentile |joint - image centre| in mm, and the locator's ms/study.
The PNG goes to ``--out`` (default ``logs/joint_locator.png``).
"""

import argparse
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src import slots                                  # noqa: E402

PLANE_SLOTS = {"sag": (0, 3), "cor": (1, 4)}          # FS first, T1 fallback


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=os.path.join(REPO, "data_subset"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plane", default="sag", choices=["sag", "cor", "both"])
    ap.add_argument("--zoom-mm", type=float, default=100.0)
    ap.add_argument("--crop-mm", type=float, default=140.0)
    ap.add_argument("--G", type=int, default=3)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--out", default=os.path.join(REPO, "logs", "joint_locator.png"))
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=60)
    return ap.parse_args(argv)


def pick_studies(image_dir, n, seed):
    uids = sorted(e.name for e in os.scandir(image_dir) if e.is_dir())
    if n >= len(uids):
        return uids
    rng = np.random.RandomState(seed)
    return [uids[i] for i in sorted(rng.choice(len(uids), n, replace=False))]


def profile_for_panel(a, spacing, polarity, crop_mm):
    """The locator's own smoothed bone-fraction profile (full-image rows, values 0-1)."""
    an, _ = slots._joint_analysis(a, spacing, crop_mm, polarity, slots.JOINT_WORK_PX, slots.JOINT_BAND)
    if an is None:
        return None
    return an["rows_full"], an["prof"]


def main(argv=None) -> int:
    args = parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    image_dir = None
    for name in ("train_series", "train_images"):
        p = os.path.join(args.data_dir, name)
        if os.path.isdir(p):
            image_dir = p
            break
    if image_dir is None:
        raise SystemExit("no train_images/ under %s" % args.data_dir)
    csv = os.path.join(args.data_dir, "train_series.csv")
    series_df = pd.read_csv(csv) if os.path.isfile(csv) else None
    uids = pick_studies(image_dir, args.n, args.seed)
    planes = ["sag", "cor"] if args.plane == "both" else [args.plane]

    panels = []          # (title, image, spacing, est, prof, polarity, fallback)
    offsets, fallbacks, ms_list, missing = [], [], [], 0
    t_all = time.perf_counter()
    for uid in uids:
        index = slots.index_study(os.path.join(image_dir, uid), series_df)
        sel = slots.select_slices(index, G=args.G, T=args.T)
        for plane in planes:
            slot = next((s for s in PLANE_SLOTS[plane] if s in sel), None)
            if slot is None:
                missing += 1
                panels.append(("%s %s: slot missing" % (uid[-8:], plane), None, None, None, None, None, True))
                continue
            jpaths, jcentre = slots.joint_slice_paths(sel[slot])
            raws = [slots._decode_raw(jp) for jp in jpaths]
            t0 = time.perf_counter()
            est = slots.locate_joint_for_slot(slot, [(a, sp) for a, sp, _ in raws],
                                              crop_mm=args.crop_mm, centre=jcentre)
            ms = (time.perf_counter() - t0) * 1000.0
            ms_list.append(ms)
            a, sp, _ = raws[jcentre]
            if a is None:
                panels.append(("%s %s: decode failed" % (uid[-8:], plane), None, None, None, None, None, True))
                missing += 1
                continue
            fallbacks.append(bool(est.fallback))
            if not est.fallback:
                offsets.append(float(np.hypot(est.dy_mm, est.dx_mm)))
            pol = slots.joint_polarity(slot)
            prof = profile_for_panel(a, sp, pol, args.crop_mm) if sp is not None else None
            title = "%s %s %s %dx%d %.2fmm | conf %.2f %s" % (
                uid[-8:], plane, slots.SLOT_NAMES[slot], a.shape[0], a.shape[1], sp[0] if sp else 0.0,
                est.conf, "FALLBACK" if est.fallback else "(%+.0f,%+.0f)mm" % (est.dy_mm, est.dx_mm))
            panels.append((title, a, sp, est, prof, pol, bool(est.fallback)))

    n_cols = args.cols
    n_rows = int(np.ceil(len(panels) / float(n_cols)))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(panels):]:
        ax.axis("off")
    for ax, (title, a, sp, est, prof, pol, fb) in zip(axes, panels):
        ax.axis("off")
        if a is None:
            ax.set_title(title, fontsize=7, color="red")
            continue
        lo, hi = np.percentile(a, (1, 99))
        ax.imshow(np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1), cmap="gray", vmin=0, vmax=1)
        rows, cols = a.shape
        if sp is not None:
            r0, r1, c0, c1, _ = slots.crop_window(rows, cols, sp, args.zoom_mm)
            ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False,
                                       ec="0.7", ls="--", lw=1.0))
            if not fb:
                r0, r1, c0, c1, _ = slots.crop_window(rows, cols, sp, args.zoom_mm, est.offset_mm)
                ax.add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, ec="r", lw=1.5))
                ax.plot([est.col], [est.row], "r+", ms=14, mew=2)
            if prof is not None:
                rows_full, p = prof
                x0 = cols - 1
                ax.plot(x0 - p * 0.25 * cols, rows_full, color="cyan", lw=1.0, alpha=0.9)
                ax.plot([x0 - 0.25 * cols, x0], [est.row, est.row], color="r" if not fb else "y", lw=1.0)
        ax.set_title(title, fontsize=6.5, color="y" if fb else "w",
                     backgroundcolor="k")
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
    fig.patch.set_facecolor("k")
    plt.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.savefig(args.out, dpi=args.dpi, facecolor="k")

    n = len(fallbacks)
    n_fb = int(sum(fallbacks))
    print("studies: %d | panels: %d | slot missing/undecodable: %d" % (len(uids), len(panels), missing))
    print("fallback: %d / %d = %.1f%%" % (n_fb, n, 100.0 * n_fb / max(1, n)))
    if offsets:
        print("|joint - image centre|: median %.1f mm | p90 %.1f mm | max %.1f mm (n=%d)"
              % (float(np.median(offsets)), float(np.percentile(offsets, 90)), float(np.max(offsets)), len(offsets)))
    if ms_list:
        print("locator: median %.2f ms / slot (G=%d, T=%d -> <= %d slices), max %.2f ms"
              % (float(np.median(ms_list)), args.G, args.T, args.T + slots._JOINT_MAX_OTHER_ANCHORS,
                 float(np.max(ms_list))))
    print("wall %.1f s (incl. header index + decode) | saved %s" % (time.perf_counter() - t_all, args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
