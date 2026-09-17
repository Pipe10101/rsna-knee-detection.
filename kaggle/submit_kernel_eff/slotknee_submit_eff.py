"""EFFICIENCY-ENTRY variant of slotknee-submit: ONE full-fit distilled student, no TTA.

Identical code path; only two defaults differ (checkpoint glob student_*_best.pt, SK_TTA=0), because
the Kaggle UI cannot pass environment variables.  Run on a T4, then submit this kernel's version to
the EFFICIENCY track.

Kaggle submission kernel: SlotKnee-S inference on the hidden test set (no internet).

Attached sources (kernel-metadata.json):
  competition  rsna-knee-abnormality-detection   (test.csv, test_series.csv, test_series/, sample_submission.csv)
  dataset      felipedeleon11/slotknee-code      (src/, scripts/, wheels/ for JPEG DICOM decoders)
  kernel       felipedeleon11/slotknee-train     (models/slotknee_full5/fold_*_best.pt)

Writes /kaggle/working/submission.csv.  A valid file is seeded from sample_submission.csv
first (scripts/infer_slotknee.py does that itself), so a crash still leaves a scoreable file.
"""
import json
import glob
import numpy as np
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
MAX_CKPTS = int(os.environ.get("SK_MAX_CKPTS", "20"))   # multi-view ensembles; each ViT-S member costs ~17 ms/study on a T4
# CPU fallback budget.  Measured on Kaggle's CPU (submit v5 log, 224/G3/T3 = 18 ViT forwards per
# study): decode 1.72 s/study, model 1.57 s/member/study -> 87 ms per ViT forward.  A layout
# with n_slots x G forwards costs 0.087 * n_slots * G s per member; members are capped so the
# projected run fits CPU_BUDGET_H for CPU_TEST_STUDIES studies (g10t1 = 60 forwards -> 5.2 s per
# member -> 2 members ~ 4.4 h; the old 18-forward layout allows 10).
CPU_BUDGET_H = float(os.environ.get("SK_CPU_BUDGET_H", "5.0"))
CPU_TEST_STUDIES = int(os.environ.get("SK_CPU_TEST_STUDIES", "1300"))
CPU_DECODE_S, CPU_FWD_S = 1.72, 0.087


def ckpt_prefix(path):
    """'g10t1_fold_0_best.pt' -> 'g10t1'; 'fold_0_best.pt' -> ''."""
    b = os.path.basename(path)
    i = b.find("fold_")
    return b[:i].rstrip("_") if i > 0 else ""


def group_weight_vector(label, prefixes, stack, n):
    """Convex weights over the n layout groups for one label.

    `stack` is the stack_weights*.json dict ({"prefixes": [...], "labels": {label: [w, ...]}},
    fitted by scripts/stack_oof.py on OOF ranks).  Falls back to equal weights whenever the
    file is absent, the label is missing, any group's prefix is unmatched, or the matched
    weights sum to 0 — equal weights are the previous, validated behaviour.
    """
    import numpy as np
    eq = np.ones(n) / n
    if not stack or label not in stack.get("labels", {}):
        return eq
    try:
        raw = np.array([float(stack["labels"][label][stack["prefixes"].index(p)]) for p in prefixes])
    except (ValueError, IndexError, TypeError):
        return eq
    if not np.isfinite(raw).all() or raw.min() < 0 or raw.sum() <= 0:
        return eq
    return raw / raw.sum()


def walk_find(pred, prune=("train_series", "test_series", "train_images", "test_images")):
    for base, dirs, files in os.walk("/kaggle/input"):
        hit = pred(base, list(dirs), files)      # test BEFORE pruning: the predicate may look for test_series
        if hit:
            return hit
        dirs[:] = [d for d in dirs if d not in prune]   # never descend into the image trees
    return None


def run(cmd, check=True, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, **kw)


def main():
    print("input mounts:", os.listdir("/kaggle/input"), flush=True)
    code = walk_find(lambda b, d, f: b if ("infer_slotknee.py" in f and os.path.basename(b) == "scripts") else None)
    code = os.path.dirname(code) if code else None
    comp = walk_find(lambda b, d, f: b if ("sample_submission.csv" in f and "test_series" in d) else None)
    # prefer the models dataset (explicitly pushed checkpoints) over any training-kernel output
    ckpt_dir = walk_find(lambda b, d, f: b if ("/datasets/" in b and any(x.endswith("_best.pt") for x in f)) else None)
    ckpt_dir = ckpt_dir or walk_find(lambda b, d, f: b if any(x.startswith("fold_") and x.endswith("_best.pt") for x in f) else None)
    print("code:", code, "\ncompetition:", comp, "\ncheckpoints:", ckpt_dir, flush=True)
    if not (code and comp and ckpt_dir):
        raise SystemExit("missing input: code/competition/checkpoints not all found")

    # Seed a scoreable submission IMMEDIATELY: if every layout group later fails (or the
    # merge crashes), /kaggle/working/submission.csv must already exist at 0.5.
    import pandas as pd
    seed = pd.read_csv(os.path.join(comp, "sample_submission.csv"))
    seed.iloc[:, 1:] = 0.5
    seed.to_csv("/kaggle/working/submission.csv", index=False)

    work = "/kaggle/working/code"
    if os.path.isdir(work):
        shutil.rmtree(work)
    shutil.copytree(code, work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.chdir(work)

    # Offline install of the JPEG-Lossless / JPEG2000 DICOM decoders (the test set may
    # use them; the train set did not).  Failure is non-fatal: undecodable slices are
    # counted and zero-filled by src/slots.py.
    wheels = os.path.join(work, "wheels")
    if os.path.isdir(wheels):
        run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wheels,
             "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "-q"], check=False)
        # dicomsdl: same pixels (parity-tested in tests/test_decoder_parity.py), faster decode.
        # Decode dominates this kernel's wall-clock, and infer falls back to pydicom per file
        # if anything about it surprises us, so a failed install costs nothing but speed.
        run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", wheels,
             "dicomsdl", "-q"], check=False)

    # SK_CKPT_GLOB selects WHICH checkpoints in the models dataset this entry uses, so the
    # 5-fold ACCURACY set and the single full-fit EFFICIENCY student can live in ONE dataset
    # version (push-models with a prefix: "student=DIR" -> student_fold_0_best.pt).
    #   accuracy entry   : SK_CKPT_GLOB=fold_*_best.pt          (the five folds)
    #   efficiency entry : SK_CKPT_GLOB=student_*_best.pt SK_TTA=0
    # Default keeps the old behaviour (everything), which is what you want only when the
    # dataset holds exactly one set.
    # The UI's Save & Run All cannot set environment variables, so each ENTRY is its own kernel
    # with the choice baked into the default: this kernel = ACCURACY (the fold_* set, TTA on);
    # slotknee-submit-eff = EFFICIENCY (student_* set, TTA off).  SK_* still override for API runs.
    ckpt_glob = os.environ.get("SK_CKPT_GLOB", "student_*_best.pt")
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "**", ckpt_glob), recursive=True))
    print(f"checkpoint glob: {ckpt_glob}", flush=True)
    if not ckpts:
        raise SystemExit(f"no checkpoints match {ckpt_glob!r} under {ckpt_dir} -- check SK_CKPT_GLOB / push-models prefix")
    if len(ckpts) > MAX_CKPTS:
        print(f"WARN: {len(ckpts)} checkpoints > MAX_CKPTS={MAX_CKPTS}; keeping the first {MAX_CKPTS}", flush=True)
        ckpts = ckpts[:MAX_CKPTS]
    print("using checkpoints:", [os.path.relpath(c, ckpt_dir) for c in ckpts], flush=True)

    # Group checkpoints by INPUT LAYOUT (resolution / anchors / slots / zoom).  Models trained
    # on different views of the study are decorrelated, which is where ensembling pays most;
    # each group is decoded once, then the groups are rank-averaged per finding.
    import torch
    groups = {}
    for c in ckpts:
        ck = torch.load(c, map_location="cpu")
        hp = ck.get("hparams", {}); lay = ck.get("slot_layout") or {}
        # 2026-09-09: the anchor band (trim_frac) and crop size are part of the input layout too --
        # a t35 "central" fold grouped with a 0.15 fold would be decoded on the wrong slices.
        # The key must separate every layout infer_slotknee._check_layouts separates, or a group's
        # members get decoded with the first member's spec and infer exits, losing the whole group.
        key = (int(hp.get("P", 224)), int(hp.get("T", 3)), int(hp.get("n_slots", 6)), int(lay.get("G", 3)),
               str(lay.get("zoom_mm")), tuple(lay.get("zoom_slots") or ()),
               float(lay.get("trim_frac", 0.15)), float(lay.get("crop_mm", 140.0)),
               str(lay.get("zoom_center", "image")), json.dumps(lay.get("zoom_spec"), sort_keys=True, default=str),
               tuple(lay.get("slot_names") or ()))
        groups.setdefault(key, []).append(c)
        del ck
    print("layout groups:", {str(k): len(v) for k, v in groups.items()}, flush=True)

    # Per-label per-group stacking weights (scripts/stack_oof.py), shipped in the models
    # dataset as stack_weights*.json keyed by checkpoint filename prefix (g10t1/base224/...).
    # Anything missing or malformed -> equal weights, the previous behaviour.
    stack = None
    # 2026-09-17: the file must be named stack_weights_ADOPTED.json.  A weights file that merely sits
    # in the models dataset used to be applied automatically, so an unscored, sub-gate fit (or a stale
    # one) could silently change a re-scored submission -- including a re-run of an already-scored
    # version.  Adoption is now a deliberate rename, and the applied file is echoed in the log.
    wfiles = sorted(glob.glob(os.path.join(ckpt_dir, "**", "stack_weights_ADOPTED.json"), recursive=True)
                    or glob.glob(os.path.join(os.path.dirname(ckpt_dir), "stack_weights_ADOPTED.json")))
    if wfiles:
        try:
            with open(wfiles[0]) as fh:
                stack = json.load(fh)
            if not (isinstance(stack.get("prefixes"), list) and isinstance(stack.get("labels"), dict)):
                raise ValueError("needs 'prefixes' list and 'labels' dict")
            print(f"stack weights: {wfiles[0]} prefixes={stack['prefixes']}", flush=True)
        except Exception as e:
            print(f"stack weights unusable ({e}); equal weights", flush=True)
            stack = None
    else:
        print("no stack_weights_ADOPTED.json in the models dataset; equal weights over layout groups", flush=True)

    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    # A submission must never crash on the GPU lottery: Kaggle's PyTorch has no sm_60
    # kernels, so on a P100 hide the GPU and run on CPU (slower, still well inside 9 h).
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            print(f"GPU: {torch.cuda.get_device_name(0)} sm_{major}{minor}", flush=True)
            if major < 7:
                print("unsupported GPU -> falling back to CPU inference", flush=True)
                env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            print("no GPU visible -> CPU inference", flush=True)
    except Exception as e:  # pragma: no cover
        print("GPU probe failed:", e, flush=True)
    cpu_mode = env.get("CUDA_VISIBLE_DEVICES") == "" or not torch.cuda.is_available()
    if cpu_mode:
        # Time budget on CPU (measured on Kaggle, submit v5 log: 1.72 s decode + 1.57 s per
        # member per study at 224/G3/T3, ~1.75 s at G10/T1).  One layout group of up to
        # CPU_MEMBERS members, no TTA: 5 members ~ 10.5 s/study ~ 3.8 h for 1,300 studies,
        # well inside the 9 h cap; 15 members x 3 TTA shifts would take ~23 h.
        tta_on = False
        # keep ONE layout group: the one with the most anchors per study (the adopted coverage
        # layout), not whichever checkpoint path happens to sort first
        key = max(groups, key=lambda k: (k[2] * k[3], len(groups[k])))
        groups = {key: groups[key]}
        per_member = CPU_FWD_S * key[2] * key[3]            # n_slots x G forwards per study
        per_study_budget = CPU_BUDGET_H * 3600 / CPU_TEST_STUDIES - CPU_DECODE_S
        n_keep = max(1, int(per_study_budget // per_member))
        groups[key] = groups[key][:n_keep]
        n_members = len(groups[key])
        print(f"CPU mode: TTA off, 1 layout group, {per_member:.2f} s/member/study -> {n_members} checkpoint(s); "
              f"projected {CPU_TEST_STUDIES * (CPU_DECODE_S + per_member * n_members) / 3600:.1f} h "
              f"for {CPU_TEST_STUDIES} studies (budget {CPU_BUDGET_H} h)", flush=True)
    else:
        # TTA PRICED 2026-08-25 on 131 held-out studies rebuilt from DICOM at five shifts:
        #   1 shift 0.8328 | 3 shifts 0.8372 (+0.0044) | 5 shifts 0.8378 (+0.0006)
        # So 3 shifts is right and 5 is waste — do NOT add more shifts.
        # BUT the two entries diverge, because 3 shifts costs 2 extra forward passes per
        # member (model time 0.392 -> 1.176 s/study on a T4 = 0.0142 AUC-equivalents at
        # 0.01 AUC ~ 717 s):
        #   ACCURACY entry  -> SK_TTA=1 (default). Runtime is irrelevant under the 9 h cap,
        #                      so +0.0044 is free.
        #   EFFICIENCY entry-> SK_TTA=0. TTA nets **-0.0098** there: it buys 0.0044 of AUC
        #                      and spends 0.0142 of runtime. Worth ~+0.010 on that track.
        tta_on = os.environ.get("SK_TTA", "0") == "1"   # EFFICIENCY entry: TTA off by default (-0.0098 net on this track)
        print(f"TTA: {'on (3 anchor shifts)' if tta_on else 'OFF — efficiency entry (SK_TTA=0)'}",
              flush=True)

    # Decoder self-check: the hidden test set may contain JPEG-Lossless / JPEG2000 DICOMs.
    try:
        import pydicom
        from pydicom.uid import JPEG2000Lossless, JPEGLosslessSV1
        try:
            from pydicom.pixels import get_decoder
            for ts in (JPEG2000Lossless, JPEGLosslessSV1):
                dec = get_decoder(ts)
                print(f"decoder {ts.name}: available={dec.is_available}", flush=True)
        except ImportError:   # pydicom 2.x
            from pydicom.pixel_data_handlers import pylibjpeg_handler
            print("pylibjpeg handler available:", pylibjpeg_handler.is_available(), flush=True)
    except Exception as e:
        print("decoder self-check failed:", e, flush=True)
    outs, out_prefixes = [], []
    for gi, (key, members) in enumerate(groups.items()):
        out_g = f"/kaggle/working/submission_group{gi}.csv"
        cmd = [sys.executable, "scripts/infer_slotknee.py", "--data-dir", comp, "--ckpt", *members,
               "--out", out_g, "--P", str(key[0]), "--bs", "4",
               "--decoder", os.environ.get("SK_DECODER", "dicomsdl")] + (["--anchor-shift", "0,1,-1"] if tta_on else [])
        print(f"[group {gi}] P={key[0]} T={key[1]} slots={key[2]} G={key[3]} zoom={key[4]} members={len(members)} "
              f"prefix={ckpt_prefix(members[0]) or '(none)'}", flush=True)
        try:
            run(cmd, env=env)
            outs.append(out_g)
            out_prefixes.append(ckpt_prefix(members[0]))
        except subprocess.CalledProcessError as e:
            print(f"[group {gi}] FAILED ({e}); skipped", flush=True)
    import pandas as pd
    from scipy.stats import rankdata
    sample = pd.read_csv(os.path.join(comp, "sample_submission.csv"))
    final = sample.copy()
    labels = [c for c in sample.columns if c != "StudyInstanceUID"]
    dfs = [pd.read_csv(o).set_index("StudyInstanceUID").reindex(sample.StudyInstanceUID) for o in outs]
    if dfs:
        applied = {}
        for l in labels:
            wg = group_weight_vector(l, out_prefixes, stack, len(dfs))
            applied[l] = [round(float(x), 3) for x in wg]
            r = np.sum([wgt * rankdata(d[l].fillna(0.5).values) / len(d) for wgt, d in zip(wg, dfs)], axis=0)
            final[l] = r
        tmp = "/kaggle/working/submission.csv.tmp"; final.to_csv(tmp, index=False); os.replace(tmp, "/kaggle/working/submission.csv")
        print(f"rank-averaged {len(dfs)} layout group(s) -> submission.csv; "
              f"group weights per label: {applied}", flush=True)
    else:
        print("no group produced predictions; submission.csv stays seeded at 0.5", flush=True)
    print(f"done in {(time.time()-T0)/60:.1f} min", flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
