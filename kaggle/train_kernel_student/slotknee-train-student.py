# %% [code]
"""Kaggle GPU kernel: ONE full-fit SlotKnee-S student for the EFFICIENCY entry.

Distils the 5-fold ensemble into a single model trained on ALL 4,407 studies (--full-fit: no
holdout, the last EMA epoch is saved).  Targets = labels_external/teacher_g10_v3.csv, the 16-epoch
5-fold run's out-of-fold probabilities: each study's target comes from a model that never trained on
it, and the hidden test set is in no fold, so this is leak-free for a SUBMISSION model.

Why: on the efficiency score 0.01 AUC ~ 12 min of runtime over 1,300 studies.  The 5-member
ensemble costs 1.18 s/study of model time with TTA; one student costs 0.08 s/study -- 24 min
back, i.e. +0.0199 AUC-equivalents of headroom.  The student wins if it loses LESS than that
against the ensemble (docs/research_improvements_20260824.md, "Efficiency-aware re-ranking").
There is no OOF from a full fit; judge it on the public LB, efficiency track.

Attached sources (kernel-metadata.json): the code dataset (teacher + weights + labels), the
g10t1 coverage cache, and the competition (labels CSV + DICOM headers for the fold groups).
Output: /kaggle/working/models/slotknee_student/fold_0_best.pt -> push with
  python3 kaggle/kaggle_ops.py push-models --models-dir <pulled dir>   (ONE member; submit with SK_TTA=0)
"""
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
EPOCHS = int(os.environ.get("SK_EPOCHS", "16"))   # ADOPTED 2026-09-06 (+0.0160 at 16 vs 8 on 2 folds); full fit: 4,407 studies/epoch (~7.3 min on a T4) -> ~2 h
FOLDS = os.environ.get("SK_FOLDS", "0").split()   # one full-fit model; the fold index only names the checkpoint
LIMIT_MIN = os.environ.get("SK_LIMIT_MINUTES", "470")   # GPU sessions are capped at 9 h
BS = os.environ.get("SK_BS", "8")
OUT = "/kaggle/working/models/slotknee_student"
TMP = "/kaggle/temp"


def walk_find(pred, prune=("train_series", "test_series", "train_images", "test_images")):
    for base, dirs, files in os.walk("/kaggle/input"):
        hit = pred(base, list(dirs), files)      # test BEFORE pruning: the predicate may look for train_series
        if hit:
            return hit
        dirs[:] = [d for d in dirs if d not in prune]   # never descend into the 800k-file image trees
    return None


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def check_gpu():
    """Kaggle's PyTorch build dropped sm_60: a P100 cannot run any CUDA kernel.

    The API cannot choose the accelerator; it is a kernel setting.  Fail fast with
    the fix instead of dying 10 minutes later inside the first forward pass.
    """
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Kernel settings -> Accelerator -> 'GPU T4 x2', then Save & Run.")
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    print(f"GPU: {name} sm_{major}{minor}", flush=True)
    if major < 7:
        raise SystemExit(f"{name} (sm_{major}{minor}) is not supported by this PyTorch build. "
                         "Open the kernel on Kaggle -> Settings -> Accelerator -> 'GPU T4 x2' "
                         "(the choice persists across API pushes), then Save & Run.")


def main():
    print("input mounts:", os.listdir("/kaggle/input"), flush=True)
    code = walk_find(lambda b, d, f: b if ("build_slot_cache.py" in f and os.path.basename(b) == "scripts") else None)
    code = os.path.dirname(code) if code else None
    comp = walk_find(lambda b, d, f: b if ("train.csv" in f and "train_series" in d) else None)
    cache = walk_find(lambda b, d, f: b if "train_index.json" in f else None)
    print("code:", code, "\ncompetition:", comp, "\ncache:", cache, flush=True)
    if not (code and comp and cache):
        raise SystemExit("missing input: code/competition/cache not all found under /kaggle/input")

    # GPU probe deliberately runs AFTER input discovery.  A CPU push (the only kind the
    # Kaggle API can make -- it always lands on a banned P100) then still validates every
    # mount, cache name and label path before aborting, so the cheap run is a real
    # pre-flight instead of a wasted "No GPU" line that proves nothing.
    check_gpu()

    work = "/kaggle/working/code"
    if os.path.isdir(work):
        shutil.rmtree(work)
    shutil.copytree(code, work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.chdir(work)

    # The cache is a memmap read thousands of times per epoch: local disk beats the
    # network mount.  /kaggle/temp is not preserved, so it does not count against the
    # 20 GB output cap.
    local_cache = os.path.join(TMP, "slots_P224")
    if not os.path.isfile(os.path.join(local_cache, "train_index.json")):
        t = time.time()
        shutil.copytree(cache, local_cache, dirs_exist_ok=True)
        print(f"copied cache to {local_cache} in {time.time()-t:.0f}s", flush=True)

    # data_dir must hold train_gold.csv + train.csv (labels/folds) — build a tiny one.
    data_dir = os.path.join(TMP, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name in ("train.csv", "train_series.csv", "sample_submission.csv", "test.csv"):
        src = os.path.join(comp, name)
        if os.path.isfile(src):
            shutil.copy(src, data_dir)
    gold_src = os.path.join(work, "labels_external", "train_gold.csv")
    # Expose the competition DICOMs as <data_dir>/train_images (symlink, no copy) so
    # src/folds.build_groups can read one header per study and reproduce the SAME
    # scanner-grouped folds as local runs (without it Kaggle fell back to report-hash groups).
    # Only the studies the laptop has on disk get a fingerprint there (649), so link exactly
    # those uids (list shipped in the code dataset) — a whole-tree link would fingerprint all
    # 4,407 and produce a third, different split.
    link = os.path.join(data_dir, "train_images")
    uid_list = os.path.join(work, "labels_external", "fold_image_uids.txt")
    if not os.path.isdir(link) and os.path.isdir(os.path.join(comp, "train_series")):
        os.makedirs(link, exist_ok=True)
        if os.path.isfile(uid_list):
            n = 0
            for uid in open(uid_list).read().split():
                srcd = os.path.join(comp, "train_series", uid)
                if os.path.isdir(srcd):
                    os.symlink(srcd, os.path.join(link, uid)); n += 1
            print(f"fold groups: linked {n} laptop-resident studies for scanner fingerprints", flush=True)
        else:
            print("WARN: fold_image_uids.txt missing -> whole-tree link (folds will NOT match the laptop split)", flush=True)
            os.rmdir(link); os.symlink(os.path.join(comp, "train_series"), link)
    if os.path.isfile(gold_src):
        shutil.copy(gold_src, data_dir)
    else:
        # derive the 58-row gold file from train.csv (rows with any label present)
        import pandas as pd
        df = pd.read_csv(os.path.join(data_dir, "train.csv"))
        lab = [c for c in df.columns if c not in ("StudyInstanceUID", "Report")]
        g = df[df[lab].notna().all(axis=1)].copy()
        g["fold"] = (list(range(5)) * (len(g) // 5 + 1))[:len(g)]
        g.to_csv(os.path.join(data_dir, "train_gold.csv"), index=False)
        print(f"derived train_gold.csv with {len(g)} rows", flush=True)

    # ADOPTED RECIPE BACKBONE (2026-08-24): DINOv2-S **with registers**. Measured +0.0220
    # pooled OOF vs the identical-split baseline (ACL +0.0617) — registers absorb the
    # high-norm artifact tokens that otherwise pollute the patch tokens this model's
    # per-finding attention reads. Same parameter count, same speed.  SK_BACKBONE overrides
    # for a deliberate A/B; the weights for both ship in the code dataset.
    BACKBONE = os.environ.get("SK_BACKBONE", "vit_small_patch14_reg4_dinov2.lvd142m")
    weights = os.path.join(work, "weights", f"{BACKBONE}.safetensors")
    labels = os.path.join(work, "labels_external", "stevenleehans", "llm_labels_v4_blend.csv")
    wfrom = os.path.join(work, "labels_external", "stevenleehans", "llm_labels_v2.csv")
    for p in (weights, labels, wfrom):
        if not os.path.isfile(p):
            raise SystemExit(f"missing {p}")

    # SELF-DISTILLATION (ADOPTED 2026-08-24: r4distill +0.0185 pooled, every label up, gate
    # +0.0066).  Costs 0 s at inference, so it is free on the efficiency track too.
    #
    # TWO THINGS TO KNOW BEFORE READING THIS RUN'S OOF:
    # 1. The teacher targets are the champion's out-of-fold predictions.  Each study's target
    #    comes from a model that never trained on it (no memorisation), BUT the teacher models
    #    collectively saw every fold, so fold-k label information reaches a student that is
    #    then scored on fold k.  **This run's OOF is therefore INFLATED and must NOT be used as
    #    a gate metric.**  Compare recipes on non-distilled runs, or on the public LB.
    # 2. That leak does NOT touch the hidden test set, which appears in no fold at all — so
    #    distillation is legitimate and safe for a SUBMISSION model.  The uncertainty is only
    #    in HOW MUCH it helps, not whether using it is sound.
    # Expect less than +0.0185 here: distillation correlates 0.829 with the coverage layout
    # this cache already provides (both are variance reduction), so ~+0.006-0.009 is realistic.
    # RANDOM SLICE BAGGING — ADOPTED 2026-08-25 (+0.0090 vs the champion on the SAME coverage
    # layout and folds; ACL +0.032, MCL +0.019, LatMen +0.012 — the three weakest findings take
    # the three biggest gains).  Without it the model sees the identical 60 images every epoch.
    BAG = os.environ.get("SK_ANCHOR_BAG", "6")      # sample 6 of the cache's 10 anchors per epoch
    bag = ["--anchor-bag", BAG] if BAG and BAG != "0" else []

    # v2 teacher = the 5-fold ADOPTED-RECIPE run's OOF (built 2026-09-03).  Falls back to the
    # older champion teacher only if the code dataset predates it, and says so.
    # Teacher preference (2026-09-08): v3 = the 16-epoch 5-fold OOF (pooled 0.8812, expert macro-10 0.897)
    # > v2 = the 8-epoch 5-fold OOF > the original champion OOF.  Each study's target still comes
    # from a fold model that never saw it; the hidden test set is in no fold.
    teacher = None
    for cand in ("teacher_g10_v3.csv", "teacher_g10_v2.csv", "teacher_g10_oof.csv"):
        cp = os.path.join(work, "labels_external", cand)
        if os.path.isfile(cp):
            teacher = cp; print(f"teacher: {cand}", flush=True); break
    if teacher is None:
        raise SystemExit("no distillation teacher CSV found under labels_external/")
    distill = ([] if os.environ.get("SK_DISTILL", "1") == "0" or not os.path.isfile(teacher)
               else ["--distill-targets", teacher, "--distill-weight", os.environ.get("SK_DISTILL_W", "0.5")])
    if not distill:
        print("distillation OFF (SK_DISTILL=0 or teacher missing)", flush=True)

    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "scripts/train_slotknee.py", "--cache", local_cache,
           "--data-dir", data_dir, "--labels", labels, "--weights-from", wfrom,
           "--pretrained-path", weights, "--folds", *FOLDS, "--epochs", str(EPOCHS),
           "--bs", BS, "--workers", "2", "--out", OUT, "--seed", "42",
           "--backbone", BACKBONE, "--limit-minutes", LIMIT_MIN, "--full-fit"] + distill + bag
    run(cmd, env=env)

    report = {"elapsed_min": round((time.time() - T0) / 60, 1), "epochs": EPOCHS, "folds": FOLDS}
    for k in FOLDS:
        lf = os.path.join(OUT, f"fold_{k}_log.json")
        if os.path.isfile(lf):
            with open(lf) as fh:
                report[f"fold_{k}"] = json.load(fh)
    with open("/kaggle/working/train_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("fold_")}), flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
