# %% [code]
# %% [code]
"""Kaggle GPU kernel: SlotKnee-S ablation arms ON THE COVERAGE LAYOUT (g10t1) — one T4 session.

Attached sources (kernel-metadata.json is the truth; this list was refreshed 2026-09-08):
  competition  rsna-knee-abnormality-detection          (labels CSV only; pixels come from the caches)
  dataset      felipedeleon11/slotknee-code             (src/, scripts/, labels_external/, weights/, wheels/)
  kernel       felipedeleon11/slotknee-cache-g10t1      (slots_P224_g10t1: 224 px, 10 single-slice anchors, 11.9 GB — the adopted layout)
  kernel       felipedeleon11/slotknee-cache-g10t1-t35  (slots_P224_t35_g10t1: same, anchors over the central 30 %; the `central` arm)
  kernel       felipedeleon11/slotknee-cache-zoomc      (slots_P224_g10t1_zc: + COR_FS/COR_T1 joint-centred 80 mm zoom slots, S = 8)
  kernel       felipedeleon11/slotknee-cache-zoomm      (slots_P224_g10t1_zm: + COR_FS_Z80M medial-column zoom slot, S = 7, 14.4 GB; built 2026-09-08)
  kernel       felipedeleon11/slotknee-cache-zoomj      (slots_P224_g10t1_z100j: + joint-centred 100 mm zoom slots, 17.7 GB)
  kernel       felipedeleon11/slotknee-cache-g20a/-g20b (slots_P224_g20t1_partA/B: 20 anchors, merged at run time)
  kernel       felipedeleon11/slotknee-cache-g10t3a/b/c (slots_P224_g10t3_partA/B/C: T = 3, merged at run time)
  kernel       felipedeleon11/slotknee-ssl              (ssl/encoder_ep10|20|30.safetensors: MAE-continued DINOv2-S reg4, v6, 30 ep, 2026-09-08)

Reference for every arm: folds 0-1 on the adopted recipe (same seed and fold split — the data_dir exposes the
649 laptop-resident studies so src/folds.build_groups reproduces the laptop folds); the 8-epoch controls are
recipe_base_s42 (session 1) and recipe_base_s1337 (session 2), noise floor 0.0005.  A 2-fold 8-epoch arm on
the g10t1 layout takes ~95-100 min on a T4 (16 epochs ~190 min; LoRA ~2x; a 7- or 8-slot cache ~7/6-8/6 x),
so four or five 8-epoch arms fit one session; the loop stops before an arm that has < 60 min left.  The kernel
has NO skip-already-done logic (every run starts from a fresh /kaggle/working and walks ARMS from the top), so
each session is a re-cut of ARMS pushed as a new version.  Each arm's cache is copied to local disk first and
the previous copy is deleted when the next arm uses a different one.
Outputs: /kaggle/working/models/ablate_g10/<arm>_s42/fold_*_best.pt, oof_fold_*.npz, logs; ablate_report.json.
"""
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
EPOCHS = int(os.environ.get("SK_EPOCHS", "16"))   # ADOPTED 2026-09-06: every arm is measured on the 16-epoch recipe (control = recipe_e16_s42, session 2)
FOLDS = os.environ.get("SK_FOLDS", "0 1 2 3 4").split()
LIMIT_MIN = os.environ.get("SK_LIMIT_MINUTES", "470")   # GPU sessions are capped at 9 h
BS = os.environ.get("SK_BS", "8")
OUT_ROOT = "/kaggle/working/models/ablate_g10"
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



# One T4 session ≈ 9 h; ~2.5 h per g10-layout arm (p336 ≈ 2.3 h, zoomj ≈ 3.3 h).  Order = priority:
# the loop stops before any arm with < 60 min left.  Edit, then `kaggle_ops.py push-ablate-g10`.
#   HEAD_FIX: filled from `slotknee-ablate` v3's verdict (tau3 / mil / nomixer) — placeholder = tau3, the
#   candidate most likely to matter with 60 attention tokens per finding instead of 18.
HEAD_FIX = ("g10_tau3", ["--attn-tau-init", "3.0"])
# RE-PRIORITISED 2026-08-24 23:45 after pricing three levers against the trained champion
# WITHOUT spending a session (docs/research_improvements_20260824.md, "THREE LEVERS PRICED"):
#   * more anchors  -> SATURATED (6->8 anchors +0.0104, 8->10 +0.0016), so G20 is worth
#     ~+0.002-0.004 against a 0.0066 gate. Dropped.
#   * more resolution -> the model does NOT use the detail it already has: blurring away
#     everything finer than ~112 px costs 0.0012. **p336 REMOVED from this queue** — it is the
#     most expensive arm here (2.25x tokens/step, most of a session) with the weakest prior.
#   * more seeds -> +0.0011 at 0.953 rank correlation. Not a session.
# What survives is the ZOOM family: a joint-centred crop is the only input change that supplies
# information the current model has never seen, so no probe of that model can price it. That is
# exactly why it goes first — the unpriced lever is the one worth paying for.
ARMS = [
    # SESSION 5 QUEUE (re-cut 2026-09-09 after session 4; the kernel walks ARMS from the top every run).
    #
    # SESSION 4 RESULTS (v28, T4, 466 min, 8 epochs, folds 0-1; docs/slotknee_runbook.md 2026-09-09).
    # Two 8-epoch controls: recipe_base_s42 0.8650 / recipe_base_s1337 0.8662 pooled -> seed noise 0.0012,
    # gate 0.0023 for 8-epoch arms (the 0.0005 floor belongs to the 16-epoch pair):
    #   central   +0.0038 vs s42 / +0.0027 vs s1337  ADOPT on both controls (ACL +0.014, MCL +0.008, LatMen +0.013)
    #   flipswap  +0.0027 vs s42 / +0.0015 vs s1337  MARGINAL, control-dependent (LatMen +0.05, MedMen -0.015)
    #   auxslot   +0.0014 / +0.0002               NULL
    #   lora_r8   -0.0183 / -0.0195               REGRESS on all 12 labels (r 8 / lr 1e-4 / 8 ep)
    #   mirror TTA on the flipswap model (rescore step): null (-0.0003 / +0.0002 per fold)
    #
    # SESSION 5 RESULTS (v32, 09-15, 472 min, folds 0-1; docs/slotknee_runbook.md 2026-09-15 evening):
    #   central_e16  +0.0025 vs recipe_e16_s42 (0.8847 vs 0.8822)  ADOPT at 16 ep -> the 5-fold is retrained on t35
    #   ssl_ep30 / ssl_ep10  -0.078 / -0.075 vs recipe_base_s42    REGRESS on every label -> SSL stage 1 CLOSED
    #   slotemb  +0.0003                                            NULL
    #
    # SESSION 6 ORDER (v33):
    # 1. CENTRAL + FLIP-SWAP at 16 epochs, gated vs central_e16 (0.8847 on these folds): flipswap's own
    #    contribution at the real epoch budget on the adopted cache.  A mirrored medial-zoom slot is dropped by
    #    the mirror helpers (is_medial_slot), so this arm and zoomm may later stack safely.
    ("central_flip_e16", ["--flip-swap", "0.5"], "slots_P224_t35_g10t1"),                          # ~185 min
    # 2. (below) zoomm at 8 epochs, then recipe_e24 if the budget rule lets it start.
    # DONE in session 5 (v32), kept for the record -- do not re-run:
    #   ("central_e16", [], "slots_P224_t35_g10t1")
    #   ("ssl_ep30", ["--epochs", "8", "--pretrained-path", "@ssl:encoder_ep30.safetensors"], "slots_P224_g10t1")
    #   ("ssl_ep10", ["--epochs", "8", "--pretrained-path", "@ssl:encoder_ep10.safetensors"], "slots_P224_g10t1")
    #   ("slotemb", ["--epochs", "8", "--slot-tok-embed"], "slots_P224_g10t1")
    # ssl_ep20 is dropped: ep10 and ep30 agree.
    # 7. MEDIAL-COLUMN zoom slot (design review 2026-09-08, R5): one extra COR_FS crop, 80 mm window
    #    centred +20 mm distal / 32 mm medial of the located joint (rows -20..+60, cols -72..+8 mm) =
    #    a magnified medial compartment (0.357 vs 0.625 mm/px), the zoomc mechanism family (+0.0045 /
    #    +0.0028 pooled on two seeds), NOT the whole sMCL.  Cache slotknee-cache-zoomm is COMPLETE
    #    (4,407 studies; joint fallback 15.5 %, unresolved side 1.6 %, both pre-gates pass).
    #    Gate: pooled >= +0.0011 vs recipe_base_s42/s1337 AND no label below -0.005 (the cor_spec
    #    failure: MCL +0.0215 but Baker's -0.093); MCL per-label read is informational only.  Mirror
    #    passes DROP a medial slot (src.slots.is_medial_slot, 2026-09-08), so it may later stack with
    #    flip-swap safely.
    ("zoomm",       ["--epochs", "8"], "slots_P224_g10t1_zm"),                                     # ~105 min; session 6
    # 8. Where is the plateau?  24 epochs on the adopted recipe (~285 min; needs its own session).
    ("recipe_e24",  ["--epochs", "24"], "slots_P224_g10t1"),
    # 9. Sagittal specialist (cheap: 2 of 6 slots) -- completes the plane picture for a slot prior.
    ("sag_spec",    ["--epochs", "8", "--slots", "SAG_FS,SAG_T1"], "slots_P224_g10t1"),
    # Later / conditional.
    ("zoomc",       [], "slots_P224_g10t1_zc"),                 # the two adopts stacked (16 ep)
    ("nogdrop",     ["--epochs", "8", "--no-aug-gdrop"], "slots_P224_g10t1"),
    ("noshift",     ["--epochs", "8", "--no-aug-shift"], "slots_P224_g10t1"),
    ("zoomj",       ["--epochs", "8"], "slots_P224_g10t1_z100j"),
    ("g20_bag6",    ["--epochs", "8", "--anchor-bag", "6"], "slots_P224_g20t1"),
    ("gold32",      ["--epochs", "8", "--gold-weight", "32"], "slots_P224_g10t1"),
    (HEAD_FIX[0],   ["--epochs", "8"] + HEAD_FIX[1], "slots_P224_g10t1"),
    ("g10_cutrot",  ["--epochs", "8", "--cutout", "--rot-aug"], "slots_P224_g10t1"),
    ("g10t3",       ["--epochs", "8"], "slots_P224_g10t3"),
    # DONE in session 4 (v28, 2026-09-08/09), kept for the record -- do not re-run:
    #   ("flipswap", ["--epochs", "8", "--flip-swap", "0.5"], "slots_P224_g10t1"),
    #   ("auxslot",  ["--epochs", "8", "--aux-slotid", "0.1"], "slots_P224_g10t1"),
    #   ("central",  ["--epochs", "8"], "slots_P224_t35_g10t1"),
    #   ("lora_r8",  ["--epochs", "8", "--lora-rank", "8", "--lr-backbone", "1e-4"], "slots_P224_g10t1"),
]
FOLDS = os.environ.get("SK_FOLDS", "0 1").split()
# Expected wall minutes per arm (2 folds, measured 5.53 min/epoch/fold in session 4 + cache copy).  The loop
# stops BEFORE an arm whose expectation exceeds the time left (2026-09-15; the old flat "< 60 min" rule let a
# 180-min arm start with 61 min left and run truncated).  Unlisted arms fall back to 60.
EXPECTED_MIN = {"central_e16": 185, "central_flip_e16": 185, "recipe_e24": 290, "zoomc": 200,
                "ssl_ep30": 95, "ssl_ep10": 95, "ssl_ep20": 95, "slotemb": 95, "zoomm": 110}
TOTAL_MIN = int(os.environ.get("SK_TOTAL_MINUTES", "500"))


def main():
    print("input mounts:", os.listdir("/kaggle/input"), flush=True)
    code = walk_find(lambda b, d, f: b if ("build_slot_cache.py" in f and os.path.basename(b) == "scripts") else None)
    code = os.path.dirname(code) if code else None
    comp = walk_find(lambda b, d, f: b if ("train.csv" in f and "train_series" in d) else None)
    caches = {}
    for base, dirs, files in os.walk("/kaggle/input"):
        if "train_index.json" in files and os.path.basename(base).startswith("slots_"):
            caches[os.path.basename(base)] = base
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series", "train_images", "test_images")]
    print("code:", code, "\ncompetition:", comp, "\ncaches:", caches, flush=True)
    if not (code and comp and caches):
        raise SystemExit("missing input: code/competition/caches not all found under /kaggle/input")

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

    data_dir = os.path.join(TMP, "data")
    os.makedirs(data_dir, exist_ok=True)
    for name in ("train.csv", "train_series.csv", "sample_submission.csv", "test.csv"):
        src = os.path.join(comp, name)
        if os.path.isfile(src):
            shutil.copy(src, data_dir)
    shutil.copy(os.path.join(work, "labels_external", "train_gold.csv"), data_dir)
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
    weights = os.path.join(work, "weights", "vit_small_patch14_reg4_dinov2.lvd142m.safetensors")
    labels = os.path.join(work, "labels_external", "stevenleehans", "llm_labels_v4_blend.csv")
    wfrom = os.path.join(work, "labels_external", "stevenleehans", "llm_labels_v2.csv")
    teacher = os.path.join(work, "labels_external", "teacher_g10_oof.csv")
    # Every arm is measured ON TOP of the adopted recipe, not against a stale baseline.
    # Before 2026-08-29 this kernel silently used the DEFAULT backbone (plain dinov2) with
    # no distillation and no bagging -- deltas measured on a model nobody trains.  An arm
    # can still override any of these: extras are appended AFTER base, and argparse wins last.
    base = ["--backbone", "vit_small_patch14_reg4_dinov2.lvd142m",
            "--distill-targets", teacher, "--distill-weight", "0.5",
            "--anchor-bag", "6"]
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")

    report = {}
    for name, extra, cache_name in ARMS:
        left = TOTAL_MIN - (time.time() - T0) / 60
        if left < max(60, EXPECTED_MIN.get(name, 60)):
            print(f"[ablate] stop before {name}: {left:.0f} min left < {max(60, EXPECTED_MIN.get(name, 60))} expected", flush=True)
            print(f"[ablate] {left:.0f} min left: stopping before {name}", flush=True)
            break
        # A cache can arrive whole, or as parts (`<name>_partA`, `_partB`, ...) when it is too
        # big for Kaggle's 20 GB output cap (g20t1 = 23.8 GB, g10t3 = 39.8 GB).  Parts are merged
        # onto local disk at run time with scripts/merge_slot_cache.py, so no kernel ever has to
        # publish the merged cache.  Merging copies every shard (the inputs are read-only), so it
        # costs the same disk and roughly the same time as the plain copy path.
        parts = sorted(k for k in caches if k.startswith(cache_name + "_part"))
        if cache_name not in caches and not parts:
            print(f"[ablate] skip {name}: cache {cache_name} not attached (no whole copy, no parts)", flush=True)
            continue
        local_cache = os.path.join(TMP, cache_name)
        for other in os.listdir(TMP) if os.path.isdir(TMP) else []:   # local disk is finite: keep one cache copy
            if other.startswith("slots_") and other != cache_name:
                shutil.rmtree(os.path.join(TMP, other), ignore_errors=True)
                print(f"evicted local copy of {other}", flush=True)
        if not os.path.isfile(os.path.join(local_cache, "train_index.json")):
            t = time.time()
            if cache_name in caches:
                shutil.copytree(caches[cache_name], local_cache, dirs_exist_ok=True)
                print(f"copied {cache_name} in {time.time()-t:.0f}s", flush=True)
            else:
                print(f"[ablate] merging {len(parts)} parts -> {cache_name}: {parts}", flush=True)
                try:
                    run([sys.executable, "scripts/merge_slot_cache.py", "--parts",
                         *[caches[k] for k in parts], "--out", local_cache,
                         "--test-from", caches[parts[0]]], env=env)
                except subprocess.CalledProcessError as e:
                    print(f"[ablate] skip {name}: merge FAILED ({e})", flush=True)
                    shutil.rmtree(local_cache, ignore_errors=True)
                    continue
                print(f"merged {cache_name} in {time.time()-t:.0f}s", flush=True)
        # "@ssl:<file>" in the extras -> the attached slotknee-ssl output (exact filename, located once per
        # arm).  Missing file = skip the arm: training from the DINOv2 init under an ssl_* name would be a
        # silently mislabelled control.
        if any(str(e).startswith("@ssl:") for e in extra):
            resolved = []
            for e in extra:
                if str(e).startswith("@ssl:"):
                    fname = str(e)[len("@ssl:"):]
                    hit = walk_find(lambda b, d, f, fname=fname: os.path.join(b, fname) if fname in f else None)
                    if not hit:
                        print(f"[ablate] skip {name}: {fname} not found under /kaggle/input (attach felipedeleon11/slotknee-ssl)", flush=True)
                        resolved = None; break
                    # The file's own metadata (written by scripts/ssl_pretrain.py) must match what the arm asked
                    # for: a same-named file from another attached output would otherwise train silently.
                    try:
                        from safetensors import safe_open
                        with safe_open(hit, "pt") as fh:
                            meta = dict(fh.metadata() or {})
                    except Exception as exc:
                        meta = {"error": repr(exc)[:120]}
                    want_ep = "".join(ch for ch in fname.split("_ep")[-1] if ch.isdigit()) if "_ep" in fname else ""
                    bb_ok = (meta.get("backbone") == "vit_small_patch14_reg4_dinov2.lvd142m")
                    ep_ok = (not want_ep) or (str(meta.get("epoch")) == want_ep)
                    if not (bb_ok and ep_ok and meta.get("ssl") == "mae"):
                        print(f"[ablate] skip {name}: {hit} metadata {meta} does not match (backbone reg4, epoch {want_ep or 'any'}, ssl=mae)", flush=True)
                        resolved = None; break
                    print(f"[ablate] {name}: {e} -> {hit} ({os.path.getsize(hit)/2**20:.0f} MB; epoch {meta.get('epoch')}, step {meta.get('step')}, blr {meta.get('blr')}, layer_decay {meta.get('layer_decay')})", flush=True)
                    resolved.append(hit)
                else:
                    resolved.append(e)
            if resolved is None:
                continue
            extra = resolved
        # An arm may override the seed in its extras ("--seed", "1337"); name its output honestly.
        seed = extra[extra.index("--seed") + 1] if "--seed" in extra else "42"
        out = os.path.join(OUT_ROOT, f"{name}_s{seed}")
        cmd = [sys.executable, "scripts/train_slotknee.py", "--cache", local_cache, "--data-dir", data_dir,
               "--labels", labels, "--weights-from", wfrom, "--pretrained-path", weights,
               "--folds", *FOLDS, "--epochs", str(EPOCHS), "--bs", BS, "--workers", "2", "--out", out,
               "--seed", "42", "--limit-minutes", str(int(left - 15))] + base + extra
        print(f"[ablate] == {name} ({left:.0f} min left)", flush=True)
        try:
            run(cmd, env=env)
        except subprocess.CalledProcessError as e:
            print(f"[ablate] {name} FAILED: {e}", flush=True)
        n_req = int(extra[extra.index("--epochs") + 1]) if "--epochs" in extra else EPOCHS
        done = {}
        for k in FOLDS:
            lf = os.path.join(out, f"fold_{k}_log.json")
            if os.path.isfile(lf):
                with open(lf) as fh:
                    report[f"{name}/fold{k}"] = json.load(fh)
                done[k] = sum(1 for e in report[f"{name}/fold{k}"] if isinstance(e, dict) and "epoch" in e)
        # 2026-09-09: an arm that hit --limit-minutes has fewer completed epochs than requested; its OOF
        # is from an earlier epoch and is NOT comparable to the controls -- say so loudly and in the report.
        truncated = (len(done) < len(FOLDS)) or any(v < n_req for v in done.values())
        report[f"{name}/status"] = {"epochs_requested": n_req, "epochs_done": done, "truncated": truncated,
                                    "minutes": round((time.time() - T0) / 60 - (TOTAL_MIN - left), 1)}
        print(f"[ablate] {name}: epochs done {done} / {n_req}{'  ** TRUNCATED -- not comparable **' if truncated else ''}", flush=True)
    # MULTI-BAG INFERENCE, measured on the session's own control (2026-09-03).  The recipe trains
    # on random 6-of-10 anchor bags but is validated on all 10; scripts/rescore_oof.py re-scores
    # the control's held-out folds with N averaged 6-bags instead.  Inference-only, ~10 min on a
    # T4 for two folds, so it rides the tail of the session; skipped if fewer than 25 min remain.
    # If it adopts, the same read-out goes into the submit kernel (--bag 6 --bags N).
    left = TOTAL_MIN - (time.time() - T0) / 60
    # 2026-09-07: the rescore step now measures MIRROR TTA (--flip-tta) on the arm named by
    # SK_RESCORE_ARM (default flipswap, the model trained with the mirror augmentation), reporting the
    # plain and the flip-averaged read-out separately.  Multi-bag stays measured (closed, -0.002).
    _arm = os.environ.get("SK_RESCORE_ARM", "central_flip_e16")
    ctrl = [d for d in sorted(os.listdir(OUT_ROOT)) if d.startswith(_arm + "_s")] if os.path.isdir(OUT_ROOT) else []
    # 2026-09-06: multi-bag inference measured -0.0015 / -0.0020 on the control -> CLOSED.  The
    # step stays for re-use (SK_RESCORE=1) but no longer spends session time by default.
    # 2026-09-09: the rescore must use the ARM'S OWN cache (a t35-trained arm re-scored on the
    # g10t1 anchors would be the train/test mismatch this session exists to avoid).
    _arm_cache = next((c for n, _e, c in ARMS if n == _arm), "slots_P224_g10t1")
    if os.environ.get("SK_RESCORE", "1") == "1" and ctrl and left >= 25 and os.path.isfile("scripts/rescore_oof.py"):
        cdir = os.path.join(OUT_ROOT, ctrl[0]); cache_dir = os.path.join(TMP, _arm_cache)
        if not os.path.isfile(os.path.join(cache_dir, "train_index.json")) and _arm_cache in caches:
            for other in os.listdir(TMP) if os.path.isdir(TMP) else []:     # one local cache at a time (~12 GB each)
                if other.startswith("slots_") and other != _arm_cache:
                    shutil.rmtree(os.path.join(TMP, other), ignore_errors=True); print(f"[rescore] evicted {other}", flush=True)
            shutil.copytree(caches[_arm_cache], cache_dir, dirs_exist_ok=True)
        rescore = {}
        for k in FOLDS:
            ck, oof = os.path.join(cdir, f"fold_{k}_best.pt"), os.path.join(cdir, f"oof_fold_{k}.npz")
            if not (os.path.isfile(ck) and os.path.isfile(oof) and os.path.isfile(os.path.join(cache_dir, "train_index.json"))):
                continue
            print(f"[rescore] {ctrl[0]} fold {k}: mirror TTA (flip + medial/lateral swap-back) vs plain", flush=True)
            try:
                r = subprocess.run([sys.executable, "scripts/rescore_oof.py", "--cache", cache_dir, "--ckpt", ck, "--oof", oof,
                                    "--bag", os.environ.get("SK_RESCORE_BAG", "6"), "--bags", os.environ.get("SK_RESCORE_BAGS", "1"), "--flip-tta",
                                    "--out", os.path.join(cdir, f"oof_fold_{k}_bags.npz")], env=env, capture_output=True, text=True, check=True)
                rescore[f"fold_{k}"] = json.loads(r.stdout[r.stdout.index("{"): r.stdout.rindex("}") + 1])
                print("   ", {kk: rescore[f"fold_{k}"].get(kk) for kk in ("macro_full", "macro_flip_tta", "delta_flip")}, flush=True)
            except Exception as e:
                print(f"[rescore] fold {k} FAILED: {e}", flush=True)
        if rescore:
            with open("/kaggle/working/rescore_report.json", "w") as fh:
                json.dump(rescore, fh, indent=2)
    else:
        print(f"[rescore] skipped (control={bool(ctrl)}, {left:.0f} min left, tool present={os.path.isfile('scripts/rescore_oof.py')})", flush=True)

    with open("/kaggle/working/ablate_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print({k: (v[-2] if isinstance(v, list) and len(v) > 1 else v) for k, v in report.items()}, flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
