"""Kaggle GPU kernel: MAE continued pretraining of the DINOv2-S/14 encoder on the full slot cache.

Attached sources (kernel-metadata.json):
  dataset      felipedeleon11/slotknee-code             (src/, scripts/, weights/, wheels/)
  kernel       felipedeleon11/slotknee-cache-g10t1      (slots_P224_g10t1 = the COVERAGE layout every
                                                        adopted model is fine-tuned on; switched from
                                                        the pre-coverage v2 cache 2026-09-02 so the
                                                        encoder pretrains on the slice distribution it
                                                        will actually see)

No competition source: MAE needs no labels and the pixels come from the cache.
Robust to Kaggle's nested mount layout: every input is LOCATED by walking /kaggle/input,
never by a hard-coded path.  Runs offline: DINOv2 weights come from the code dataset.
Outputs: /kaggle/working/ssl/encoder.safetensors (timm state_dict, drop-in for
`train_slotknee.py --pretrained-path`), encoder_ep10/20/30.safetensors (epoch snapshots for
the ssl_epNN gate arms), mae_full.pt (resume), log.json, ssl_report.json.

Recipe (S5, baked in below): blr 1e-4 (x bs/256), layer-wise lr decay 0.8, mask 0.75, 30 epochs,
snapshots at 10/20/30.  Gate (three UI sessions, not one): ablate-g10 arms ssl_ep10/20/30 at 8 epochs
on folds 0-1 vs recipe_base_s42 / s1337, then a 16-epoch confirm vs the 0.8812 pooled anchor;
adopt only if pooled > both controls by > 0.003 with MCL / lateral meniscus not down.
"""
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
EPOCHS = os.environ.get("SK_EPOCHS", "30")
BS = os.environ.get("SK_BS", "128")
LIMIT_MIN = os.environ.get("SK_LIMIT_MINUTES", "470")   # GPU sessions are capped at 9 h
MAX_IMAGES = os.environ.get("SK_MAX_IMAGES", "")         # set for a smoke run
WORKERS = os.environ.get("SK_WORKERS", "2")
OUT = "/kaggle/working/ssl"
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
    cache = walk_find(lambda b, d, f: b if "train_index.json" in f else None)
    print("code:", code, "\ncache:", cache, flush=True)
    if not (code and cache):
        raise SystemExit("missing input: code/cache not both found under /kaggle/input")
    if not os.path.isfile(os.path.join(code, "scripts", "ssl_pretrain.py")):
        raise SystemExit("the slotknee-code dataset predates scripts/ssl_pretrain.py: "
                         "run `python3 kaggle/kaggle_ops.py package` then push the code dataset again.")
    with open(os.path.join(code, "scripts", "ssl_pretrain.py")) as fh:
        if "--snapshot-epochs" not in fh.read():
            raise SystemExit("the slotknee-code dataset predates the S5 recipe flags (--layer-decay / "
                             "--snapshot-epochs, 2026-09-08): package + push the code dataset again.")

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

    # The cache is a memmap read ~80k times per epoch: local disk beats the network
    # mount.  /kaggle/temp is not preserved, so it does not count against the 20 GB
    # output cap.
    local_cache = os.path.join(TMP, "slots_P224")
    if not os.path.isfile(os.path.join(local_cache, "train_index.json")):
        t = time.time()
        shutil.copytree(cache, local_cache, dirs_exist_ok=True)
        print(f"copied cache to {local_cache} in {time.time()-t:.0f}s", flush=True)

    # MUST match the ADOPTED recipe backbone.  This previously hardcoded the plain
    # (non-reg4) DINOv2 encoder, so the SSL output would NOT load into the reg4 models we
    # actually train — a whole GPU session producing an unusable artefact.  reg4 adds 4
    # register tokens, so the state_dicts are not interchangeable.
    BACKBONE = os.environ.get("SK_BACKBONE", "vit_small_patch14_reg4_dinov2.lvd142m")
    weights = os.path.join(work, "weights", f"{BACKBONE}.safetensors")
    if not os.path.isfile(weights):
        raise SystemExit(f"missing {weights} (SK_BACKBONE={BACKBONE})")
    print(f"SSL pretraining backbone: {BACKBONE}", flush=True)

    # S5 stage-1 recipe, BAKED IN (research_architecture_20260907.md S5; design review 2026-09-08).
    # Not env-driven on purpose: a UI "Save & Run All" cannot set SK_* variables, so anything a
    # T4 run must honour has to be a constant here.  These equal scripts/ssl_pretrain.py's
    # defaults since 2026-09-08 and are repeated so ssl_report.json is self-describing.
    BLR, LAYER_DECAY, MASK_RATIO = "1e-4", "0.8", "0.75"     # blr 1e-4 (NOT the from-scratch 1.5e-4); BEiT layer decay
    SNAPSHOTS = ["10", "20", "30"]                            # encoder_epNN.safetensors, never overwritten -> the ep10/20/30 gate arms
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "scripts/ssl_pretrain.py", "--cache", local_cache,
           "--init-path", weights, "--out", OUT, "--epochs", EPOCHS, "--bs", BS,
           "--workers", WORKERS, "--seed", "42", "--limit-minutes", LIMIT_MIN,
           "--backbone", BACKBONE, "--blr", BLR, "--layer-decay", LAYER_DECAY,
           "--mask-ratio", MASK_RATIO, "--snapshot-epochs", *SNAPSHOTS]
    if MAX_IMAGES:
        cmd += ["--max-images", MAX_IMAGES]
    run(cmd, env=env)

    report = {"elapsed_min": round((time.time() - T0) / 60, 1), "epochs": EPOCHS, "bs": BS,
              "backbone": BACKBONE, "blr": float(BLR), "layer_decay": float(LAYER_DECAY),
              "mask_ratio": float(MASK_RATIO), "snapshot_epochs": [int(e) for e in SNAPSHOTS], "init": "dinov2"}
    for snap in sorted(f for f in os.listdir(OUT) if f.startswith("encoder_ep") and f.endswith(".safetensors")) if os.path.isdir(OUT) else []:
        report.setdefault("snapshots", []).append({"file": snap, "bytes": os.path.getsize(os.path.join(OUT, snap))})
    lf = os.path.join(OUT, "log.json")
    if os.path.isfile(lf):
        with open(lf) as fh:
            report["log"] = json.load(fh)
    with open("/kaggle/working/ssl_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    enc = os.path.join(OUT, "encoder.safetensors")
    print(json.dumps({k: v for k, v in report.items() if k != "log"}),
          "\nencoder:", enc, os.path.getsize(enc) if os.path.isfile(enc) else "MISSING", flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
