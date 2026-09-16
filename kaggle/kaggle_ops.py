"""Package the code dataset and drive the SlotKnee kernels on Kaggle (cache builders, train, ablate, ssl, submit).

Uses the Kaggle *Python* API (the CLI binary is not installed on this machine).
Credentials: ~/.kaggle/kaggle.json.  Every action is explicit; nothing runs unless you
name it.

    python3 kaggle/kaggle_ops.py package        # kaggle/slotknee_code/ <- src/, scripts/, labels, DINOv2 weights, wheels
    python3 kaggle/kaggle_ops.py push-code      # version the private dataset felipedeleon11/slotknee-code
    python3 kaggle/kaggle_ops.py push-kernel    # cache builder (CPU)   -> felipedeleon11/slotknee-cache-builder
    python3 kaggle/kaggle_ops.py push-train     # 5-fold training (GPU) -> felipedeleon11/slotknee-train
    python3 kaggle/kaggle_ops.py push-submit    # inference (GPU, no internet) -> felipedeleon11/slotknee-submit
    python3 kaggle/kaggle_ops.py status [--kernel train|submit|cache]
    python3 kaggle/kaggle_ops.py log    [--kernel ...]          # print the last stderr/stdout lines
    python3 kaggle/kaggle_ops.py fetch  [--kernel ...] --out DIR  # download a kernel's output
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "felipedeleon11"
CODE_SLUG = "slotknee-code"
KERNELS = {
    "cache": ("slotknee-cache-builder", os.path.join(ROOT, "kaggle", "cache_builder")),
    "cache252": ("slotknee-cache-p252", os.path.join(ROOT, "kaggle", "cache_builder_p252")),
    "cachet30": ("slotknee-cache-t30", os.path.join(ROOT, "kaggle", "cache_builder_t30")),
    "cacheg10": ("slotknee-cache-g10t1", os.path.join(ROOT, "kaggle", "cache_builder_g10t1")),
    "cacheg10t35": ("slotknee-cache-g10t1-t35", os.path.join(ROOT, "kaggle", "cache_builder_g10t1_t35")),   # central-block anchors (trim 0.35)
    "cachezoom": ("slotknee-cache-zoom", os.path.join(ROOT, "kaggle", "cache_builder_zoom")),
    "cachezoomj": ("slotknee-cache-zoomj", os.path.join(ROOT, "kaggle", "cache_builder_zoomj")),   # joint-centred zoom, g10t1 layout (step 6)
    "cachezoomc": ("slotknee-cache-zoomc", os.path.join(ROOT, "kaggle", "cache_builder_zoomc")),   # CORONAL joint-line zoom (COR_FS/COR_T1 @80 mm), g10t1 layout -> slots_P224_g10t1_zc
    "cachezoomm": ("slotknee-cache-zoomm", os.path.join(ROOT, "kaggle", "cache_builder_zoomm")),   # CORONAL medial-column zoom (COR_FS @80 mm, centre +20 distal/32 medial of the joint) -> slots_P224_g10t1_zm
    "cacheg20a": ("slotknee-cache-g20a", os.path.join(ROOT, "kaggle", "cache_builder_g20a")),   # G=20 coverage cache, first 2204 studies
    "cacheg20b": ("slotknee-cache-g20b", os.path.join(ROOT, "kaggle", "cache_builder_g20b")),   # G=20 coverage cache, studies 2205..4407
    "cache336": ("slotknee-cache-p336g4", os.path.join(ROOT, "kaggle", "cache_builder_p336")),
    # G=10, T=3 coverage cache (39.8 GB): three parts, merged at run time by the ablate kernel.
    "cacheg10t3a": ("slotknee-cache-g10t3a", os.path.join(ROOT, "kaggle", "cache_builder_g10t3a")),
    "cacheg10t3b": ("slotknee-cache-g10t3b", os.path.join(ROOT, "kaggle", "cache_builder_g10t3b")),
    "cacheg10t3c": ("slotknee-cache-g10t3c", os.path.join(ROOT, "kaggle", "cache_builder_g10t3c")),
    # P=336 ON the coverage layout (G=10): 4407 x 6 x 10 x 336^2 = 29.85 GB > Kaggle's 20 GB output
    # cap, so it ships as two halves (14.93 + 14.92 GB) merged by scripts/merge_slot_cache.py.
    # "cachep336g10" is an ALIAS for part A so the plan's name resolves; push BOTH halves.
    "cachep336g10a": ("slotknee-cache-p336g10a", os.path.join(ROOT, "kaggle", "cache_builder_p336g10a")),
    "cachep336g10b": ("slotknee-cache-p336g10b", os.path.join(ROOT, "kaggle", "cache_builder_p336g10b")),
    "cachep336g10": ("slotknee-cache-p336g10a", os.path.join(ROOT, "kaggle", "cache_builder_p336g10a")),
    "train": ("slotknee-train", os.path.join(ROOT, "kaggle", "train_kernel")),
    "traing10": ("slotknee-train-g10", os.path.join(ROOT, "kaggle", "train_kernel_g10")),
    "submit": ("slotknee-submit", os.path.join(ROOT, "kaggle", "submit_kernel")),
    "submiteff": ("slotknee-submit-eff", os.path.join(ROOT, "kaggle", "submit_kernel_eff")),   # efficiency entry: student only, TTA off
    "ablate": ("slotknee-ablate", os.path.join(ROOT, "kaggle", "ablate_kernel")),
    "ablateg10": ("slotknee-ablate-g10", os.path.join(ROOT, "kaggle", "ablate_kernel_g10")),   # arms on the coverage layout (zoomj / p336 / head fix)
    "trainfinal": ("slotknee-train-final", os.path.join(ROOT, "kaggle", "train_kernel_final")),   # 5 folds x 10 epochs of the winning recipe
    "trainstudent": ("slotknee-train-student", os.path.join(ROOT, "kaggle", "train_kernel_student")),   # ONE full-fit distilled student: the efficiency entry
    "ssl": ("slotknee-ssl", os.path.join(ROOT, "kaggle", "ssl_kernel")),
    "decodebench": ("slotknee-decode-bench", os.path.join(ROOT, "kaggle", "decode_bench")),   # CPU: pydicom vs dicomsdl on the real test path
}
CODE_DIR = os.path.join(ROOT, "kaggle", "slotknee_code")
HF_DINOV2 = os.path.expanduser(
    "~/.cache/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m")


def api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    a = KaggleApi()
    a.authenticate()
    return a


def package():
    """Copy exactly what the kernels need; nothing else."""
    if os.path.isdir(CODE_DIR):
        shutil.rmtree(CODE_DIR)
    os.makedirs(CODE_DIR)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store")
    shutil.copytree(os.path.join(ROOT, "src"), os.path.join(CODE_DIR, "src"), ignore=ignore)
    shutil.copytree(os.path.join(ROOT, "scripts"), os.path.join(CODE_DIR, "scripts"), ignore=ignore)
    labels_src = os.path.join(ROOT, "data_subset", "labels_external")
    if os.path.isdir(labels_src):
        shutil.copytree(labels_src, os.path.join(CODE_DIR, "labels_external"), ignore=ignore)
    gold = os.path.join(ROOT, "data_subset", "train_gold.csv")   # 58 rows + stratified folds
    if os.path.isfile(gold):
        os.makedirs(os.path.join(CODE_DIR, "labels_external"), exist_ok=True)
        shutil.copy(gold, os.path.join(CODE_DIR, "labels_external", "train_gold.csv"))
    # Studies whose DICOMs are on this laptop's disk.  src/folds.build_groups fingerprints the
    # scanner of every study it finds under <data_dir>/train_images and hashes the report for
    # the rest, so the fold split depends on WHICH studies are on disk.  The training kernels
    # expose exactly these uids (per-uid symlinks into the competition's train_series/) so
    # Kaggle reproduces the laptop folds and OOFs stay comparable across machines.
    img_dir = os.path.join(ROOT, "data_subset", "train_images")
    if os.path.isdir(img_dir):
        uids = sorted(u for u in os.listdir(img_dir)
                      if not u.startswith(".") and os.path.isdir(os.path.join(img_dir, u)))
        os.makedirs(os.path.join(CODE_DIR, "labels_external"), exist_ok=True)
        with open(os.path.join(CODE_DIR, "labels_external", "fold_image_uids.txt"), "w") as fh:
            fh.write("\n".join(uids) + "\n")
        print(f"fold_image_uids.txt: {len(uids)} laptop-resident studies (fold grouping on Kaggle)")
    req = os.path.join(ROOT, "requirements.txt")
    if os.path.isfile(req):
        shutil.copy(req, CODE_DIR)
    # DINOv2-S weights: Kaggle runs offline, timm cannot download.  The HF blob IS the
    # safetensors file; give it the name timm expects so _resolve_weights_file finds it.
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    # NOTE: this function rmtree's CODE_DIR first, so weights staged by hand into
    # kaggle/slotknee_code/weights/ are DESTROYED by the next package().  Any encoder an arm
    # needs must be listed here, not copied in manually.
    wanted = ["vit_small_patch14_dinov2.lvd142m",          # the baseline encoder (required)
              "vit_small_patch14_reg4_dinov2.lvd142m",     # ADOPTED recipe backbone (2026-08-24)
              "vit_base_patch14_dinov2.lvd142m",           # vitb arm (Phase B)
              "vit_small_patch16_dinov3.lvd1689m"]         # dinov3s arm (Meta licence — see POLICY 4)
    for model in wanted:
        blobs = glob.glob(os.path.join(hub, f"models--timm--{model}", "blobs", "*"))
        blobs = [b for b in blobs if os.path.getsize(b) > 10 * 2**20]
        if blobs:
            os.makedirs(os.path.join(CODE_DIR, "weights"), exist_ok=True)
            shutil.copy(blobs[0], os.path.join(CODE_DIR, "weights", f"{model}.safetensors"))
            print(f"weights: {model} ({os.path.getsize(blobs[0])/2**20:.0f} MB)")
        elif model == wanted[0]:
            print("WARNING: DINOv2-S weights not found in the HF cache; training on Kaggle will fail")
        else:
            print(f"note: {model} not in the HF cache; its arm cannot run on Kaggle until packaged")
    # Licences MUST travel with the weights.  Meta's DINOv3 licence s1.b.i requires "a copy of
    # this Agreement with any such DINO Materials", and rule 2.8(a) makes the weights a PUBLIC
    # Kaggle dataset at prize verification — so shipping them bare would be a breach exactly
    # when it counts.  DINOv2 is Apache-2.0 (no such obligation) but is listed for provenance.
    lic_src = os.path.join(ROOT, "kaggle", "licenses")
    if os.path.isdir(lic_src) and os.path.isdir(os.path.join(CODE_DIR, "weights")):
        dst = os.path.join(CODE_DIR, "weights", "LICENSES")
        shutil.copytree(lic_src, dst, ignore=ignore)
        print(f"licences: {', '.join(sorted(os.listdir(dst)))} -> weights/LICENSES/")
    elif any("dinov3" in m for m in wanted):
        print("WARNING: DINOv3 weights packaged with no licence copy — required by its licence s1.b.i")
    wheels = os.path.join(ROOT, "kaggle", "wheels")
    if os.path.isdir(wheels):
        shutil.copytree(wheels, os.path.join(CODE_DIR, "wheels"))
    meta = {"title": "SlotKnee code", "id": f"{USER}/{CODE_SLUG}",
            "licenses": [{"name": "CC0-1.0"}]}
    with open(os.path.join(CODE_DIR, "dataset-metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(CODE_DIR) for f in fs)
    print(f"packaged {CODE_DIR}: {total/1e6:.1f} MB")


def push_code():
    a = api()
    try:
        a.dataset_create_version(CODE_DIR, version_notes=time.strftime("%Y-%m-%d %H:%M"),
                                 quiet=False, dir_mode="zip")
        print("versioned existing dataset", f"{USER}/{CODE_SLUG}")
    except Exception as e:  # first push
        print("create_version failed (", str(e)[:120], ") -> creating new dataset")
        a.dataset_create_new(CODE_DIR, public=False, quiet=False, dir_mode="zip")
        print("created dataset", f"{USER}/{CODE_SLUG}")


MODELS_SLUG = "slotknee-models"


def push_models(models_dirs):
    """Upload fold checkpoints as the private dataset felipedeleon11/slotknee-models.

    ``models_dirs``: one or more ``[prefix=]DIR`` entries.  Every ``fold_*_best.pt`` under DIR is
    staged FLAT as ``<prefix>_fold_k_best.pt`` (plain ``fold_k_best.pt`` when no prefix), so several
    model sets (e.g. the g10t1 coverage folds and the old 224/3x3 folds) travel in ONE dataset
    version and the submit kernel, which globs ``*_best.pt`` recursively and groups checkpoints by
    their recorded input layout, rank-averages the views.
    """
    if isinstance(models_dirs, str):
        models_dirs = [models_dirs]
    stage = os.path.join(ROOT, "kaggle", "slotknee_models")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    staged = []
    for entry in models_dirs:
        prefix, _, d = entry.rpartition("=")
        ckpts = sorted(glob.glob(os.path.join(d, "fold_*_best.pt")))
        if not ckpts:
            raise SystemExit(f"no fold_*_best.pt under {d}")
        for c in ckpts:
            name = (prefix + "_" if prefix else "") + os.path.basename(c)
            shutil.copy(c, os.path.join(stage, name)); staged.append(name)
        for extra in glob.glob(os.path.join(d, "fold_*_log.json")):
            shutil.copy(extra, os.path.join(stage, (prefix + "_" if prefix else "") + os.path.basename(extra)))
    with open(os.path.join(stage, "dataset-metadata.json"), "w") as fh:
        json.dump({"title": "SlotKnee models", "id": f"{USER}/{MODELS_SLUG}",
                   "licenses": [{"name": "CC0-1.0"}]}, fh, indent=2)
    total = sum(os.path.getsize(os.path.join(stage, f)) for f in os.listdir(stage))
    print(f"staged {len(staged)} checkpoints ({total/2**20:.0f} MB):", staged, flush=True)
    a = api()
    try:
        a.dataset_create_version(stage, version_notes=time.strftime("%Y-%m-%d %H:%M"), quiet=False)
        print("versioned", f"{USER}/{MODELS_SLUG}", "with", staged)
    except Exception as e:
        print("create_version failed (", str(e)[:120], ") -> creating new dataset")
        a.dataset_create_new(stage, public=False, quiet=False)
        print("created", f"{USER}/{MODELS_SLUG}")


def push_kernel(which):
    slug, folder = KERNELS[which]
    r = api().kernels_push(folder)
    ver = getattr(r, "versionNumber", None) or getattr(r, "version_number", None)
    err = getattr(r, "error", None)
    if err or not ver:
        # Kaggle reports refusals here, not as exceptions (e.g. "Maximum batch GPU session
        # count of 2 reached." while two GPU kernels run); version 0 means nothing was created.
        print("PUSH FAILED", f"{USER}/{slug}:", err or "no version returned", file=sys.stderr)
        return None
    print("pushed kernel", f"{USER}/{slug}", "version", ver)
    return ver


def status(which):
    slug, _ = KERNELS[which]
    r = api().kernels_status(f"{USER}/{slug}")
    print(r)
    return r


def fetch(which, out):
    slug, _ = KERNELS[which]
    os.makedirs(out, exist_ok=True)
    api().kernels_output(f"{USER}/{slug}", path=out, force=True, quiet=False)
    print("downloaded to", out)


def _expected_shards(cache_dir):
    """{filename: expected bytes} from <cache_dir>/train_index.json (+ mask/test files)."""
    idx = os.path.join(cache_dir, "train_index.json")
    if not os.path.isfile(idx):
        return {}
    j = json.load(open(idx))
    P, G, T = j["P"], j["G"], j["T"]
    S = len(j.get("slot_names") or []) or 6
    per = S * G * T * P * P
    exp = {s["x"]: s["n"] * per for s in j["shards"]}
    exp["train_mask.u8"] = len(j["studies"]) * S
    return exp


def fetch_resumable(which, out, rounds=12):
    """kernels_output with skip-existing, repeated until every shard matches its index size.

    Kaggle's output endpoint drops long transfers; a 12 GB cache rarely survives one
    pass.  Short/zero files are deleted before each retry so they get re-fetched.
    """
    slug, _ = KERNELS[which]
    os.makedirs(out, exist_ok=True)
    a = api()
    for r in range(1, rounds + 1):
        # delete files whose size does not match the index (once the index exists)
        for sub in glob.glob(os.path.join(out, "slots_*")):
            exp = _expected_shards(sub)
            for name, nbytes in exp.items():
                p = os.path.join(sub, name)
                if os.path.exists(p) and os.path.getsize(p) != nbytes:
                    print(f"round {r}: removing short file {name} ({os.path.getsize(p)} != {nbytes})", flush=True)
                    os.remove(p)
        try:
            a.kernels_output(f"{USER}/{slug}", path=out, force=False, quiet=False)
        except Exception as e:
            print(f"round {r}: transfer error {type(e).__name__}: {str(e)[:120]}", flush=True)
        ok = True
        for sub in glob.glob(os.path.join(out, "slots_*")):
            exp = _expected_shards(sub)
            if not exp:
                ok = False; continue
            for name, nbytes in exp.items():
                p = os.path.join(sub, name)
                if not (os.path.exists(p) and os.path.getsize(p) == nbytes):
                    ok = False
        if ok and glob.glob(os.path.join(out, "slots_*")):
            print("downloaded to", out, "(all shards verified)", flush=True)
            return True
        time.sleep(20)
    print("FETCH INCOMPLETE after", rounds, "rounds", flush=True)
    return False


def _kernel_output_files(a, slug):
    """[(file_name, url)] for a kernel's output, via the SDK listing (signed URLs)."""
    from kaggle.api.kaggle_api_extended import ApiListKernelSessionOutputRequest
    files = []
    token = None
    with a.build_kaggle_client() as kaggle:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name = USER
            req.kernel_slug = slug
            if token:
                req.page_token = token
            resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
            files += [(it.file_name, it.url) for it in resp.files]
            token = getattr(resp, "next_page_token", None)
            if not token:
                break
    return files


def _download_range(url, dst, expected=None, chunk=8 << 20, retries=30):
    """Streamed download with HTTP Range resume; never holds a file in RAM."""
    import requests
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    for attempt in range(retries):
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        if expected is not None and have == expected:
            return True
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=(20, 120)) as r:
                if r.status_code == 416:           # already complete
                    return True
                if have and r.status_code != 206:  # server ignored Range: restart the file
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                r.raise_for_status()
                total = None
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr:
                    total = int(cr.split("/")[-1])
                elif r.headers.get("Content-Length") and not have:
                    total = int(r.headers["Content-Length"])
                with open(dst, mode) as fh:
                    for part in r.iter_content(chunk_size=chunk):
                        if part:
                            fh.write(part)
            size = os.path.getsize(dst)
            target = expected if expected is not None else total
            if target is None or size == target:
                return True
            print(f"  {os.path.basename(dst)}: {size}/{target} bytes, resuming", flush=True)
        except Exception as e:
            print(f"  {os.path.basename(dst)}: attempt {attempt+1} {type(e).__name__}: {str(e)[:80]}", flush=True)
            time.sleep(min(60, 5 * (attempt + 1)))
    return False


def fetch_robust(which, out):
    """Per-file, range-resumable download of a kernel's output with index-size verification.

    The stock kernels_output() reads each file fully into RAM and restarts a shard from
    zero after every connection reset, which Kaggle's output endpoint does often on
    multi-GB transfers.  This streams to disk and resumes from the last byte.
    """
    slug, _ = KERNELS[which]
    os.makedirs(out, exist_ok=True)
    a = api()
    files = _kernel_output_files(a, slug)
    print(f"{slug}: {len(files)} output files", flush=True)
    # small files (indexes) first so expected shard sizes are known before the big ones
    files.sort(key=lambda t: (1 if "train_x" in t[0] else 0, t[0]))
    ok_all = True
    for name, url in files:
        dst = os.path.join(out, name)
        sub = os.path.dirname(dst)
        exp = _expected_shards(sub) if os.path.basename(sub).startswith("slots_") else {}
        expected = exp.get(os.path.basename(dst))
        ok = _download_range(url, dst, expected=expected)
        if not ok:
            # signed URL may have expired: refresh the listing once and retry
            refreshed = dict(_kernel_output_files(a, slug))
            ok = _download_range(refreshed.get(name, url), dst, expected=expected)
        ok_all &= ok
        print(f"  {'ok ' if ok else 'FAIL'} {name} ({os.path.getsize(dst)/2**30:.2f} GiB)", flush=True)
    print(("downloaded to " + out + " (all shards verified)") if ok_all else "FETCH INCOMPLETE", flush=True)
    return ok_all


def log(which, n=40):
    slug, _ = KERNELS[which]
    d = os.path.join(ROOT, "kaggle", ".logs", slug)
    os.makedirs(d, exist_ok=True)
    api().kernels_output(f"{USER}/{slug}", path=d, force=True, quiet=True)
    lf = os.path.join(d, f"{slug}.log")
    if not os.path.isfile(lf):
        print("no log file in output"); return
    raw = open(lf, errors="replace").read()
    try:
        ev = json.loads(raw)                      # the log is a JSON array of events
    except Exception:
        ev = []
        for line in raw.splitlines():             # fallback: one event per line, tolerate junk
            line = line.strip().lstrip(",").rstrip("]").rstrip(",")
            if line.startswith("{"):
                try: ev.append(json.loads(line))
                except Exception: pass
    for e in ev[-n:]:
        print(f"[{e['stream_name'][:3]}] {e['data']}", end="")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["package", "push-code", "push-kernel", "push-cache252", "push-cachet30", "push-cacheg10", "push-cacheg10t35", "push-cachezoom", "push-cachezoomj", "push-cachezoomc", "push-cache-zoomc", "push-cachezoomm", "push-cacheg20a", "push-cacheg20b", "push-cache336",
                                      "push-cacheg10t3a", "push-cacheg10t3b", "push-cacheg10t3c",
                                      "push-cache-p336g10", "push-cache-p336g10a", "push-cache-p336g10b",
                                      "push-train", "push-train-g10",
                                      "push-submit", "push-submit-eff", "push-ablate", "push-ablate-g10", "push-train-final", "push-train-student", "push-ssl", "push-decode-bench", "push-models", "status", "log", "fetch"])
    p.add_argument("--kernel", default=None, choices=list(KERNELS))
    p.add_argument("--out", default=os.path.join(ROOT, "cache", "slots_P224_full"))
    p.add_argument("--models-dir", action="append", default=None,
                   help="[prefix=]DIR with fold_*_best.pt; repeatable (each set gets its prefix in the flat dataset)")
    p.add_argument("-n", type=int, default=40)
    args = p.parse_args()
    if args.action == "push-models":
        push_models(args.models_dir or [os.path.join(ROOT, "models", "slotknee_full5")])
        return
    which = args.kernel or {"push-kernel": "cache", "push-cache252": "cache252", "push-cachet30": "cachet30", "push-cacheg10": "cacheg10", "push-cacheg10t35": "cacheg10t35", "push-cachezoom": "cachezoom", "push-cachezoomj": "cachezoomj",
                            "push-cachezoomc": "cachezoomc", "push-cache-zoomc": "cachezoomc",   # both spellings; the hyphenated one is what the plan named
                            "push-cachezoomm": "cachezoomm",
                            "push-cacheg20a": "cacheg20a", "push-cacheg20b": "cacheg20b", "push-cache336": "cache336",
                            "push-cacheg10t3a": "cacheg10t3a", "push-cacheg10t3b": "cacheg10t3b", "push-cacheg10t3c": "cacheg10t3c",
                            # every choice above MUST also appear here: push-cache336 once shipped in this
                            # dict without being in the choices list and could not be invoked at all.
                            "push-cache-p336g10": "cachep336g10", "push-cache-p336g10a": "cachep336g10a",
                            "push-cache-p336g10b": "cachep336g10b",
                            "push-train": "train", "push-train-g10": "traing10",
                            "push-submit": "submit", "push-submit-eff": "submiteff", "push-ablate": "ablate", "push-ablate-g10": "ablateg10", "push-train-final": "trainfinal", "push-train-student": "trainstudent", "push-ssl": "ssl",
                            "push-decode-bench": "decodebench"}.get(args.action, "cache")
    if args.action == "package":
        package()
    elif args.action == "push-code":
        push_code()
    elif args.action.startswith("push-"):   # any kernel push: push-kernel, push-train, push-ablate, push-cache*, ...
        push_kernel(which)
    elif args.action == "status":
        status(which)
    elif args.action == "log":
        log(which, args.n)
    elif args.action == "fetch":
        fetch_robust(which, args.out)


if __name__ == "__main__":
    main()
