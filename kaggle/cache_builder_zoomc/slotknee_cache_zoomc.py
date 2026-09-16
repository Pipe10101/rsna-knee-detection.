"""Kaggle CPU kernel: the CORONAL JOINT-LINE zoom cache on the g10t1 coverage layout.

WHY (main, 2026-08-24): scoring the champion's OOF against the 58 true radiologist-labelled
studies splits the 12 findings into label-limited and MODEL-limited.  The six model-limited
gaps are MCL 0.295, Lateral Meniscus 0.230, Medial Meniscus 0.132, PF OA 0.114, ACL 0.109,
Lateral OA 0.090 — and FOUR of them (MCL, both menisci, both OA compartments) are read on
CORONAL images at the tibiofemoral joint line, while the only zoom view built so far is the
SAGITTAL ACL-notch one (cache_builder_zoomj).  A coronal joint-line zoom is therefore the
highest-value untried INPUT, and it is an input change, not more capacity — the pattern that
made coverage (+0.0325) and reg4 (+0.0220) the only large winners so far.

Layout: the 6 base slots (140 mm) + 2 zoom slots, COR_FS_Z80J and COR_T1_Z80J — 80 mm crops
(0.36 mm/px at P=224) centred on the tibiofemoral joint located by src/slots.locate_joint —
x 10 single-slice anchors x 224 px.  S = 8, cut from the SAME single decode per slice as the
base crops (build_study_tensor locates the joint once per base slot and shares it).
Size: 4,407 x 8 x 10 x 1 x 224 x 224 = 16.5 GiB, under the 20 GB working cap and exactly the
footprint slotknee-cache-zoomj already completed in one session.

Locator health, measured locally on 40 real studies before this kernel was written
(scripts path: build_study_tensor + locate_joint_for_slot on data_subset/train_images):
    COR_FS  40/40 present,  5 fell back to the image centre (12 %), median conf 0.64
    COR_T1  34/40 present, 15 fell back (44 %; "no bone masses" — bright-marrow polarity
            is weaker on T1), median conf 0.89 where it does fire
    SAG_FS  39/40 present,  3 fell back ( 8 %)  <- the rate the shipped zoomj cache runs at
The COR_FS zoom is therefore genuinely joint-centred; median joint offset is +14 mm below the
image centre (|dy| >= 5 mm in 33/35 located slots), so the crop window really moves — it is
not a rescaled duplicate of the base slot.  A fallback does NOT degenerate to the base crop:
it is still an 80 mm zoom, just centred on the image.  Set SLOT_ZOOM_SPEC="COR_FS:80:joint"
to build the COR_FS-only (S = 7, 14.4 GiB) variant if the T1 fallback rate proves harmful.

Runs as a CPU script kernel with the competition attached read-only and the code dataset
(felipedeleon11/slotknee-code) attached under /kaggle/input.

Output (under /kaggle/working, downloadable with `kaggle kernels output`):
    slots_P224_g10t1_zc/train_x.u8 (+ shards), train_mask.u8, train_index.json
    slots_P224_g10t1_zc/test_*  (the 3 example test studies; proves the test path)
    cache_builder_report.json

Nothing here needs the network.
"""
import json
import os
import shutil
import subprocess
import sys
import time

T0 = time.time()
P = int(os.environ.get("SLOT_P", "224"))
G = int(os.environ.get("SLOT_G", "10"))
T = int(os.environ.get("SLOT_T", "1"))
CROP_MM = os.environ.get("SLOT_CROP_MM", "140")
TRIM = os.environ.get("SLOT_TRIM", "0.15")
# Per-entry zoom vocabulary (scripts/build_slot_cache.py --zoom-spec, BASE:MM[:CENTER]);
# names follow slots.zoom_slot_name -> COR_FS_Z80J, COR_T1_Z80J.
ZOOM_SPEC = os.environ.get("SLOT_ZOOM_SPEC", "COR_FS:80:joint,COR_T1:80:joint")
SUFFIX = os.environ.get("SLOT_SUFFIX", "zc")          # cache name tag: slots_P224_g10t1_zc
SKIP = os.environ.get("SLOT_SKIP", "0")               # resume/shard: skip the first N train studies
LIMIT_N = os.environ.get("SLOT_LIMIT_N", "")          # "" = to the end of the sorted uid list
LIMIT_MIN = os.environ.get("SLOT_LIMIT_MINUTES", "500")   # leave margin under the 9 h cap
OUT = (f"/kaggle/working/slots_P{P}"
       + ("" if TRIM == "0.15" else f"_t{int(float(TRIM)*100)}")
       + ("" if (G, T) == (3, 3) else f"_g{G}t{T}")
       + (f"_{SUFFIX}" if SUFFIX else ""))

CODE_CANDIDATES = [
    "/kaggle/input/nidhogg/pytorch/default",
    "/kaggle/input/slotknee-code",
    "/kaggle/input/datasets/felipedeleon11/slotknee-code",
]
PRUNE = ("train_series", "test_series", "train_images", "test_images")


def walk_find(pred, prune=PRUNE):
    for base, dirs, files in os.walk("/kaggle/input"):
        hit = pred(base, list(dirs), files)      # test BEFORE pruning: the predicate may look for train_series
        if hit:
            return hit
        dirs[:] = [d for d in dirs if d not in prune]   # never descend into the 800k-file image trees
    return None


def find_code_dir():
    for c in CODE_CANDIDATES:
        if os.path.isdir(os.path.join(c, "src")) and os.path.isdir(os.path.join(c, "scripts")):
            return c
    hit = walk_find(lambda b, d, f: b if ("build_slot_cache.py" in f and os.path.basename(b) == "scripts") else None)
    if not hit:
        raise SystemExit("code dataset not found under /kaggle/input")
    return os.path.dirname(hit)


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main():
    code = find_code_dir()
    work_code = "/kaggle/working/code"
    if os.path.isdir(work_code):
        shutil.rmtree(work_code)
    shutil.copytree(code, work_code, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.chdir(work_code)
    sys.path.insert(0, work_code)

    print("input mounts:", os.listdir("/kaggle/input"), flush=True)

    # Fail fast if the attached code dataset predates --zoom-spec: the whole point of this
    # kernel is the per-entry zoom vocabulary, and a stale dataset would otherwise build a
    # plain 6-slot cache under the zc name and quietly poison every arm gated on it.
    with open(os.path.join(work_code, "scripts", "build_slot_cache.py")) as fh:
        if "--zoom-spec" not in fh.read():
            raise SystemExit("attached slotknee-code predates --zoom-spec; run "
                             "`kaggle_ops.py package && kaggle_ops.py push-code` first")
    from src import slots as _slots           # validates the spec (names/mm/centre) before any I/O
    _names = _slots.slot_names_ext(zoom_spec=ZOOM_SPEC)
    print("zoom_spec=%s -> S=%d %s" % (ZOOM_SPEC, len(_names), _names), flush=True)

    # Newer Kaggle images nest sources (/kaggle/input/competitions/<slug>,
    # /kaggle/input/datasets/<user>/<slug>); the bootstrap scans one level, so
    # locate the competition root here and hand it over explicitly.
    comp_root = walk_find(lambda b, d, f: b if ("train.csv" in f and "train_series" in d) else None)
    print("competition root:", comp_root, flush=True)
    boot_cmd = [sys.executable, "scripts/kaggle_bootstrap.py"]
    if comp_root:
        boot_cmd += ["--competition-dir", comp_root]

    # Writable data_dir whose image dirs are symlinks into the read-only mount.
    boot = subprocess.run(boot_cmd, capture_output=True, text=True)
    print(boot.stdout, flush=True)
    if boot.returncode != 0:
        print("kaggle_bootstrap.py FAILED, stderr:\n" + boot.stderr, flush=True)
        raise SystemExit(1)
    data_dir = None
    for line in boot.stdout.splitlines():
        if "data_dir=" in line:
            data_dir = line.split("data_dir=", 1)[1].strip().split()[0]
    if not data_dir or not os.path.isdir(data_dir):
        raise SystemExit("kaggle_bootstrap.py did not report a data_dir")

    common = ["--data-dir", data_dir, "--out", OUT, "--P", str(P), "--G", str(G), "--T", str(T),
              "--crop-mm", str(CROP_MM), "--trim-frac", str(TRIM), "--shard-size", "512"]
    if ZOOM_SPEC:
        common += ["--zoom-spec", ZOOM_SPEC]

    # --skip/--limit select this kernel's slice of the TRAIN studies; the 3-study test split
    # is built whole, and only when nothing is skipped (a skipping run would step past it).
    train_only = []
    if LIMIT_N:
        train_only += ["--limit", LIMIT_N]
    if SKIP and SKIP != "0":
        train_only += ["--skip", SKIP]

    # Test path first: 3 example studies, seconds, proves the inference-time builder works
    # on the real test layout (test_series/ + test_series.csv) with these zoom slots.
    if not (SKIP and SKIP != "0"):
        run([sys.executable, "scripts/build_slot_cache.py", "--split", "test"] + common)

    # Train: the long one.  --limit-minutes makes it stop cleanly and resumably.
    run([sys.executable, "scripts/build_slot_cache.py", "--split", "train",
         "--limit-minutes", str(LIMIT_MIN)] + common + train_only)

    report = {"elapsed_min": round((time.time() - T0) / 60, 1), "P": P, "G": G, "T": T,
              "crop_mm": CROP_MM, "zoom_spec": ZOOM_SPEC, "slot_names": _names,
              "skip": SKIP, "limit": LIMIT_N, "out": OUT}
    for split in ("train", "test"):
        idx = os.path.join(OUT, f"{split}_index.json")
        if os.path.isfile(idx):
            with open(idx) as fh:
                j = json.load(fh)
            report[split] = {"n_studies": len(j.get("studies", [])), "stats": j.get("stats", {})}
    total = 0
    for root, _, files in os.walk(OUT):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    report["output_gb"] = round(total / 2**30, 2)
    with open("/kaggle/working/cache_builder_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    shutil.rmtree(work_code, ignore_errors=True)   # keep the output small


if __name__ == "__main__":
    main()
