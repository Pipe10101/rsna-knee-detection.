"""Kaggle CPU kernel: settle how much a faster DICOM decoder saves on the test path.

Decoding dominates the submit kernel's wall-clock (measured: 3607 ms/study decode vs
1176 ms model, ~70%), so the decoder is the biggest efficiency lever available.  Locally
`dicomsdl` is 3.2x faster per slice but only 1.24x per study, because every local file is
uncompressed 'Explicit VR Little Endian'.  The open question is what the *competition's*
files look like on Kaggle - if the test set is JPEG2000 / JPEG-Lossless the gain should be
far larger, which is exactly why the submit kernel installs the pylibjpeg decoders.

What this kernel reports (all to stdout + /kaggle/working/decode_bench_report.json):
  1. transfer-syntax histogram of the visible TEST studies (and a sample of TRAIN studies)
  2. ms/study for `src.slots.build_study_tensor(P, G, T)` with pydicom vs dicomsdl, the
     ratio, and the projected minutes for a 1300-study hidden test set
  3. pixel parity between the two decoders on a sample of test slices
  4. the 3-anchor-shift TTA build the way inference actually does it, old (index walked per
     shift) vs new (`index_study` once, `index=` reused), each on a COLD study and a WARM
     one -- this sizes the index-reuse win, which is NOT 2 x the cold index walk: only the
     first walk of a study is cold, shifts 2 and 3 hit a warm page cache.

Section 4 runs FIRST, before the histogram and the A/B, because those touch every study and
would destroy the only cold cache we get.  Its two cold cells use DIFFERENT studies so
neither warms the other, alternating which study of each pair takes which condition.

Attached sources (kernel-metadata.json):
  competition  rsna-knee-abnormality-detection   (test.csv, test_series.csv, test_series/)
  dataset      felipedeleon11/slotknee-code      (src/, scripts/, wheels/)
  dataset      felipedeleon11/slotknee-wheels    (pylibjpeg trio + dicomsdl wheels)

NOTE ON WHEELS: `kaggle/slotknee_code/wheels` (-> the slotknee-code dataset) is the CANONICAL
wheel location.  felipedeleon11/slotknee-wheels exists only because this kernel was built
while slotknee-code still lacked the dicomsdl wheel; it is a convenience mirror.  Add new
wheels to kaggle/wheels/ + re-push slotknee-code, and refresh the mirror only if it is still
attached here, so the two cannot drift.

The code dataset currently on Kaggle predates `slots.set_decoder`, so this kernel installs
the same switch itself (`_install_decoder_shim`, body copied verbatim from src/slots.py's
`_decode_raw_dicomsdl`) and prefers the shipped implementation whenever the dataset does
carry one.  Either way what gets timed is main's decode path, and the shim counts how many
slices dicomsdl actually decoded natively vs silently fell back to pydicom - without that
counter a 1.0x ratio would be indistinguishable from "dicomsdl never ran".
"""
import json
import os
import random
import shutil
import subprocess
import sys
import time

T0 = time.time()
P = int(os.environ.get("SK_P", "224"))
G = int(os.environ.get("SK_G", "10"))
T = int(os.environ.get("SK_T", "1"))
HIDDEN_STUDIES = int(os.environ.get("SK_HIDDEN_STUDIES", "1300"))   # projection target
TRAIN_SAMPLE = int(os.environ.get("SK_TRAIN_SAMPLE", "40"))         # 3 visible test studies is a thin timer
PARITY_SLICES = int(os.environ.get("SK_PARITY_SLICES", "20"))
TRAIN_SYNTAX_SCAN = int(os.environ.get("SK_TRAIN_SYNTAX_SCAN", "400"))
SHIFTS = tuple(int(s) for s in os.environ.get("SK_SHIFTS", "0,1,-1").split(","))   # submit kernel's TTA
COLD_PAIRS = int(os.environ.get("SK_COLD_PAIRS", "12"))   # train study PAIRS for the cold cells (2 studies each)


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


def find_wheel_dirs():
    """Every directory under /kaggle/input holding .whl files (wheels may live in either dataset)."""
    out = []
    for base, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series", "train_images", "test_images")]
        if any(f.endswith(".whl") for f in files):
            out.append(base)
    return out


def install_wheels():
    dirs = find_wheel_dirs()
    print("wheel dirs:", dirs, flush=True)
    if not dirs:
        print("WARNING: no wheels found; dicomsdl will be unavailable", flush=True)
        return
    links = []
    for d in dirs:
        links += ["--find-links", d]
    # pylibjpeg trio: JPEG-Lossless / JPEG2000 handlers for pydicom (the hidden test set may
    # need them).  dicomsdl: the fast decoder under test.  Both installs are non-fatal.
    run([sys.executable, "-m", "pip", "install", "--no-index"] + links +
        ["pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg", "-q"], check=False)
    run([sys.executable, "-m", "pip", "install", "--no-index"] + links + ["dicomsdl", "-q"], check=False)


def install_decoder_shim(slots):
    """Return (set_decoder, counters) for a shipped slots.py that may predate the switch.

    Rebinds the module-global ``slots._decode_raw``; ``build_study_tensor`` and
    ``_decode_crops`` both resolve it as a global, so the whole decode path follows.
    """
    import numpy as np
    orig = slots._decode_raw
    if getattr(slots, "_DECODER", None) is not None:
        slots._DECODER = "pydicom"           # keep any shipped switch out of the way
    state = {"decoder": "pydicom", "native": 0, "fallback": 0}

    def _dicomsdl_raw(path):
        """Verbatim from src/slots.py::_decode_raw_dicomsdl (parity-tested there)."""
        import dicomsdl
        d = dicomsdl.open(path)
        a = d.pixelData(storedvalue=True)
        info = d.getPixelDataInfo()
        syntax = str(getattr(d, "TransferSyntaxUID", "") or "unknown")
        try:                                   # prefer the human-readable pydicom name
            from pydicom.uid import UID
            syntax = str(UID(syntax).name)
        except Exception:
            pass
        if a is None:
            raise ValueError("dicomsdl returned no pixel data")
        a = np.asarray(a)
        if str(info.get("PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            a = a.max() - a
        spacing = None
        ps = getattr(d, "PixelSpacing", None)
        if ps is not None and len(ps) == 2:
            r, c = slots._to_float(ps[0]), slots._to_float(ps[1])
            spacing = None if (r is None or c is None) else (r, c)
        return a, spacing, syntax

    fast = getattr(slots, "_decode_raw_dicomsdl", None) or _dicomsdl_raw

    def _decode_raw(path):
        if state["decoder"] == "dicomsdl":
            try:
                a, spacing, syntax = fast(path)
                if a.ndim == 3:                # same multi-frame / RGB handling as pydicom's
                    if a.shape[-1] in (3, 4) and a.shape[0] != a.shape[-1]:
                        a = a[..., :3].mean(axis=-1)
                    else:
                        a = a[a.shape[0] // 2]
                if a.ndim == 2 and a.size:
                    state["native"] += 1
                    return a, spacing, syntax
            except Exception:
                pass                           # any surprise -> the proven pydicom path
            state["fallback"] += 1
        return orig(path)

    def set_decoder(name):
        if name == "dicomsdl":
            try:
                import dicomsdl  # noqa: F401
            except Exception:
                print("decoder: dicomsdl not importable; staying on pydicom", flush=True)
                name = "pydicom"
        state["decoder"] = name
        return name

    slots._decode_raw = _decode_raw
    print("decoder switch: %s fast path, shim dispatcher"
          % ("shipped slots._decode_raw_dicomsdl" if fast is not _dicomsdl_raw else "kernel-local"), flush=True)
    return set_decoder, state


def syntax_of(path):
    import pydicom
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, specific_tags=["SOPInstanceUID"])
        return str(ds.file_meta.TransferSyntaxUID.name)
    except Exception as e:
        return "unreadable(%s)" % type(e).__name__


def study_files(study_dir):
    out = []
    for base, _, files in os.walk(study_dir):
        out += [os.path.join(base, f) for f in sorted(files) if not f.startswith(".")]
    return sorted(out)


def hist_add(h, k, n=1):
    h[k] = h.get(k, 0) + n


def three_shift(slots, study_dir, lookup, mode):
    """One study built for every TTA anchor shift, the OLD way or the NEW way.

    mode "old": ``build_study_tensor`` per shift with no ``index=`` -> the header walk is
                repeated once per shift (what shipped before the fix).
    mode "new": ``index_study`` once, ``index=`` passed to every shift -- mirrors the fixed
                ``_decode_study`` in scripts/infer_slotknee.py (owned elsewhere; replicated
                here rather than imported so this measurement cannot drift with its API).
    Hashing happens after the clock stops so it never lands in the timing.
    """
    import hashlib
    xs, per = [], []
    t0 = time.perf_counter()
    index = slots.index_study(study_dir, lookup) if mode == "new" else None
    ms_index_once = (time.perf_counter() - t0) * 1000.0 if mode == "new" else 0.0
    n_files = 0
    for sh in SHIFTS:
        ts = time.perf_counter()
        x, m, info = slots.build_study_tensor(study_dir, lookup, P=P, G=G, T=T,
                                              anchor_shift=int(sh), index=index)
        per.append({"shift": int(sh), "ms": (time.perf_counter() - ts) * 1000.0,
                    "ms_index": float(info.get("ms_index", 0.0)),
                    "ms_decode": float(info.get("ms_decode", 0.0))})
        n_files = int(info.get("n_files", 0))
        xs.append(x)
    ms_total = (time.perf_counter() - t0) * 1000.0
    return {"mode": mode, "ms_total": ms_total, "ms_index_once": ms_index_once,
            "n_files": n_files, "per_shift": per,
            "hashes": [hashlib.md5(x.tobytes()).hexdigest() for x in xs]}


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def shift_bench(slots, set_decoder, label, image_dir, lookup, pairs, warm_uids):
    """old-vs-new x cold-vs-warm for the 3-shift TTA build.

    ``pairs``: [(uid_a, uid_b)] of NEVER-TOUCHED studies -- one takes "old" cold, the other
    "new" cold, alternating by pair index so neither condition inherits a list-order bias.
    Two different studies per pair is the whole point: running both conditions on one study
    would let the first warm the second.  The warm cells then run BOTH conditions on the SAME
    (now warm) study, which also gives a like-for-like tensor-identity check.
    """
    set_decoder("pydicom")            # index cost is header-only, so the decoder is irrelevant here
    print("\n=== 4. %s: %d-shift TTA build %s, old (index per shift) vs new (index once) ==="
          % (label, len(SHIFTS), list(SHIFTS)), flush=True)
    cells = {"cold_old": [], "cold_new": [], "warm_old": [], "warm_new": []}
    ident_ok = ident_n = 0
    for i, (ua, ub) in enumerate(pairs):
        u_old, u_new = (ua, ub) if i % 2 == 0 else (ub, ua)
        c_old = three_shift(slots, os.path.join(image_dir, u_old), lookup.get(u_old, {}), "old")
        c_new = three_shift(slots, os.path.join(image_dir, u_new), lookup.get(u_new, {}), "new")
        cells["cold_old"].append(c_old)
        cells["cold_new"].append(c_new)
        print("  pair %2d COLD old %s %7.0f ms (%d files, shifts %s) | new %s %7.0f ms "
              "(%d files, index-once %5.0f ms, shifts %s)"
              % (i, u_old[-8:], c_old["ms_total"], c_old["n_files"],
                 [round(p["ms"]) for p in c_old["per_shift"]], u_new[-8:], c_new["ms_total"],
                 c_new["n_files"], c_new["ms_index_once"], [round(p["ms"]) for p in c_new["per_shift"]]),
              flush=True)
    for u in warm_uids:
        # Explicit throwaway pass so the warm cells really are warm: a study that has never
        # been read would hand its cold first-touch cost to whichever condition ran first.
        three_shift(slots, os.path.join(image_dir, u), lookup.get(u, {}), "new")
        w_old = three_shift(slots, os.path.join(image_dir, u), lookup.get(u, {}), "old")
        w_new = three_shift(slots, os.path.join(image_dir, u), lookup.get(u, {}), "new")
        cells["warm_old"].append(w_old)
        cells["warm_new"].append(w_new)
        ident_n += len(SHIFTS)
        ident_ok += sum(1 for a, b in zip(w_old["hashes"], w_new["hashes"]) if a == b)
        print("  warm %s old %7.0f ms | new %7.0f ms | tensors identical %d/%d shifts"
              % (u[-8:], w_old["ms_total"], w_new["ms_total"],
                 sum(1 for a, b in zip(w_old["hashes"], w_new["hashes"]) if a == b), len(SHIFTS)),
              flush=True)
    out = {"label": label, "shifts": list(SHIFTS), "n_pairs": len(pairs),
           "tensors_identical": "%d/%d" % (ident_ok, ident_n)}
    for k, v in cells.items():
        out["ms_" + k] = _mean([c["ms_total"] for c in v])
        out["n_files_" + k] = _mean([c["n_files"] for c in v])
    out["saving_cold_ms"] = out["ms_cold_old"] - out["ms_cold_new"]
    out["saving_warm_ms"] = out["ms_warm_old"] - out["ms_warm_new"]
    for cond in ("cold", "warm"):
        for way in ("old", "new"):
            out["min_%d_%s_%s" % (HIDDEN_STUDIES, cond, way)] = out["ms_%s_%s" % (cond, way)] * HIDDEN_STUDIES / 60000.0
        out["min_%d_saving_%s" % (HIDDEN_STUDIES, cond)] = out["saving_%s_ms" % cond] * HIDDEN_STUDIES / 60000.0
    # mean files/study per cold cell: the two cold cells are DIFFERENT studies, so if these
    # two numbers diverge the saving is contaminated by study size, not by the fix.
    print("  %s COLD cell sizes: old %.0f files/study, new %.0f files/study"
          % (label, out["n_files_cold_old"], out["n_files_cold_new"]), flush=True)
    print("  %s COLD: old %.0f ms/study -> new %.0f ms/study, saving %.0f ms/study "
          "(%d studies: %.1f -> %.1f min, saves %.1f min)"
          % (label, out["ms_cold_old"], out["ms_cold_new"], out["saving_cold_ms"], HIDDEN_STUDIES,
             out["min_%d_cold_old" % HIDDEN_STUDIES], out["min_%d_cold_new" % HIDDEN_STUDIES],
             out["min_%d_saving_cold" % HIDDEN_STUDIES]), flush=True)
    print("  %s WARM: old %.0f ms/study -> new %.0f ms/study, saving %.0f ms/study "
          "(%d studies: %.1f -> %.1f min, saves %.1f min) | tensors identical %s"
          % (label, out["ms_warm_old"], out["ms_warm_new"], out["saving_warm_ms"], HIDDEN_STUDIES,
             out["min_%d_warm_old" % HIDDEN_STUDIES], out["min_%d_warm_new" % HIDDEN_STUDIES],
             out["min_%d_saving_warm" % HIDDEN_STUDIES], out["tensors_identical"]), flush=True)
    out["cells"] = cells
    return out


def main():
    print("input mounts:", os.listdir("/kaggle/input"), flush=True)
    code = walk_find(lambda b, d, f: b if ("infer_slotknee.py" in f and os.path.basename(b) == "scripts") else None)
    code = os.path.dirname(code) if code else None
    comp = walk_find(lambda b, d, f: b if ("sample_submission.csv" in f and "test_series" in d) else None)
    print("code:", code, "\ncompetition:", comp, flush=True)
    if not (code and comp):
        raise SystemExit("missing input: code/competition not both found")

    work = "/kaggle/working/code"
    if os.path.isdir(work):
        shutil.rmtree(work)
    shutil.copytree(code, work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    os.chdir(work)
    sys.path.insert(0, work)

    install_wheels()
    try:
        import dicomsdl
        print("dicomsdl", getattr(dicomsdl, "__version__", "?"), "importable", flush=True)
    except Exception as e:
        print("dicomsdl NOT importable:", e, flush=True)

    import numpy as np
    import pandas as pd
    import src.slots as slots
    print("shipped slots.set_decoder present:", hasattr(slots, "set_decoder"), flush=True)
    set_decoder, dstate = install_decoder_shim(slots)

    # ---- study lists -----------------------------------------------------
    test_dir = next((os.path.join(comp, n) for n in ("test_series", "test_images")
                     if os.path.isdir(os.path.join(comp, n))), None)
    train_dir = next((os.path.join(comp, n) for n in ("train_series", "train_images")
                      if os.path.isdir(os.path.join(comp, n))), None)
    test_uids = sorted(e.name for e in os.scandir(test_dir) if e.is_dir()) if test_dir else []
    train_uids = sorted(e.name for e in os.scandir(train_dir) if e.is_dir()) if train_dir else []
    print("test studies: %d (%s)\ntrain studies: %d" % (len(test_uids), test_dir, len(train_uids)), flush=True)

    def load_lookup(split):
        p = os.path.join(comp, "%s_series.csv" % split)
        if not os.path.isfile(p):
            return {}
        df = pd.read_csv(p, dtype=str)
        if "StudyInstanceUID" not in df.columns:
            return {}
        return {str(s): slots.series_lookup(g) for s, g in df.groupby("StudyInstanceUID", sort=False)}

    test_lookup, train_lookup = load_lookup("test"), load_lookup("train")
    report = {"P": P, "G": G, "T": T, "n_test_studies": len(test_uids),
              "n_train_studies": len(train_uids), "shifts": list(SHIFTS)}
    rng = random.Random(0)

    # ---- 4. anchor-shift index reuse (FIRST: it is the only cold-cache-sensitive test) ----
    # Everything below this point reads every study, so a cold cell measured later would be a
    # lie.  Test first (3 studies: 2 cold cells + 1 warm cell), then train pairs for a sample
    # large enough that per-study size variation between the two cold cells averages out.
    if len(test_uids) >= 3:
        report["shift_bench_test"] = shift_bench(
            slots, set_decoder, "test", test_dir, test_lookup,
            pairs=[(test_uids[0], test_uids[1])], warm_uids=[test_uids[2]])
    else:
        print("skipping the test shift bench: need >=3 visible test studies", flush=True)
    cold_uids = rng.sample(train_uids, min(2 * COLD_PAIRS, len(train_uids))) if train_uids else []
    pairs = [(cold_uids[2 * i], cold_uids[2 * i + 1]) for i in range(len(cold_uids) // 2)]
    if pairs:
        report["shift_bench_train"] = shift_bench(
            slots, set_decoder, "train-cold-pairs", train_dir, train_lookup,
            pairs=pairs, warm_uids=[pairs[0][0]])

    # ---- 1. transfer-syntax histograms -----------------------------------
    test_paths = []
    test_syntax, test_syntax_by_study = {}, {}
    for u in test_uids:
        fs = study_files(os.path.join(test_dir, u))
        test_paths += fs
        per = {}
        for f in fs:
            s = syntax_of(f)
            hist_add(test_syntax, s)
            hist_add(per, s)
        test_syntax_by_study[u] = per
    print("\n=== TEST transfer syntaxes (%d files over %d studies) ===" % (len(test_paths), len(test_uids)), flush=True)
    for k, v in sorted(test_syntax.items(), key=lambda kv: -kv[1]):
        print("  %-45s %6d  (%.1f%%)" % (k, v, 100.0 * v / max(1, len(test_paths))), flush=True)
    report["test_syntax_hist"] = test_syntax
    report["test_syntax_by_study"] = test_syntax_by_study
    report["n_test_files"] = len(test_paths)

    # Train is fully visible here: if ANY competition file is compressed, this finds it.
    scan = train_uids if len(train_uids) <= TRAIN_SYNTAX_SCAN else rng.sample(train_uids, TRAIN_SYNTAX_SCAN)
    train_syntax, n_train_scanned = {}, 0
    for u in sorted(scan):
        sd = os.path.join(train_dir, u)
        try:
            series = [e.path for e in os.scandir(sd) if e.is_dir()] or [sd]
        except OSError:
            continue
        for s_dir in series:                    # one file per series is enough for a syntax census
            fs = [os.path.join(s_dir, f) for f in sorted(os.listdir(s_dir)) if not f.startswith(".")]
            if fs:
                hist_add(train_syntax, syntax_of(fs[0]))
                n_train_scanned += 1
    print("\n=== TRAIN transfer syntaxes (%d series sampled over %d studies) ===" % (n_train_scanned, len(scan)), flush=True)
    for k, v in sorted(train_syntax.items(), key=lambda kv: -kv[1]):
        print("  %-45s %6d  (%.1f%%)" % (k, v, 100.0 * v / max(1, n_train_scanned)), flush=True)
    report["train_syntax_hist"] = train_syntax
    report["n_train_series_scanned"] = n_train_scanned

    # ---- 2. timing -------------------------------------------------------
    import hashlib

    def timed_pass(uids, image_dir, lookup, decoder, label, keep_hash=True):
        got = set_decoder(decoder)
        if got != decoder:
            print("  requested %s but got %s" % (decoder, got), flush=True)
        dstate["native"] = dstate["fallback"] = 0
        ms, ms_dec, ms_idx, hashes, n_fail, n_dec = [], [], [], {}, 0, 0
        for u in uids:
            t = time.perf_counter()
            x, m, info = slots.build_study_tensor(os.path.join(image_dir, u), lookup.get(u, {}),
                                                  P=P, G=G, T=T)
            ms.append((time.perf_counter() - t) * 1000.0)
            ms_dec.append(float(info.get("ms_decode", 0.0)))
            ms_idx.append(float(info.get("ms_index", 0.0)))
            n_fail += int(info.get("n_decode_fail", 0))
            n_dec += int(info.get("n_decoded", 0))
            if keep_hash:
                hashes[u] = hashlib.md5(x.tobytes()).hexdigest()
        out = {"decoder": got, "n": len(ms), "ms_per_study": sum(ms) / max(1, len(ms)),
               "ms_decode": sum(ms_dec) / max(1, len(ms_dec)), "ms_index": sum(ms_idx) / max(1, len(ms_idx)),
               "n_decoded": n_dec, "n_decode_fail": n_fail,
               "dicomsdl_native_slices": dstate["native"], "dicomsdl_fallback_slices": dstate["fallback"],
               "hashes": hashes}
        print("  [%s/%s] %8.1f ms/study (decode %7.1f, index %6.1f) | %d slices decoded, %d failed | "
              "dicomsdl native %d / fallback %d"
              % (label, got, out["ms_per_study"], out["ms_decode"], out["ms_index"],
                 n_dec, n_fail, dstate["native"], dstate["fallback"]), flush=True)
        return out

    def bench(uids, image_dir, lookup, label):
        """Warm pass (untimed, fills the page cache for BOTH measured passes), then A/B."""
        if not uids:
            return None
        print("\n=== timing %s: %d studies, P=%d G=%d T=%d ===" % (label, len(uids), P, G, T), flush=True)
        t = time.perf_counter()
        timed_pass(uids, image_dir, lookup, "pydicom", "warmup", keep_hash=False)
        print("  warmup pass took %.1f s" % (time.perf_counter() - t), flush=True)
        a = timed_pass(uids, image_dir, lookup, "pydicom", label)
        b = timed_pass(uids, image_dir, lookup, "dicomsdl", label)
        # reversed round: guards against page-cache / thermal drift favouring whoever ran second
        b2 = timed_pass(uids, image_dir, lookup, "dicomsdl", label + "-r2")
        a2 = timed_pass(uids, image_dir, lookup, "pydicom", label + "-r2")
        py = min(a["ms_per_study"], a2["ms_per_study"])
        sdl = min(b["ms_per_study"], b2["ms_per_study"])
        res = {"pydicom": a, "dicomsdl": b, "pydicom_r2": a2, "dicomsdl_r2": b2,
               "ms_pydicom": py, "ms_dicomsdl": sdl, "ratio": py / max(1e-9, sdl),
               "min_%d_pydicom" % HIDDEN_STUDIES: py * HIDDEN_STUDIES / 60000.0,
               "min_%d_dicomsdl" % HIDDEN_STUDIES: sdl * HIDDEN_STUDIES / 60000.0,
               "decode_only_ratio": a["ms_decode"] / max(1e-9, b["ms_decode"])}
        same = sum(1 for u in a["hashes"] if a["hashes"][u] == b["hashes"].get(u))
        res["tensor_identical"] = "%d/%d" % (same, len(a["hashes"]))
        print("  RESULT %s: pydicom %.0f ms/study, dicomsdl %.0f ms/study, ratio %.2fx "
              "(decode-only %.2fx) | %d studies -> %.1f min vs %.1f min | tensors identical %s"
              % (label, py, sdl, res["ratio"], res["decode_only_ratio"], HIDDEN_STUDIES,
                 res["min_%d_pydicom" % HIDDEN_STUDIES], res["min_%d_dicomsdl" % HIDDEN_STUDIES],
                 res["tensor_identical"]), flush=True)
        for r in (a, b, a2, b2):
            r.pop("hashes", None)
        return res

    report["test_bench"] = bench(test_uids, test_dir, test_lookup, "test")
    tr = train_uids if len(train_uids) <= TRAIN_SAMPLE else rng.sample(train_uids, TRAIN_SAMPLE)
    report["train_bench"] = bench(sorted(tr), train_dir, train_lookup, "train-sample")

    # ---- 3. slice-level parity ------------------------------------------
    print("\n=== parity: slots._decode_raw, pydicom vs dicomsdl ===", flush=True)
    sample = test_paths[:: max(1, len(test_paths) // max(1, PARITY_SLICES))][:PARITY_SLICES] if test_paths else []
    same = diff = failed = 0
    dstate["native"] = dstate["fallback"] = 0
    details = []
    for p in sample:
        set_decoder("pydicom")
        a1, sp1, sy1 = slots._decode_raw(p)
        set_decoder("dicomsdl")
        a2, sp2, sy2 = slots._decode_raw(p)
        if a1 is None or a2 is None:
            failed += 1
            details.append({"file": os.path.basename(p), "status": "decode_failed"})
        elif a1.shape == a2.shape and np.array_equal(a1, a2):
            same += 1
        else:
            diff += 1
            d = {"file": os.path.basename(p), "shape_py": list(a1.shape), "shape_sdl": list(a2.shape),
                 "syntax": sy1}
            if a1.shape == a2.shape:
                d["max_abs_diff"] = float(np.abs(a1.astype("f8") - a2.astype("f8")).max())
                d["n_differing_px"] = int((a1 != a2).sum())
            details.append(d)
    print("  %d slices: %d identical, %d differing, %d undecodable | dicomsdl native %d / fell back %d"
          % (len(sample), same, diff, failed, dstate["native"], dstate["fallback"]), flush=True)
    if details:
        print("  detail:", json.dumps(details)[:1500], flush=True)
    report["parity"] = {"n": len(sample), "identical": same, "differing": diff, "failed": failed,
                        "dicomsdl_native": dstate["native"], "dicomsdl_fallback": dstate["fallback"],
                        "details": details}

    report["elapsed_min"] = round((time.time() - T0) / 60, 2)
    with open("/kaggle/working/decode_bench_report.json", "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    slim = {k: ({kk: vv for kk, vv in v.items() if kk != "cells"} if k.startswith("shift_bench") else v)
            for k, v in report.items() if k != "test_syntax_by_study"}
    print("\n=== REPORT ===\n" + json.dumps(slim, indent=2, default=str), flush=True)
    print("done in %.1f min" % ((time.time() - T0) / 60), flush=True)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
