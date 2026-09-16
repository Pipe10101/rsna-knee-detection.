"""Tests for scripts/compare_arms.py on synthetic oof_fold_*.npz files (runs in a few seconds).

    python3 -m pytest tests/test_compare_arms.py -q
"""

import importlib.util
import json
import os
import shutil
import sys

import numpy as np
import pytest
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "compare_arms", os.path.join(ROOT, "scripts", "compare_arms.py"))
ca = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ca)

L = len(ca.LABELS)


# ---------------------------------------------------------------------------
# Synthetic OOF writers
# ---------------------------------------------------------------------------
def make_fold(rng, n, fold, signal=1.0, shift=0.0, uids=None, y=None, w=None):
    """y: soft targets in [0,1] (binarised at 0.5 by the tool); w: 0 = silent cell.
    logits = signal * (y>0.5) + noise + shift, so `signal` sets quality, `shift` the
    per-fold calibration offset that separates pooled from per-fold AUC."""
    uids = np.array([f"f{fold}_u{i}" for i in range(n)]) if uids is None else uids
    if y is None:
        y = np.clip(rng.random((n, L)).round(2), 0, 1)
    if w is None:
        w = (rng.random((n, L)) > 0.2).astype(np.float32) * 0.5
    logits = signal * (y > 0.5) + rng.normal(0, 1, (n, L)) + shift
    return dict(uids=uids, logits=logits.astype(np.float32), y=y.astype(np.float32),
                w=w.astype(np.float32), mask=np.ones((n, 6), np.uint8),
                best_thresholds=np.full(L, 0.5))


def write_run(d, folds, n=120, seed=0, signal=1.0, shifts=None, base=None, sub=None):
    """Write oof_fold_<k>.npz under d (optionally under a nested sub-dir). If `base` (a dict
    fold -> npz dict) is given, reuse its uids/y/w so the arm shares the baseline's split."""
    rng = np.random.default_rng(seed)
    out = {}
    target = os.path.join(d, sub) if sub else d
    os.makedirs(target, exist_ok=True)
    for f in folds:
        kw = {}
        if base is not None and f in base:
            kw = dict(uids=base[f]["uids"], y=base[f]["y"], w=base[f]["w"])
        z = make_fold(rng, n, f, signal=signal, shift=(shifts or {}).get(f, 0.0), **kw)
        np.savez_compressed(os.path.join(target, f"oof_fold_{f}.npz"), **z)
        out[f] = z
    return out


def manual_pooled(zs, folds):
    lg = np.concatenate([zs[f]["logits"] for f in folds]).astype(np.float64)
    y = np.concatenate([zs[f]["y"] for f in folds])
    w = np.concatenate([zs[f]["w"] for f in folds])
    aucs = []
    for i in range(L):
        m = w[:, i] > 0
        yt = (y[m, i] > 0.5).astype(int)
        if len(np.unique(yt)) > 1:
            aucs.append(roc_auc_score(yt, lg[m, i]))
    return float(np.mean(aucs))


# ---------------------------------------------------------------------------
# Scoring semantics
# ---------------------------------------------------------------------------
def test_pooled_is_not_mean_of_per_fold(tmp_path):
    # Each fold is perfectly separable on its own, but a large per-fold offset makes the
    # pooled (concatenated) AUC much lower: the tool must report the pooled number.
    zs = write_run(str(tmp_path), [0, 1], signal=6.0, shifts={0: 0.0, 1: 12.0}, seed=1)
    run = ca.load_run(str(tmp_path))
    pooled = ca.pooled_macro(run, [0, 1])
    per_fold = np.mean([ca.pooled_macro(run, [f]) for f in (0, 1)])
    assert per_fold > 0.99
    assert pooled < per_fold - 0.1
    assert pooled == pytest.approx(manual_pooled(zs, [0, 1]), abs=1e-9)


def test_weights_binarisation_and_one_class_skip(tmp_path):
    rng = np.random.default_rng(3)
    n = 60
    y = rng.random((n, L))
    w = np.ones((n, L), np.float32)
    # Label 0: the cells with w == 0 carry garbage logits that would ruin the AUC if counted.
    w[:30, 0] = 0.0
    # Label 1: one class only once binarised at 0.5 (soft values all < 0.5) -> skipped.
    y[:, 1] = rng.random(n) * 0.4
    # Label 2: soft targets straddling 0.5; the AUC must use the binarised target.
    z = make_fold(rng, n, 0, signal=4.0, y=y, w=w)
    z["logits"][:30, 0] = -50.0 + 100.0 * (y[:30, 0] < 0.5)   # anti-signal where w == 0
    z["logits"][:, 2] = (y[:, 2] > 0.5) * 10.0 + rng.normal(0, 0.01, n)
    np.savez_compressed(tmp_path / "oof_fold_0.npz", **z)
    run = ca.load_run(str(tmp_path))
    aucs = ca.per_label_auc(ca.pool(run, [0]))
    yt0 = (y[30:, 0] > 0.5).astype(int)
    assert aucs[0] == pytest.approx(roc_auc_score(yt0, z["logits"][30:, 0]))
    assert np.isnan(aucs[1])
    assert aucs[2] == pytest.approx(1.0)
    assert ca.macro(aucs) == pytest.approx(np.nanmean(aucs))


def test_verdict_thresholds():
    noise = 0.006
    assert ca.verdict(0.0121, noise) == "ADOPT"
    assert ca.verdict(0.012, noise) == "NULL"
    assert ca.verdict(0.0, noise) == "NULL"
    assert ca.verdict(-0.012, noise) == "NULL"
    assert ca.verdict(-0.0121, noise) == "REGRESS"
    assert ca.verdict(float("nan"), noise) == "n/a"


# ---------------------------------------------------------------------------
# Fold handling, discovery, combine, noise
# ---------------------------------------------------------------------------
def test_common_fold_restriction_and_recursive_discovery(tmp_path):
    base_dir, arm_dir = str(tmp_path / "base"), str(tmp_path / "kaggle_arm_v3")
    bz = write_run(base_dir, [0, 1, 2, 3, 4], seed=10)
    # Arm only has folds 0-1, nested as a fetched Kaggle output would be.
    az = write_run(arm_dir, [0, 1], seed=11, base=bz, sub=os.path.join("output", "models", "run"))
    base, arm = ca.load_run(base_dir), ca.load_run(arm_dir)
    assert sorted(arm.folds) == [0, 1]
    rec = ca.compare(base, arm, noise=0.006)
    assert rec["folds"] == [0, 1]
    assert rec["uid_overlap"] == pytest.approx(1.0)
    assert rec["n_studies"] == 240
    assert rec["base_auc"] == pytest.approx(manual_pooled(bz, [0, 1]), abs=1e-9)
    assert rec["auc"] == pytest.approx(manual_pooled(az, [0, 1]), abs=1e-9)
    assert rec["delta"] == pytest.approx(rec["auc"] - rec["base_auc"])
    # The baseline's 5-fold pooled AUC differs from its folds-0/1 pooled AUC (sanity).
    assert ca.pooled_macro(base, [0, 1, 2, 3, 4]) != pytest.approx(rec["base_auc"], abs=1e-6)
    # --folds 0 narrows further; a requested fold absent from the arm is dropped.
    rec0 = ca.compare(base, arm, noise=0.006, folds_req=[0, 3])
    assert rec0["folds"] == [0] and rec0["n_studies"] == 120
    # No common folds at all -> n/a verdict, no crash.
    rec_none = ca.compare(base, arm, noise=0.006, folds_req=[4])
    assert rec_none["verdict"] == "n/a" and rec_none["n_studies"] == 0


def test_common_uids_mode_for_different_splits(tmp_path):
    base_dir, arm_dir = str(tmp_path / "base"), str(tmp_path / "t4")
    bz = write_run(base_dir, [0, 1], seed=20)
    # Arm uses a different split: the same 240 studies reshuffled into 3 folds.
    rng = np.random.default_rng(21)
    all_uids = np.concatenate([bz[0]["uids"], bz[1]["uids"]])
    all_y = np.concatenate([bz[0]["y"], bz[1]["y"]])
    all_w = np.concatenate([bz[0]["w"], bz[1]["w"]])
    perm = rng.permutation(len(all_uids))
    os.makedirs(arm_dir)
    for f, chunk in enumerate(np.array_split(perm, 3)):
        z = make_fold(rng, len(chunk), f, uids=all_uids[chunk], y=all_y[chunk], w=all_w[chunk])
        np.savez_compressed(os.path.join(arm_dir, f"oof_fold_{f}.npz"), **z)
    base, arm = ca.load_run(base_dir), ca.load_run(arm_dir)
    rec = ca.compare(base, arm, noise=0.006)                  # fold-matched: overlap is partial
    assert rec["folds"] == [0, 1] and 0.0 < rec["uid_overlap"] < 0.999
    assert "common-uids" in rec.get("note", "")
    rec_u = ca.compare(base, arm, noise=0.006, common_uids=True)
    assert rec_u["folds"] == [0, 1, 2] and rec_u["uid_overlap"] == pytest.approx(1.0)
    assert rec_u["n_studies"] == rec_u["n_baseline"] == 240


def test_combine_row_is_rank_average(tmp_path):
    base_dir, a_dir, b_dir = (str(tmp_path / d) for d in ("base", "a", "b"))
    bz = write_run(base_dir, [0, 1], seed=30, signal=0.5)
    write_run(a_dir, [0, 1], seed=31, signal=1.0, base=bz)
    write_run(b_dir, [0, 1], seed=32, signal=1.0, base=bz)
    base, a, b = (ca.load_run(d) for d in (base_dir, a_dir, b_dir))
    comb = ca.combine_runs(a, b)
    assert comb.name == "combine(a+b)" and sorted(comb.folds) == [0, 1]
    for f in (0, 1):
        n = len(a.folds[f].uids)
        exp = 0.5 * (np.column_stack([rankdata(a.folds[f].logits[:, j]) / n for j in range(L)])
                     + np.column_stack([rankdata(b.folds[f].logits[:, j]) / n for j in range(L)]))
        assert np.allclose(comb.folds[f].logits, exp)
        assert np.array_equal(comb.folds[f].uids, a.folds[f].uids)
    # Two independent noisy arms of equal quality: their rank-average beats each one.
    rec_a, rec_b = ca.compare(base, a, 0.006), ca.compare(base, b, 0.006)
    rec_c = ca.compare(base, comb, 0.006)
    assert rec_c["auc"] > max(rec_a["auc"], rec_b["auc"])
    assert rec_c["verdict"] == "ADOPT"
    # Arms with reordered / partially overlapping uids are aligned by uid.
    fb = b.folds[0]
    b.folds[0] = ca.FoldData(fb.uids[::-1][:-5], fb.logits[::-1][:-5], fb.y[::-1][:-5], fb.w[::-1][:-5])
    comb2 = ca.combine_runs(a, b, folds=[0])
    assert len(comb2.folds[0].uids) == len(fb.uids) - 5
    pos = {u: i for i, u in enumerate(fb.uids)}
    ra = rankdata(a.folds[0].logits[5:, 0]) / (len(fb.uids) - 5)
    rb = rankdata(np.array([fb.logits[pos[u], 0] for u in comb2.folds[0].uids])) / (len(fb.uids) - 5)
    assert np.allclose(comb2.folds[0].logits[:, 0], 0.5 * (ra + rb))


def test_noise_from_two_seeds_same_split(tmp_path):
    s1, s2 = str(tmp_path / "baseline_s42"), str(tmp_path / "baseline_s1337")
    z1 = write_run(s1, [0, 1, 2], seed=40)
    write_run(s2, [0, 1], seed=41, base=z1)
    a, b = ca.load_run(s1), ca.load_run(s2)
    nz = ca.noise_from_seeds(a, b)
    assert nz["naive_folds"] == [0, 1] and nz["same_split"] is True
    assert nz["naive"] == pytest.approx(abs(ca.pooled_macro(a, [0, 1]) - ca.pooled_macro(b, [0, 1])))
    # Same split: the paired estimate is over the 240 shared studies (seed 2 has no fold 2).
    assert nz["paired_n"] == 240 and nz["paired"] is not None
    assert nz["value"] == max(nz["naive"], nz["paired"]) and nz["used"] in ("naive", "paired")
    assert "naive fold-matched" in ca._noise_src(nz) and "paired common-uids" in ca._noise_src(nz)


def test_noise_from_two_seeds_different_split(tmp_path):
    # baseline_s1337 on a reshuffled split (train_slotknee.py seeds the folds from --seed):
    # the fold-matched "naive" number is not paired; the common-uids estimate is.
    s1, s2 = str(tmp_path / "baseline_s42"), str(tmp_path / "baseline_s1337")
    z1 = write_run(s1, [0, 1], seed=70)
    rng = np.random.default_rng(71)
    all_uids = np.concatenate([z1[0]["uids"], z1[1]["uids"]])
    all_y = np.concatenate([z1[0]["y"], z1[1]["y"]])
    all_w = np.concatenate([z1[0]["w"], z1[1]["w"]])
    perm = rng.permutation(len(all_uids))
    os.makedirs(s2)
    for f, chunk in enumerate(np.array_split(perm, 2)):
        z = make_fold(rng, len(chunk), f, uids=all_uids[chunk], y=all_y[chunk], w=all_w[chunk])
        np.savez_compressed(os.path.join(s2, f"oof_fold_{f}.npz"), **z)
    a, b = ca.load_run(s1), ca.load_run(s2)
    nz = ca.noise_from_seeds(a, b)
    assert nz["same_split"] is False and nz["paired_n"] == 240
    assert nz["paired"] == pytest.approx(abs(ca.pooled_macro(a, [0, 1]) - ca.pooled_macro(b, [0, 1])))
    assert nz["value"] == max(nz["naive"], nz["paired"])
    assert "splits differ" in ca._noise_src(nz)
    # Per-fold overlap is exposed on the comparison record and in the table.
    rec = ca.compare(a, b, noise=0.006)
    assert set(rec["uid_overlap_per_fold"]) == {"0", "1"}
    assert all(0.0 < v < 1.0 for v in rec["uid_overlap_per_fold"].values())
    same = ca.compare(a, a, noise=0.006)
    assert same["uid_overlap_per_fold"] == {"0": 1.0, "1": 1.0}


# ---------------------------------------------------------------------------
# CLI end to end
# ---------------------------------------------------------------------------
def test_cli_markdown_and_json(tmp_path, capsys):
    base_dir = str(tmp_path / "baseline_s42")
    bz = write_run(base_dir, [0, 1], seed=50, signal=0.8)
    good = str(tmp_path / "good_s42")
    write_run(good, [0, 1], seed=51, signal=3.0, base=bz)          # clearly better -> ADOPT
    bad = str(tmp_path / "bad_s42")
    write_run(bad, [0, 1], seed=52, signal=0.0, base=bz)           # chance level -> REGRESS
    same = str(tmp_path / "same_s42")                               # byte-identical copy -> NULL
    os.makedirs(same)
    for f in (0, 1):
        shutil.copy(os.path.join(base_dir, f"oof_fold_{f}.npz"), same)
    out_json = str(tmp_path / "out.json")
    rc = ca.main([good, bad, same, base_dir, "--baseline", base_dir, "--noise", "0.01",
                  "--combine", good, same, "--json", out_json, "--folds", "0", "1"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "| arm | folds | n | uid-ovl | AUC | per-fold | delta | d focus | verdict |" in text
    assert text.count("| ADOPT |") >= 1 and "| REGRESS |" in text and "| NULL |" in text
    assert "combine(good_s42+same_s42)" in text
    with open(out_json) as fh:
        j = json.load(fh)
    assert j["noise"] == {"value": 0.01, "source": "given", "detail": None}
    assert j["baseline"]["folds"] == [0, 1] and j["baseline"]["n_studies"] == 240
    by = {r["arm"]: r for r in j["arms"]}
    assert set(by) == {"good_s42", "bad_s42", "same_s42"}    # baseline itself is skipped
    assert by["good_s42"]["verdict"] == "ADOPT" and by["bad_s42"]["verdict"] == "REGRESS"
    assert by["same_s42"]["verdict"] == "NULL" and by["same_s42"]["delta"] == pytest.approx(0.0)
    assert len(by["good_s42"]["per_label_delta"]) == L
    assert j["combined"]["arm"] == "combine(good_s42+same_s42)"
    assert j["combined"]["folds"] == [0, 1]
    assert np.isfinite(j["combined"]["auc"])
    assert by["good_s42"]["win"] is True and by["bad_s42"]["win"] is False
    assert set(by["good_s42"]["per_fold_auc"]) == {"0", "1"} and set(j["baseline"]["per_fold_auc"]) == {"0", "1"}
    assert "| per-fold | delta | d focus |" in text and "0,1" in text
    assert j["focus_labels"] == ["ACL", "MCL", "Lateral Meniscus"]
    g = by["good_s42"]
    assert g["focus_delta"] == pytest.approx(np.mean([g["per_label_delta"][l] for l in j["focus_labels"]]))
    assert j["baseline"]["focus_auc"] == pytest.approx(
        np.mean([j["baseline"]["per_label_auc"][l] for l in j["focus_labels"]]))
    # Custom focus set; unknown labels are rejected.
    rc = ca.main([good, "--baseline", base_dir, "--focus", "Fracture", "--json", out_json])
    assert rc == 0
    with open(out_json) as fh:
        j2 = json.load(fh)
    assert j2["focus_labels"] == ["Fracture"]
    assert j2["arms"][0]["focus_delta"] == pytest.approx(j2["arms"][0]["per_label_delta"]["Fracture"])
    with pytest.raises(SystemExit):
        ca.main([good, "--baseline", base_dir, "--focus", "Nope"])


def test_cli_arms_flag_call_shape(tmp_path, capsys):
    # --arms option + --noise <dir of a second baseline seed>: noise = |pooled(base) - pooled(seed2)|.
    base_dir = str(tmp_path / "baseline_s42")
    bz = write_run(base_dir, [0, 1], seed=60, signal=0.8)
    seed2 = str(tmp_path / "baseline_s1337")
    write_run(seed2, [0, 1], seed=61, signal=0.8, base=bz)
    arm = str(tmp_path / "tb6_s42")
    write_run(arm, [0, 1, 2], seed=62, signal=0.8, base=bz)       # extra fold 2 must be ignored
    out_json = str(tmp_path / "out.json")
    rc = ca.main(["--baseline", base_dir, "--arms", arm, seed2, base_dir, "--folds", "0", "1",
                  "--noise", seed2, "--json", out_json])
    assert rc == 0
    text = capsys.readouterr().out
    a, b = ca.load_run(base_dir), ca.load_run(seed2)
    expected_noise = abs(ca.pooled_macro(a, [0, 1]) - ca.pooled_macro(b, [0, 1]))
    with open(out_json) as fh:
        j = json.load(fh)
    assert j["noise"]["detail"]["naive"] == pytest.approx(expected_noise)
    assert j["noise"]["value"] == pytest.approx(max(j["noise"]["detail"]["naive"], j["noise"]["detail"]["paired"]))
    assert "seed-to-seed" in j["noise"]["source"] and j["noise"]["detail"]["used"] in ("naive", "paired")
    by = {r["arm"]: r for r in j["arms"]}
    assert set(by) == {"tb6_s42", "baseline_s1337"}
    assert by["tb6_s42"]["folds"] == [0, 1] and by["tb6_s42"]["n_studies"] == 240
    assert by["baseline_s1337"]["verdict"] == "NULL"               # |delta| == naive <= noise
    assert by["baseline_s1337"]["uid_overlap_per_fold"] == {"0": 1.0, "1": 1.0}
    assert f"noise = {j['noise']['value']:.4f}" in text and "1.0000/1.0000" in text
    # A bad --noise value is rejected.
    with pytest.raises(SystemExit):
        ca.main(["--baseline", base_dir, "--arms", arm, "--noise", "not-a-number-or-dir"])
