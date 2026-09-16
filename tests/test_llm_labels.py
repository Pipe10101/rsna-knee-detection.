"""Tests for src/llm_labels.py (SlotKnee-S spec §5).

Run:  python3 -m pytest tests/test_llm_labels.py -q        (< 60 s on CPU)

Synthetic CSVs throughout; the regex-fallback tests use data_subset/ and skip when
it is absent.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.llm_labels as LL                                   # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data_subset")
GOLD = os.path.join(DATA, "train_gold.csv")
TRAIN = os.path.join(DATA, "train.csv")

UIDS = ["1.2.3.%d" % i for i in range(1, 7)]


def _variant_a(tmp_path):
    """Lower-case / underscore / 'bakers_cyst' / 'study_id' spellings."""
    cols = ["study_id", "acl", "mcl", "Medial_Meniscus", "lateral meniscus",
            "medial_oa", "LATERAL OA", "pf_oa", "effusion", "synovitis",
            "bakers_cyst", "contusion", "fracture"]
    rows = []
    for i, uid in enumerate(UIDS[:4]):
        vals = [0.9, 0.1, 0.5, 0.5, 0.2, 0.5, 0.5, 0.8, 0.5, 0.5, 0.05, 0.95]
        vals = [(v + 0.01 * i) if v != 0.5 else 0.5 for v in vals]
        rows.append([uid] + vals)
    df = pd.DataFrame(rows, columns=cols)
    p = tmp_path / "a.csv"
    df.to_csv(p, index=False)
    return str(p), df


def _variant_b(tmp_path):
    """'p_' prefix, exact label spelling, one label missing, out-of-range values."""
    cols = ["StudyInstanceUID"] + ["p_" + lab for lab in LL.LABELS if lab != "Fracture"]
    rows = []
    for uid in UIDS[2:6]:
        rows.append([uid] + [0.7, 0.3, 0.5, 1.3, -0.2, 0.5, 0.5, 0.6, 0.5, 0.5, np.nan])
    df = pd.DataFrame(rows, columns=cols)
    p = tmp_path / "b.csv"
    df.to_csv(p, index=False)
    return str(p), df


# ---------------------------------------------------------------------------
# column matching
# ---------------------------------------------------------------------------

def test_match_label_columns_is_case_space_apostrophe_insensitive():
    cols = ["study_id", "acl", "Medial_Meniscus", "bakers", "Baker's", "PF_OA",
            "lateral-meniscus", "p_Effusion", "Synovitis_prob", "unrelated"]
    m = LL.match_label_columns(cols)
    assert m["ACL"] == "acl"
    assert m["Medial Meniscus"] == "Medial_Meniscus"
    assert m["Baker's"] in ("bakers", "Baker's")           # exact wins when both exist
    assert m["PF OA"] == "PF_OA"
    assert m["Lateral Meniscus"] == "lateral-meniscus"
    assert m["Effusion"] == "p_Effusion"
    assert m["Synovitis"] == "Synovitis_prob"
    assert "Fracture" not in m
    assert LL.match_id_column(cols) == "study_id"
    assert LL.match_id_column(["StudyInstanceUID", "uid"]) == "StudyInstanceUID"


def test_load_variant_a_schema_and_values(tmp_path):
    path, raw = _variant_a(tmp_path)
    df = LL.load_llm_labels([path])
    assert df.columns.tolist() == [LL.ID_COL] + LL.LABELS
    assert len(df) == 4
    assert df[LL.LABELS].min().min() >= 0.0 and df[LL.LABELS].max().max() <= 1.0
    row = df.set_index(LL.ID_COL).loc[UIDS[0]]
    assert row["ACL"] == pytest.approx(0.9)
    assert row["Baker's"] == pytest.approx(0.5)                      # bakers_cyst column matched
    assert row["Contusion"] == pytest.approx(0.05)
    assert row["Fracture"] == pytest.approx(0.95)
    assert row["Medial Meniscus"] == pytest.approx(0.5)


def test_load_variant_b_clips_and_fills_missing(tmp_path):
    path, _ = _variant_b(tmp_path)
    df = LL.load_llm_labels([path]).set_index(LL.ID_COL)
    assert df["Lateral Meniscus"].iloc[0] == pytest.approx(1.0)      # 1.3 clipped
    assert df["Medial OA"].iloc[0] == pytest.approx(0.0)             # -0.2 clipped
    assert df["Fracture"].iloc[0] == pytest.approx(0.5)              # column absent -> silent
    assert df["Contusion"].iloc[0] == pytest.approx(0.5)             # NaN -> silent


def test_blend_two_files_mean_and_union(tmp_path):
    pa, _ = _variant_a(tmp_path)
    pb, _ = _variant_b(tmp_path)
    df = LL.load_llm_labels([pa, pb], blend="mean").set_index(LL.ID_COL)
    assert sorted(df.index) == sorted(UIDS)                          # union of studies
    # UIDS[2] is in both: ACL = mean(0.9 + 0.02, 0.7)
    assert df.loc[UIDS[2], "ACL"] == pytest.approx((0.92 + 0.7) / 2)
    # Fracture is missing from file b -> file a's value, NOT averaged with 0.5.
    assert df.loc[UIDS[2], "Fracture"] == pytest.approx(0.97)
    # UIDS[0] only in a, UIDS[5] only in b.
    assert df.loc[UIDS[0], "ACL"] == pytest.approx(0.9)
    assert df.loc[UIDS[5], "ACL"] == pytest.approx(0.7)
    assert df.loc[UIDS[5], "Fracture"] == pytest.approx(0.5)
    first = LL.load_llm_labels([pa, pb], blend="first").set_index(LL.ID_COL)
    assert first.loc[UIDS[2], "ACL"] == pytest.approx(0.92)
    mx = LL.load_llm_labels([pa, pb], blend="max").set_index(LL.ID_COL)
    assert mx.loc[UIDS[2], "ACL"] == pytest.approx(0.92)
    with pytest.raises(ValueError):
        LL.load_llm_labels([pa], blend="geometric")


def test_minus_one_sentinel_and_percent_scale(tmp_path):
    df = pd.DataFrame({"StudyInstanceUID": ["a", "b"],
                       **{lab: [-1, 1] for lab in LL.LABELS}})
    out = LL.load_llm_labels([df]).set_index(LL.ID_COL)
    assert out.loc["a", "ACL"] == pytest.approx(0.5)                # sentinel -> silent
    pct = pd.DataFrame({"StudyInstanceUID": ["a", "b"],
                        **{lab: [90, 10] for lab in LL.LABELS}})
    out = LL.load_llm_labels([pct]).set_index(LL.ID_COL)
    assert out.loc["a", "ACL"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# weights and fills
# ---------------------------------------------------------------------------

def test_cell_weights():
    w = LL.cell_weights(np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
    assert np.allclose(w, [1.0, 0.5, 0.0, 0.5, 1.0])
    assert LL.cell_weights(0.5) == 0.0


def test_fill_silent_synovitis():
    df = pd.DataFrame({"Synovitis": [0.5, 0.5, 0.2, 0.5],
                       "Effusion": [0.9, 0.5, 0.9, 0.1]})
    out = LL.fill_silent_synovitis(df)
    assert out["Synovitis"].tolist() == pytest.approx([0.7, 0.5, 0.2, 0.3])
    assert df["Synovitis"].iloc[0] == 0.5                            # input untouched


# ---------------------------------------------------------------------------
# build_targets
# ---------------------------------------------------------------------------

def _gold():
    g = pd.DataFrame({"StudyInstanceUID": [UIDS[1], "gold-only"],
                      **{lab: [1.0, 0.0] for lab in LL.LABELS},
                      "fold": [3, 1]})
    g.loc[0, "ACL"] = 0.0
    return g


def test_build_targets_schema_weights_gold_override(tmp_path):
    pa, _ = _variant_a(tmp_path)
    llm = LL.load_llm_labels([pa])
    out = LL.build_targets(llm, _gold(), gold_weight=8.0)
    expect = [LL.ID_COL] + LL.Y_COLS + LL.W_COLS + ["is_gold", "fold"]
    assert out.columns.tolist() == expect
    assert len(out) == 5 and out[LL.ID_COL].is_unique
    o = out.set_index(LL.ID_COL)
    # non-gold: y = p, w = 2|p-0.5|, silent cells weigh 0
    assert o.loc[UIDS[0], "y_ACL"] == pytest.approx(0.9)
    assert o.loc[UIDS[0], "w_ACL"] == pytest.approx(0.8)
    assert o.loc[UIDS[0], "w_Medial Meniscus"] == 0.0
    assert not o.loc[UIDS[0], "is_gold"] and np.isnan(o.loc[UIDS[0], "fold"])
    # gold override: hard 0/1, weight 8 on every cell, fold kept
    assert o.loc[UIDS[1], "is_gold"]
    assert o.loc[UIDS[1], "y_ACL"] == 0.0 and o.loc[UIDS[1], "y_MCL"] == 1.0
    assert (o.loc[UIDS[1], LL.W_COLS] == 8.0).all()
    assert o.loc[UIDS[1], "fold"] == 3
    # gold study absent from the LLM file is appended
    assert o.loc["gold-only", "is_gold"] and (o.loc["gold-only", LL.Y_COLS] == 0.0).all()
    assert o.loc["gold-only", "fold"] == 1
    assert out["is_gold"].sum() == 2


def test_build_targets_min_weight_gate_and_no_gold(tmp_path):
    pa, _ = _variant_a(tmp_path)
    llm = LL.load_llm_labels([pa])
    out = LL.build_targets(llm, None, min_weight=0.7)
    assert "fold" not in out.columns and not out["is_gold"].any()
    o = out.set_index(LL.ID_COL)
    assert o.loc[UIDS[0], "w_ACL"] == pytest.approx(0.8)            # 0.8 >= 0.7 kept
    assert o.loc[UIDS[0], "w_MCL"] == pytest.approx(0.8)
    assert o.loc[UIDS[0], "w_Medial OA"] == 0.0                      # 2|0.2-0.5| = 0.6 < 0.7 gated
    assert o.loc[UIDS[0], "w_Effusion"] == 0.0                       # 0.6 gated
    assert o.loc[UIDS[0], "y_Medial OA"] == pytest.approx(0.2)       # target untouched by the gate
    ungated = LL.build_targets(llm, None).set_index(LL.ID_COL)
    assert ungated.loc[UIDS[0], "w_Medial OA"] == pytest.approx(0.6)


def test_build_targets_accepts_yw_schema():
    df = pd.DataFrame({"StudyInstanceUID": ["x", "y"]})
    for lab in LL.LABELS:
        df["y_" + lab] = [0.3, 0.8]
        df["w_" + lab] = [0.1, 0.9]
    out = LL.build_targets(df, _gold()).set_index(LL.ID_COL)
    assert out.loc["x", "y_ACL"] == pytest.approx(0.3) and out.loc["x", "w_ACL"] == pytest.approx(0.1)
    assert len(out) == 4 and out["is_gold"].sum() == 2


def test_build_targets_gold_nan_cell_has_zero_weight():
    g = _gold()
    g.loc[0, "MCL"] = np.nan
    out = LL.build_targets(None, g).set_index(LL.ID_COL)
    assert out.loc[UIDS[1], "w_MCL"] == 0.0 and out.loc[UIDS[1], "w_ACL"] == 8.0


# ---------------------------------------------------------------------------
# regex fallback (calls src.labels)
# ---------------------------------------------------------------------------

def test_regex_fallback_synthetic_without_images():
    df = pd.DataFrame({
        "StudyInstanceUID": ["s1", "s2", "s3"],
        "Report": ["Complete tear of the anterior cruciate ligament. Large joint effusion.",
                   "ACL intact. No effusion. Menisci normal.",
                   ""],
    })
    out = LL.regex_fallback(df, data_dir="/nonexistent", images_only=True)
    assert out.columns.tolist()[:1 + 24] == [LL.ID_COL] + LL.Y_COLS + LL.W_COLS
    assert len(out) == 2                                               # empty report dropped
    o = out.set_index(LL.ID_COL)
    assert o.loc["s1", "y_ACL"] > 0.7 and o.loc["s1", "w_ACL"] > 0.5
    assert o.loc["s2", "y_ACL"] < 0.2 and o.loc["s2", "w_ACL"] > 0.5
    assert o.loc["s1", "w_Fracture"] == 0.0                            # unmentioned -> 0 weight
    assert 0.0 < o.loc["s1", "y_Fracture"] < 0.5                       # ... but target is the prior
    assert not o["is_gold"].any()


@pytest.mark.skipif(not (os.path.exists(TRAIN) and os.path.isdir(os.path.join(DATA, "train_images"))),
                    reason="data_subset not present")
def test_regex_fallback_on_data_subset_matches_train_weights():
    t0 = time.time()
    out = LL.regex_fallback(TRAIN, DATA, images_only=True)
    elapsed = time.time() - t0
    assert elapsed < 60, elapsed
    n_disk = len([d for d in os.listdir(os.path.join(DATA, "train_images")) if not d.startswith(".")])
    assert len(out) == n_disk, (len(out), n_disk)
    assert out[LL.ID_COL].is_unique
    y = out[LL.Y_COLS].to_numpy(); w = out[LL.W_COLS].to_numpy()
    assert np.isfinite(y).all() and y.min() >= 0 and y.max() <= 1
    assert np.isfinite(w).all() and w.min() >= 0 and w.max() <= 1
    assert (w == 0).mean() > 0.05 and (w > 0.5).mean() > 0.2          # both verdict kinds present

    # Equivalence with train.derived_cell_weights (labels_weight=1, no gate).
    try:
        import src.labels as L
        import src.train as T
    except Exception as e:                                            # pragma: no cover
        pytest.skip(f"src.train not importable: {e}")
    import types
    frame = pd.read_csv(TRAIN, engine="python")
    sub = frame[frame[LL.ID_COL].astype(str).isin(out[LL.ID_COL].head(20))].head(20)
    derived = L.label_dataframe(sub, labels=LL.LABELS)
    cfg = types.SimpleNamespace(labels_weight=1.0, labels_min_conf=0.0,
                                soft_targets=True, labels_conf_suffix="__conf")
    vals, wts = T.derived_cell_weights(derived, LL.LABELS, cfg)
    mine = out.set_index(LL.ID_COL).loc[sub[LL.ID_COL].astype(str)]
    assert np.allclose(mine[LL.Y_COLS].to_numpy(), vals, atol=1e-6)
    assert np.allclose(mine[LL.W_COLS].to_numpy(), wts, atol=1e-6)

    # And the gold override composes on top of it.
    if os.path.exists(GOLD):
        gold = pd.read_csv(GOLD)
        tgt = LL.build_targets(out, gold, gold_weight=8.0)
        assert tgt["is_gold"].sum() == len(gold)
        assert tgt["fold"].notna().sum() == len(gold)
        assert (tgt.loc[tgt["is_gold"], LL.W_COLS] == 8.0).all().all()
        assert len(tgt) == len(set(out[LL.ID_COL]) | set(gold[LL.ID_COL].astype(str)))


# ---------------------------------------------------------------------------
# the real public label files (skip when absent)
# ---------------------------------------------------------------------------

EXT = os.path.join(DATA, "labels_external")
V4 = os.path.join(EXT, "stevenleehans", "llm_labels_v4_blend.csv")
V2 = os.path.join(EXT, "stevenleehans", "llm_labels_v2.csv")
FULL = os.path.join(EXT, "stevenleehans", "llm_labels_full.csv")
PK = os.path.join(EXT, "pilkwang", "report_labels_v2.csv")


def _macro_auc(probs, gold):
    from sklearn.metrics import roc_auc_score
    m = gold.merge(probs, on=LL.ID_COL, suffixes=("_gold", ""))
    aucs = {}
    for lab in LL.LABELS:
        y = m[lab + "_gold"].astype(int)
        aucs[lab] = roc_auc_score(y, m[lab]) if y.nunique() == 2 else np.nan
    return float(np.nanmean(list(aucs.values()))), aucs


@pytest.mark.skipif(not (os.path.exists(V4) and os.path.exists(GOLD)), reason="v4_blend / gold not present")
def test_v4_blend_loads_scores_and_builds_targets():
    v4 = LL.load_llm_labels([V4])
    assert len(v4) == 4407 and v4[LL.ID_COL].is_unique
    assert v4.columns.tolist() == [LL.ID_COL] + LL.LABELS
    assert v4[LL.LABELS].min().min() >= 0.0 and v4[LL.LABELS].max().max() <= 1.0
    gold = pd.read_csv(GOLD)
    macro, per = _macro_auc(v4, gold)
    print(f"\nv4_blend macro AUC vs 58 gold = {macro:.4f}")
    assert 0.85 < macro < 0.95, (macro, per)

    tgt = LL.build_targets(v4, gold, gold_weight=8.0)
    assert len(tgt) == 4407                                   # gold is a subset of v4
    assert tgt["is_gold"].sum() == 58
    assert (tgt.loc[tgt["is_gold"], LL.W_COLS] == 8.0).all().all()
    assert set(tgt.loc[tgt["is_gold"], LL.Y_COLS].to_numpy().ravel()) <= {0.0, 1.0}
    assert tgt["fold"].notna().sum() == 58
    w = tgt.loc[~tgt["is_gold"], LL.W_COLS].to_numpy()
    frac0 = float((w == 0).mean())
    print(f"v4_blend: fraction of non-gold cells with weight 0 = {frac0:.4f}; "
          f"fraction at 0.25 (silent+absent) = {np.isclose(tgt.loc[~tgt['is_gold'], LL.Y_COLS].to_numpy(), 0.25).mean():.4f}")
    # v4_blend averages a 0.5-silent reader with a hard 0/1 reader, so silence sits at
    # 0.25/0.75 and almost no cell is exactly 0.5.  The 25 % figure belongs to full.csv.
    assert frac0 < 0.01
    if os.path.exists(FULL):
        full = LL.load_llm_labels([FULL])
        frac0_full = float((LL.cell_weights(full[LL.LABELS].to_numpy()) == 0).mean())
        print(f"llm_labels_full: fraction of cells with weight 0 = {frac0_full:.4f}")
        assert 0.20 < frac0_full < 0.30
        if os.path.exists(V2):
            v2 = LL.load_llm_labels([V2]).set_index(LL.ID_COL)
            filled = LL.fill_silent_synovitis(full).set_index(LL.ID_COL).loc[v2.index]
            assert np.allclose(filled["Synovitis"], v2["Synovitis"], atol=1e-3)   # v2 == fill(full)


@pytest.mark.skipif(not (os.path.exists(V4) and os.path.exists(PK)), reason="v4_blend / pilkwang not present")
def test_blend_v4_with_pilkwang_and_conf_columns():
    pk = LL.load_llm_labels([PK], with_conf=True)
    assert pk.columns.tolist() == [LL.ID_COL] + LL.LABELS + LL.CONF_COLS
    assert len(pk) == 4406
    unk = np.isclose(pk["ACL"], 0.28)
    assert unk.any() and np.allclose(pk.loc[unk, "ACL__conf"], 0.05)
    bl = LL.load_llm_labels([V4, PK])
    assert len(bl) == 4407 and bl.columns.tolist() == [LL.ID_COL] + LL.LABELS
    assert bl[LL.LABELS].min().min() >= 0.0 and bl[LL.LABELS].max().max() <= 1.0
    v4 = LL.load_llm_labels([V4]).set_index(LL.ID_COL)
    uid = pk[LL.ID_COL].iloc[0]
    assert bl.set_index(LL.ID_COL).loc[uid, "ACL"] == pytest.approx(
        (v4.loc[uid, "ACL"] + pk.set_index(LL.ID_COL).loc[uid, "ACL"]) / 2)
    # explicit confidences become the weights
    tgt = LL.build_targets(pk, None)
    assert np.allclose(tgt[LL.W_COLS].to_numpy(), pk[LL.CONF_COLS].to_numpy())
    blc = LL.load_llm_labels([V4, PK], with_conf=True).set_index(LL.ID_COL)
    assert blc.loc[uid, "ACL__conf"] == pytest.approx(
        (LL.cell_weights(v4.loc[uid, "ACL"]) + pk.set_index(LL.ID_COL).loc[uid, "ACL__conf"]) / 2)
    if os.path.exists(GOLD):
        gold = pd.read_csv(GOLD)
        macro, _ = _macro_auc(bl, gold)
        print(f"\nv4_blend + pilkwang (mean) macro AUC vs gold = {macro:.4f}")
        assert macro > 0.85
