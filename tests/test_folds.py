"""Tests for src/folds.py (SlotKnee-S spec §5).

Run:  python3 -m pytest tests/test_folds.py -q        (< 60 s on CPU)

Fingerprint tests read <= 3 real studies from data_subset/train_images and skip when
they are absent; everything else is synthetic.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.folds as F                                         # noqa: E402
from src.llm_labels import LABELS                             # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data_subset")
IMG = os.path.join(DATA, "train_images")
HAVE_IMAGES = os.path.isdir(IMG) and len(os.listdir(IMG)) > 0


def _studies(n=3):
    return sorted(d for d in os.listdir(IMG) if not d.startswith("."))[:n]


# ---------------------------------------------------------------------------
# report_group
# ---------------------------------------------------------------------------

def test_report_group_normalises_case_and_whitespace():
    a = F.report_group("ACL  intact.\n No effusion.")
    b = F.report_group("acl intact. no effusion.")
    c = F.report_group("ACL intact. Small effusion.")
    assert a == b and a != c and len(a) == 40
    assert F.report_group("", row_id="u1") == "empty:u1"
    assert F.report_group(None, row_id="u2") == "empty:u2"
    assert F.report_group("   ") != F.report_group("   ")          # per-row unique without an id
    assert F.report_group(float("nan"), row_id=7) == "empty:7"


# ---------------------------------------------------------------------------
# scanner_fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_unknown_for_missing_or_empty_dir(tmp_path):
    assert F.scanner_fingerprint(str(tmp_path / "nope")) == F.UNKNOWN
    (tmp_path / "series").mkdir()
    assert F.scanner_fingerprint(str(tmp_path)) == F.UNKNOWN
    (tmp_path / "series" / "x.dcm").write_bytes(b"not a dicom")
    assert F.scanner_fingerprint(str(tmp_path)) == F.UNKNOWN


@pytest.mark.skipif(not HAVE_IMAGES, reason="data_subset/train_images not present")
def test_fingerprint_on_real_studies():
    for uid in _studies(3):
        fp = F.scanner_fingerprint(os.path.join(IMG, uid))
        assert fp != F.UNKNOWN and "|" in fp, fp
        assert fp == F.scanner_fingerprint(os.path.join(IMG, uid))          # memoised, stable
        raw = F.scanner_fingerprint(os.path.join(IMG, uid), freq_decimals=None)
        assert raw.split("|")[:2] == fp.split("|")[:2]                       # same vendor/model
        assert "\n" not in fp


# ---------------------------------------------------------------------------
# build_groups
# ---------------------------------------------------------------------------

def test_build_groups_same_report_same_group_else_report_hash(tmp_path):
    df = pd.DataFrame({
        "StudyInstanceUID": ["a", "b", "c", "d", "e"],
        "Report": ["Normal knee.", "normal  KNEE.", "Torn ACL.", "", ""],
    })
    g = F.build_groups(df, data_dir=str(tmp_path))           # no images -> no fingerprints
    assert g.name == "group" and list(g.index) == list(df.index)
    assert g["a" == df["StudyInstanceUID"]].iloc[0] == g.iloc[1]   # a and b share a report
    assert g.iloc[2] != g.iloc[0]
    assert g.iloc[3] != g.iloc[4]                                  # empty reports stay apart
    assert g.nunique() == 4


@pytest.mark.skipif(not HAVE_IMAGES, reason="data_subset/train_images not present")
def test_build_groups_uses_fingerprint_and_report_override():
    uids = _studies(3)
    df = pd.DataFrame({"StudyInstanceUID": uids + ["offdisk"],
                       "Report": ["r1", "r2", "r2", "r4"]})
    g = F.build_groups(df, DATA).tolist()
    assert g[0].startswith("s:") and "|" in g[0]                   # on-disk, unique report -> scanner
    assert g[1] == g[2] and g[1].startswith("r:")                  # shared report wins over scanner
    assert g[3].startswith("r:")                                   # off-disk -> own report hash
    # Report lookup from train.csv when df has none.
    if os.path.exists(os.path.join(DATA, "train.csv")):
        g2 = F.build_groups(pd.DataFrame({"StudyInstanceUID": uids}), DATA)
        assert len(g2) == 3 and all(x.startswith(("s:", "r:")) for x in g2)


# ---------------------------------------------------------------------------
# assign_grouped_folds
# ---------------------------------------------------------------------------

def _synthetic(n=120, seed=0):
    rng = np.random.RandomState(seed)
    uids = ["u%03d" % i for i in range(n)]
    reports = ["report %d" % i for i in range(n)]
    # three duplicate-report clusters
    for i in range(4):
        reports[i] = "template A"
    for i in range(10, 13):
        reports[i] = "template B"
    reports[20] = reports[21] = "template C"
    df = pd.DataFrame({"StudyInstanceUID": uids, "Report": reports})
    for lab in LABELS:
        df["y_" + lab] = rng.rand(n)
    df["fold"] = np.nan
    gold_idx = list(range(0, n, 10))                                # 12 gold rows, includes u000 (template A)
    df.loc[gold_idx, "fold"] = [i % 5 for i in range(len(gold_idx))]
    df["group"] = F.build_groups(df, data_dir="/nonexistent")
    # fake scanner groups to exercise multi-row groups; scannerX holds one gold row
    # (u040, fold 4), scannerY holds none -- no conflicting gold folds inside a group.
    df.loc[31:45, "group"] = "s:scannerX"
    df.loc[51:59, "group"] = "s:scannerY"
    return df


def test_folds_respect_groups_gold_and_balance():
    df = _synthetic()
    folds = F.assign_grouped_folds(df, n_folds=5, seed=42)
    assert folds.shape == (len(df),) and folds.dtype.kind == "i"
    assert set(folds) == set(range(5))                               # every fold non-empty
    # groups never straddle folds
    per_group = pd.Series(folds).groupby(df["group"].to_numpy()).nunique()
    assert (per_group == 1).all(), per_group[per_group > 1]
    # identical reports land in one fold
    for tpl in ("template A", "template B", "template C"):
        assert len(set(folds[df["Report"] == tpl])) == 1
    # gold rows keep their fold; template-A mates inherit u000's fold
    kept = df["fold"].notna().to_numpy()
    assert (folds[kept] == df.loc[kept, "fold"].to_numpy().astype(int)).all()
    assert (folds[df["Report"] == "template A"] == int(df.loc[0, "fold"])).all()
    assert (folds[df["group"] == "s:scannerX"] == int(df.loc[40, "fold"])).all()   # inherit gold mate's fold
    # roughly balanced
    counts = np.bincount(folds, minlength=5)
    assert counts.min() >= len(df) // 5 - 12 and counts.max() <= len(df) // 5 + 12, counts


def test_conflicting_gold_folds_inside_one_group_are_kept_as_is():
    """Gold folds are authoritative even when two gold rows share a group: the gold
    rows keep their own folds (the group then straddles, by construction) and the
    non-gold mates inherit the majority fold."""
    df = _synthetic()
    df.loc[31:45, "fold"] = np.nan
    df.loc[[31, 32, 33], "fold"] = [2, 2, 0]
    folds = F.assign_grouped_folds(df, n_folds=5, seed=42)
    assert folds[31] == 2 and folds[32] == 2 and folds[33] == 0
    assert (folds[34:46] == 2).all()


def test_folds_deterministic_and_seed_sensitive():
    df = _synthetic()
    a = F.assign_grouped_folds(df, seed=42)
    b = F.assign_grouped_folds(df, seed=42)
    c = F.assign_grouped_folds(df, seed=7)
    assert (a == b).all()
    assert (a != c).any()


def test_folds_without_keep_col_and_with_strat_col():
    df = _synthetic().drop(columns=["fold"])
    df["site"] = ["p", "q"] * (len(df) // 2)
    f1 = F.assign_grouped_folds(df, n_folds=4, seed=1)
    assert set(f1) == set(range(4))
    f2 = F.assign_grouped_folds(df, n_folds=4, seed=1, strat_col="site", keep_col=None)
    assert set(f2) == set(range(4))
    per_group = pd.Series(f2).groupby(df["group"].to_numpy()).nunique()
    assert (per_group == 1).all()


def test_positive_bucket_and_errors():
    df = pd.DataFrame({"y_a": [0, 1, 1, 1, 1], "y_b": [0, 0, 1, 1, 1],
                       "y_c": [0, 0, 0, 1, 1], "y_d": [0, 0, 0, 1, 1], "y_e": [0, 0, 0, 0, 1]})
    assert F.positive_bucket(df).tolist() == [0, 1, 1, 2, 3]
    with pytest.raises(ValueError):
        F.assign_grouped_folds(df.assign(StudyInstanceUID=list("abcde")))   # no group column
    small = df.assign(group=["g1", "g1", "g2", "g2", "g2"])
    with pytest.raises(ValueError):
        F.assign_grouped_folds(small, n_folds=5, keep_col=None)             # 2 groups < 5 folds
    bad = df.assign(group=list("abcde"), fold=[9, np.nan, np.nan, np.nan, np.nan])
    with pytest.raises(ValueError):
        F.assign_grouped_folds(bad, n_folds=5)
    assert F.assign_grouped_folds(df.iloc[:0].assign(group=[]), n_folds=5).shape == (0,)


@pytest.mark.skipif(not (HAVE_IMAGES and os.path.exists(os.path.join(DATA, "train_gold.csv"))),
                    reason="data_subset not present")
def test_gold_rows_keep_their_fold_with_real_csvs():
    gold = pd.read_csv(os.path.join(DATA, "train_gold.csv"))
    df = gold[["StudyInstanceUID", "Report", "fold"] + LABELS].copy()
    for lab in LABELS:
        df["y_" + lab] = df[lab]
    # add a handful of synthetic non-gold rows so there is something to split
    extra = pd.DataFrame({"StudyInstanceUID": ["x%d" % i for i in range(20)],
                          "Report": ["synthetic %d" % i for i in range(20)]})
    df = pd.concat([df, extra], ignore_index=True)
    df["group"] = F.build_groups(df, DATA)
    folds = F.assign_grouped_folds(df, n_folds=5, seed=42)
    g = df["fold"].notna().to_numpy()
    assert (folds[g] == df.loc[g, "fold"].astype(int).to_numpy()).all()
    assert set(folds) == set(range(5))


# ---------------------------------------------------------------------------
# FoldIntegrityError -- regression guard for the silent round-robin fallback
#
# train_slotknee.py used to wrap assign_grouped_folds in a bare `except:` and quietly
# substitute round-robin folds on ANY failure.  Ungrouped folds let scanner identity
# straddle the split and inflate CV by ~0.053, so a broken run scored BETTER than a
# correct one with nothing in the log to say so.  The script now re-raises integrity
# failures and only absorbs capacity failures, which relies on this type split.
# ---------------------------------------------------------------------------
def _healthy_df(n=200, n_groups=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "StudyInstanceUID": [f"s{i}" for i in range(n)],
        "group": [f"g{i % n_groups}" for i in range(n)],
        **{f"y_{k}": rng.integers(0, 2, n).astype(float) for k in range(12)},
    })


def test_integrity_error_is_a_valueerror():
    """Subclassing keeps every existing `except ValueError` caller working."""
    assert issubclass(F.FoldIntegrityError, ValueError)


def test_missing_group_column_is_an_integrity_error():
    with pytest.raises(F.FoldIntegrityError):
        F.assign_grouped_folds(_healthy_df().drop(columns=["group"]), n_folds=5, seed=42)


def test_corrupt_kept_fold_is_an_integrity_error():
    """A healthy dataset with one out-of-range kept fold: broken, not merely small."""
    df = _healthy_df()
    df["fold"] = np.nan
    df.loc[0, "fold"] = 99
    with pytest.raises(F.FoldIntegrityError):
        F.assign_grouped_folds(df, n_folds=5, seed=42)


def test_too_small_is_not_an_integrity_error():
    """Capacity limits must stay absorbable, or 6-study smoke runs cannot train."""
    tiny = _healthy_df(n=6, n_groups=6)
    try:
        F.assign_grouped_folds(tiny, n_folds=5, seed=42)
    except F.FoldIntegrityError:                     # pragma: no cover
        pytest.fail("a too-small dataset was misclassified as an integrity failure")
    except ValueError:
        pass                                          # plain ValueError is the capacity signal
