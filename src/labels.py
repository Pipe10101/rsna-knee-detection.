"""Weak supervision from free-text radiology reports.

WHY THIS MODULE EXISTS
----------------------
``data_subset/train.csv`` has 4407 rows.  All 4407 carry a non-empty ``Report``.
649 of them have DICOM pixels on disk under ``data_subset/train_images/``.
Only 58 of those 649 carry expert labels (``data_subset/train_gold.csv``).

So today the pipeline trains on 58 studies and throws away 591 studies that have
both real MRI pixels and a radiologist's written findings.  ``src/train.py``
explicitly lists ``"Report"`` among the columns it drops.

This module turns that free text into *soft* per-study targets so those 591
studies can contribute.  It is deliberately NOT a model: no torch, no sklearn,
no network, no extra pip dependency -- just ``re`` + ``unicodedata``.  It runs in
milliseconds over the whole corpus and costs nothing at inference time.

WHY RULES AND NOT A TEXT MODEL
------------------------------
The reports are formulaic ("Medial meniscus: No tear."), so precision from
regex is high; and a text model trained on 58 gold studies would be exactly as
weak as the image teacher in ``src/pseudo_label.py``, whose own docstring
concedes it is "trained on 58 studies and is *weaker* than the task demands".
A report is INDEPENDENT evidence: a radiologist looked at the knee and wrote
down what they saw.

FIVE VERDICTS, NOT TWO
----------------------
For every (study, label) the engine returns one of these, together with the
probability that the verdict is RIGHT -- measured on the 58 gold studies, 696
label cells:

  verdict          n     P(y=1)   soft target
  positive       208      0.784   0.65 - 0.92, per label
  weak_positive   84      0.512   ~0.51
  unmentioned    222      0.126   the per-label PRIOR
  weak_negative   32      0.062   ~0.06
  negative       150      0.027   ~0.03

The "unmentioned" row is the one that matters.  A finding the report never
mentions is still present 12.6% of the time (95% CI [0.082, 0.170]).  Writing 0
there fabricates confident negatives and is worse than not using the report at
all -- so it gets the calibrated prior instead, at confidence 0 so it costs the
loss nothing.

MULTILINGUAL
------------
The corpus is NOT three languages.  Measured over all 4407 reports with
``detect_language`` below:

    en 1736 | es 682 | tr 546 | hr 406 | el 321 | de 261 | bg 220 | nl 153 | fr 82

(hr = Croatian/Bosnian/Serbian latin, el = Greek, bg = Bulgarian cyrillic.)
Medical vocabulary is mostly Greco-Latin and therefore cognate across all nine,
so the lexicons below are multilingual unions applied after an aggressive
diacritic fold (``normalize_text``).  Negation and hedging, which are NOT
cognate, are handled per language -- note in particular that Turkish negates
*after* the finding ("... yoktur", "... izlenmedi", "... normaldir"), so a
prefix-only negation scope would silently invert 546 reports.

HOW GOOD IS IT
--------------
Against the 58 gold studies, macro ROC-AUC of the soft score vs the expert
label:

    in-sample                 0.869
    leave-one-fold-out        0.815   (calibration refit per fold)

Per label it ranges from 0.97 (ACL) to 0.69 (Effusion).  See
``evaluate_against_gold`` and ``python3 src/labels.py`` for the full table, and
the module's final report for which labels are NOT reliable.

Public API
----------
    normalize_text(text)            -> folded lowercase text
    detect_language(text)           -> 'en'|'es'|'tr'|'hr'|'de'|'nl'|'fr'|'el'|'bg'|'unknown'
    extract_findings(text)          -> {label: Finding}
    report_soft_targets(text)       -> ({label: prob}, {label: confidence})
    label_dataframe(df, ...)        -> DataFrame of soft targets / confidences
    write_label_csv(df, out_path)   -> CSV in the schema train.py's
                                       load_derived_labels() reads
    evaluate_against_gold(gold_df)  -> per-label agreement table (needs pandas)
    calibrate_priors / calibrate_states(gold_df, folds=...)
                                    -> refit the constants, optionally on
                                       training folds only

CLI
---
    python3 src/labels.py                       # language mix + agreement table
    python3 src/labels.py --write PATH.csv --images-only
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# The 12 competition targets.  Spelling (spaces, apostrophe) matches the CSV
# headers exactly; do not "tidy" these strings.
# ---------------------------------------------------------------------------
KNEE_LABELS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

# ---------------------------------------------------------------------------
# CALIBRATION
#
# Everything in this block was MEASURED against the 58 gold studies, by running
# the rules and counting how often each verdict was right (696 label cells):
#
#     verdict          n     P(y=1)
#     positive       208      0.784
#     weak_positive   84      0.512
#     unmentioned    222      0.126     <- the number that matters
#     weak_negative   32      0.062
#     negative       150      0.027
#
# That ordering is the whole product: the rules do not just say yes/no, they
# say yes/no with a known error rate, and the soft target IS that error rate.
#
# HONEST CAVEAT: these are aggregate statistics estimated on the same 58
# studies train.py validates on.  They are per-label reliabilities, not
# per-study information -- no validation study's label is recoverable from
# them -- but they are not free of the gold set either.  ``calibrate_states``
# and ``calibrate_priors`` accept a ``folds`` argument so a caller that cares
# can refit them on the training folds only.
# ---------------------------------------------------------------------------

# Probability that an UNMENTIONED finding is nevertheless present.
# Aggregate over the 222 unmentioned cells: 0.126, 95% CI [0.082, 0.170].
#
# NOTE this is lower than the 0.29 the previous implementation reported, and
# both numbers are correct: 0.29 was the residual for a rule set with much
# lower coverage.  The stronger the rules, the more selected the leftover
# "unmentioned" pool is, and the fewer real positives hide in it.  Do NOT
# transplant one project's prior onto another project's rules.
UNMENTIONED_PRIOR = 0.20        # deliberately above the point estimate

# P(y=1 | rules found no mention), per label: the raw count on the gold set
# Beta-shrunk towards UNMENTIONED_PRIOR (pseudo_count=8), because some labels
# have only a handful of unmentioned rows.  Never 0: a finding nobody wrote
# down is not a finding that is absent.
LABEL_PRIORS: Dict[str, float] = {
    "ACL": 0.114,
    "MCL": 0.080,
    "Medial Meniscus": 0.173,
    "Lateral Meniscus": 0.094,
    "Medial OA": 0.070,
    "Lateral OA": 0.068,
    "PF OA": 0.207,
    "Effusion": 0.236,
    "Synovitis": 0.355,     # the label reports mention least, and miss most
    "Baker's": 0.065,
    "Contusion": 0.130,
    "Fracture": 0.175,
}

# Aggregate reliability per verdict, used as the shrinkage target and as the
# fallback for a label that is not in LABEL_STATE_PROB.
STATE_RELIABILITY: Dict[str, float] = {
    "positive": 0.784,
    "weak_positive": 0.512,
    "weak_negative": 0.062,
    "negative": 0.027,
}

# Per-label reliability of each verdict: measured on gold, Beta-shrunk towards
# STATE_RELIABILITY with pseudo_count=10, then made monotone
# (negative <= weak_negative <= prior <= weak_positive <= positive).
#
# Read the extremes as the honest ones:  a rule-asserted ACL tear is right 91%
# of the time; a rule-asserted LATERAL OA is right 65% of the time, because the
# gold rater's severity threshold for that compartment is stricter than the
# reports' wording.  Both deserve to be trained on -- at their own strength.
LABEL_STATE_PROB: Dict[str, Dict[str, float]] = {
    "ACL":              {"positive": 0.914, "weak_positive": 0.596, "weak_negative": 0.103, "negative": 0.008},
    "MCL":              {"positive": 0.856, "weak_positive": 0.478, "weak_negative": 0.072, "negative": 0.007},
    "Medial Meniscus":  {"positive": 0.828, "weak_positive": 0.525, "weak_negative": 0.056, "negative": 0.011},
    "Lateral Meniscus": {"positive": 0.917, "weak_positive": 0.506, "weak_negative": 0.048, "negative": 0.043},
    "Medial OA":        {"positive": 0.819, "weak_positive": 0.509, "weak_negative": 0.062, "negative": 0.056},
    "Lateral OA":       {"positive": 0.645, "weak_positive": 0.548, "weak_negative": 0.061, "negative": 0.055},
    "PF OA":            {"positive": 0.744, "weak_positive": 0.570, "weak_negative": 0.062, "negative": 0.011},
    "Effusion":         {"positive": 0.716, "weak_positive": 0.471, "weak_negative": 0.062, "negative": 0.018},
    "Synovitis":        {"positive": 0.772, "weak_positive": 0.512, "weak_negative": 0.041, "negative": 0.027},
    "Baker's":          {"positive": 0.754, "weak_positive": 0.465, "weak_negative": 0.059, "negative": 0.014},
    "Contusion":        {"positive": 0.735, "weak_positive": 0.445, "weak_negative": 0.033, "negative": 0.018},
    "Fracture":         {"positive": 0.811, "weak_positive": 0.507, "weak_negative": 0.034, "negative": 0.017},
}

# Fallback band edges when no calibration entry exists.  Never 1.0 or 0.0: the
# rules are good, not infallible, and a hard target on a wrong call is the most
# expensive kind of training-set error.
P_POSITIVE = 0.90
P_WEAK_POSITIVE = 0.62
P_NEGATIVE = 0.08

# Within a verdict band, nudge the probability by the strength of the evidence
# so two "positive" studies with different evidence are still rankable.  Small
# enough that it can never cross into another band.
CAL_SPREAD = 0.03

# Confidence reported for an UNMENTIONED cell.
#
# The TARGET of such a cell is the calibrated prior and never 0 -- writing 0
# there would fabricate a confident negative, which is the whole point of this
# module.  The CONFIDENCE is a separate question: a prior is knowledge about a
# population, not about this knee, so at 0.0 the cell contributes its prior to
# nothing and costs the loss nothing.  train.py's derived_cell_weights()
# multiplies this straight into the per-cell loss weight and is written against
# exactly that contract, so leave it at 0.0 unless you deliberately want the
# net trained towards base rates; then the prior in the target column is what
# it will be trained towards.
UNMENTIONED_CONFIDENCE = 0.0

# Evidence weights, in log-odds.
W_POS = 2.6
W_WEAK = 0.85
W_NEG = -2.6
W_WEAK_NEG = -1.1
# When a report both asserts and denies a finding (findings section vs
# impression, or two knees filed under one study), trust the assertion more:
# radiologists write down what they see and enumerate normals by template.
NEG_DISCOUNT_WHEN_POS = 0.45
LOGIT_CLIP = 3.2


# ===========================================================================
# 1. Normalisation
# ===========================================================================

# Characters with no canonical decomposition that we still want folded.
_TRANSLIT = {
    "ı": "i",   # Turkish dotless i
    "İ": "i",   # Turkish capital dotted I (lower() would leave a combiner)
    "ß": "ss",  # German sharp s -> Aussenmeniskus
    "đ": "d", "Đ": "d",   # Croatian d-stroke
    "ł": "l", "Ł": "l",   # Polish l-stroke
    "ø": "o", "Ø": "o",
    "æ": "ae", "Æ": "ae",
    "œ": "oe", "Œ": "oe",  # French oedeme
    "ς": "σ",             # Greek final sigma -> sigma
    "’": "'", "‘": "'", "´": "'",
    "–": "-", "—": "-", "‑": "-",
    " ": " ",
}


def normalize_text(text) -> str:
    """Lowercase, fold diacritics, and normalise whitespace.

    Fold-first, decompose-second, so Turkish/Croatian letters that have no
    canonical decomposition still collapse.  After this:

        "Sin anomalias" == "Sin anomalias"   (accent variants unify)
        "Ön çapraz bağ" -> "on capraz bag"
        "µηνίσκου"      -> "μηνισκου"        (MICRO SIGN folds to Greek mu)
        "Außenmeniskus" -> "aussenmeniskus"

    Greek and Cyrillic keep their own script (they are matched by their own
    patterns); only their accents are stripped.
    """
    if text is None:
        return ""
    s = str(text)
    if not s:
        return ""
    s = "".join(_TRANSLIT.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = s.replace("ς", "σ")          # final sigma again, post-lower
    s = re.sub(r"[ \t\r\f\v]+", " ", s)
    return s


# ===========================================================================
# 2. Language identification
#
# No langdetect / langid in this environment (checked), and adding a pip
# dependency to a Kaggle *offline* pipeline is not free.  Script detection plus
# closed-class function words is more than accurate enough on a 9-language,
# single-domain corpus.
# ===========================================================================

_LANG_STOPWORDS = {
    "en": r"\b(the|and|is|are|no|of|with|there|without|intact|normal|joint|knee)\b",
    "es": r"\b(de|la|el|los|las|del|sin|con|no|se|en|y|que|rodilla|hallazgos)\b",
    # Folding strips the Turkish-specific letters, so the profile leans on
    # closed-class words and the -dIr / izlen- verb endings instead.
    "tr": (r"\b(ve|ile|yok|yoktur|var|bir|bu|ise|daha|normaldir|normaldi|"
           r"izlenmistir|izlendi|izleniyor|izlenmektedir|mevcut|mevcuttur|"
           r"diz|bulgular|kemik|eklem|sivi|baglar|kesiminde|duzeyinde|"
           r"gorunum|gorunumde|saptanmistir|olarak|sonuc|hafif|artmis)\b"),
    # French needs its contractions and pronouns: a short French sentence is
    # mostly "il n'y a pas de ...", whose only content words ("de", "la") are
    # also Spanish, so a stopword list without them mis-tags French as Spanish.
    "fr": (r"\b(le|la|les|des|du|au|aux|est|sont|sans|avec|dans|pas|une|un|et|"
           r"il|elle|ne|ou|par|sur|cette|genou|kyste|croise|croises|"
           r"articulaire|epanchement|constatations)\b|\b[dlnjscmt]'"),
    "nl": r"\b(de|het|een|van|geen|met|zijn|niet|er|en|op|bij|wordt|knie|bevindingen)\b",
    "de": r"\b(der|die|das|den|dem|und|nicht|kein|keine|keiner|mit|ist|im|bei|ohne)\b",
    "hr": r"\b(uz|bez|se|je|na|nema|u|i|prati|zgloba|zglob|meniska|menisk|hrskavice|urednog)\b",
    "it": r"\b(del|della|dei|delle|nel|nella|con|senza|non|il|lo|gli|ginocchio)\b",
    "pt": r"\b(do|da|dos|das|no|na|com|sem|nao|para|joelho)\b",
}
_LANG_RE = {k: re.compile(v) for k, v in _LANG_STOPWORDS.items()}


def _dominant_script(folded: str) -> str:
    cyr = grk = lat = 0
    for ch in folded:
        o = ord(ch)
        if 0x0400 <= o <= 0x04FF:
            cyr += 1
        elif 0x0370 <= o <= 0x03FF:
            grk += 1
        elif ch.isalpha() and o < 0x0250:
            lat += 1
    if cyr >= max(grk, lat):
        return "cyrillic" if cyr else "latin"
    if grk >= lat:
        return "greek" if grk else "latin"
    return "latin"


def detect_language(text) -> str:
    """Return an ISO-ish language tag for a report.

    Distribution over data_subset/train.csv (4407 rows), measured:
        en 1741, es 681, tr 546, hr 406, el 321, de 260, bg 220, nl 151, fr 80
    """
    folded = normalize_text(text)
    if not folded.strip():
        return "unknown"
    script = _dominant_script(folded)
    if script == "cyrillic":
        return "bg"          # corpus is Bulgarian; other cyrillic reads the same rules
    if script == "greek":
        return "el"
    scores = {k: len(rx.findall(folded)) for k, rx in _LANG_RE.items()}
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "unknown"
    return best


# ===========================================================================
# 3. Segmentation
# ===========================================================================

# Sentence-ish split.  A full stop ends a clause when whitespace follows, or
# when at least four word characters precede it -- Turkish reports routinely
# run sentences together with no space ("izlenmistir.Medial kollateral..."),
# which would otherwise merge an ACL rupture into the MCL sentence, while
# "1.5 Tesla", "II.stupnja", "dif.dg." and "m.quadriceps" must stay whole.
_CLAUSE_SPLIT = re.compile(
    r"(?:\n+|[;•]|(?<![0-9])\.(?=\s|$)|(?<=\w\w\w\w)\.(?![0-9\s])"
    r"|(?<=\s)-\s|(?<=\s)\*|(?<=\s)>|^>)",
    re.M,
)

# Contrastive conjunctions reset the polarity of a sentence, so they end a
# clause too.  "... dISInda normal" (Turkish "apart from these, normal") and
# "... otherwise unremarkable" are the two commonest shapes in this corpus.
_CONTRAST_SPLIT = re.compile(
    r"\b(?:but|however|except|apart from|aside from|"
    r"ancak|disinda|haricinde|"
    r"salvo|excepto|pero|"
    r"jedoch|abgesehen von|"
    r"behalve|"
    r"sauf|"
    r"osim|"
    r"освен)\b"
)


def split_clauses(folded: str) -> List[str]:
    """Split folded text into negation scopes.

    A trailing-colon fragment is a SECTION HEADER, not a clause: the structured
    English and French templates write

        Fractures :
        Aucune.

        Medial collateral ligament (MCL):
        Normal.

    so the header is glued to the following line.  Without this, "fractures :"
    is a bare positive mention of a fracture and "aucune"/"normal" floats away
    with nothing to negate -- it inverts the label.
    """
    raw: List[str] = []
    for part in _CLAUSE_SPLIT.split(folded):
        for sub in _CONTRAST_SPLIT.split(part):
            sub = sub.strip(" \t,-")
            if sub:
                raw.append(sub)

    out: List[str] = []
    pending = ""
    for seg in raw:
        if pending:
            seg = pending + " " + seg
            pending = ""
        if seg.endswith(":") and len(seg) <= 80:
            pending = seg
            continue
        out.append(seg.strip(" \t,:-"))
    if pending:
        out.append(pending.strip(" \t,:-"))
    return [c for c in out if c]


# ===========================================================================
# 4. Lexicons
#
# Everything below is written in FOLDED form (see normalize_text): lowercase,
# no diacritics, Turkish/Croatian letters already collapsed to ASCII.
# ===========================================================================

def _rx(pattern: str) -> "re.Pattern":
    return re.compile(pattern, re.I)


# --- laterality ------------------------------------------------------------
MEDIAL = (
    r"medial\w*|medyal\w*|mediaal|mediale\w*|medijaln\w*|"
    r"intern[oa]s?|innen|"
    r"εσω|"                                    # el: eso (inner)
    r"медиал\w*|вътреш\w*"  # bg
)
LATERAL = (
    r"lateral\w*|lateraal|laterale\w*|lateraln\w*|"
    r"extern[oa]s?|aussen|"
    r"εξω|"                                    # el: exo (outer)
    r"латерал\w*|външ\w*"        # bg
)
_MEDIAL_RE = _rx(MEDIAL)
_LATERAL_RE = _rx(LATERAL)

# --- negation, per language ------------------------------------------------
# Direction matters and is NOT the same across these nine languages.
#
#   PRE   the negator stands before the finding:
#         "no joint effusion", "sin derrame", "geen hydrops", "kein Erguss",
#         "bez znakova rupture", "δεν παρατηρούνται", "няма данни за руптура"
#   POST  the negator stands after it -- this is Turkish's whole strategy
#         ("... yoktur", "... izlenmedi", "... eslik etmiyor"), plus the
#         template answer form "Baker cyst: None." / "Fractures : Aucune."
#   SYM   an explicit normality assertion, which lands on either side
#         ("ACL is intact", "Normaal voorkomen menisci", "menisküs normaldir")
#
# A symmetric window looks harmless until it eats a trailing qualifier:
# "Subchondral insufficiency fracture ... WITHOUT articular surface collapse"
# was being read as "no fracture".  That is one of the 58 gold studies.
_NEG_PRE = (
    r"\bno\b|\bnot\b|\bnor\b|without|absence|absent|free of|negative for|"
    r"\bsin\b|\bausencia\b|\bausente\b|\bningun\w*\b|\bnegativo\b|"
    r"\bgeen\b|\bzonder\b|\bniet\b|"
    r"\bkein\w*\b|\bohne\b|\bnicht\b|"
    r"\bpas de\b|\bpas d'\b|\bsans\b|\baucun\w*\b|"
    r"\bbez\b|\bnema\b|\bnisu\b|\bnije\b|\bne prati\b|\bne pokazuje\b|"
    r"δεν|χωρισ|ουδεν\w*|"
    r"няма|\bбез\b|липсв\w*|не се"
)
_NEG_POST = (
    r"\byok\b|\byoktur\b|\bizlenmedi\b|\bizlenmemistir\b|\bizlenmemektedir\b|"
    r"\bsaptanmadi\b|\bsaptanmamistir\b|\bgorulmedi\b|\bgozlenmedi\b|"
    r"\bdegildir\b|\bdegil\b|\betmiyor\b|\betmemektedir\b|\bmevcut degil\b|"
    r"\bnone\b|\bnil\b|\baucune?\b|\bningun\w*\b|\bnegatif\b"
)
# "normal", "intact", "preserved", "unremarkable" -- an explicit normality
# assertion, which is the commonest way these reports state a negative.
_NEG_NORMAL = (
    r"\bnormal\w*\b|\bintact\w*\b|unremarkable|preserved|\bpatent\b|"
    r"within normal limits|\bwnl\b|\bcongruent\b|no evidence|"
    r"conservad\w*|integr[oa]s?\b|dentro de limites normales|sin alteraciones|"
    r"sin signos|sin cambios|\bnormales?\b|"
    r"normaldir|normaldi|korunmus|korunmustur|dogal|dogaldir|olagan|"
    r"uredan|uredn\w*|primjeren\w*|ocuvan\w*|odrzanog kontinuiteta|bez znakova|"
    r"unauffallig|regelrecht|\bintakt\w*\b|erhalten|"
    r"normaal|normale|ongestoord|"
    r"\bindemne\b|sans particularite|"
    r"φυσιολογικ\w*|ακεραι\w*|"   # el: fysiologik / akerai
    r"εντοσ του φυσιολογικ\w*|"
    r"αξιολογ\w*|"                                                # el: 'no notable findings'
    r"нормал\w*|запазен\w*|интакт\w*|"  # bg
    r"без особености|съхранен\w*|б\.о\."
)
_NEG_PRE_RE = _rx(_NEG_PRE)
_NEG_POST_RE = _rx(_NEG_POST)
_NEG_NORMAL_RE = _rx(_NEG_NORMAL)

# Hedging -> downgrade a positive to weak_positive.
_HEDGE_RE = _rx(
    r"suspect\w*|suspicious|possible|possibly|probable|probably|likely|"
    r"cannot be excluded|can't totally excluded|cannot totally|r/o\b|rule out|"
    r"question\w*|\bmay\b|\bmight\b|equivocal|indeterminate|"
    r"sospech\w*|probable\w*|sugier\w*|sugest\w*|impresiona|no descartar|podria|"
    r"suphel\w*|olasilik\w*|dusundur\w*|supheli|"
    r"moguce|vjerojatno|dif\.?dg|sumnja|"
    r"moglich\w*|verdacht|\bdd\b|wohl|"
    r"mogelijk\w*|verdenking|"
    r"probablement|suspicion|"
    r"πιθαν\w*|υποψ\w*|"
    r"вероятн\w*|съмнен\w*|не може да се изключи"
)

# --- pathology cue: tear / rupture -----------------------------------------
TEAR = (
    r"\btears?\b|\btorn\b|tearing|"
    r"ruptur\w*|\brupt\b|rotur\w*|desgarr\w*|disrupt\w*|maceration|"
    r"yirtik\w*|yirtig\w*|yirti\b|kopma|"
    r"\briss\w*|einriss\w*|abriss\w*|ausriss\w*|"
    r"scheur\w*|ruptuur\w*|"
    r"dechirure\w*|"
    r"puknuc\w*|"
    r"ρηξ\w*|ρηγμ\w*|"                                        # el: rixi
    r"руптур\w*|скъсван\w*|разкъсван\w*|скъсан\w*"
)
_TEAR_RE = _rx(TEAR)

# --- pathology cue: degeneration (weak) ------------------------------------
DEGEN = (
    r"degenerat\w*|degenerativ\w*|dejenerat\w*|dejenerasyon\w*|meniskopat\w*|"
    r"mucoid|mukoid|mukoz\w*|myxoid|miksoid|"
    r"εκφυλ\w*|"                                                      # el: ekfyl (degeneration)
    r"дегенератив\w*|дегенерац\w*"
)
_DEGEN_RE = _rx(DEGEN)

# meniscal extrusion / subluxation, a real but weaker sign
EXTRUSION = (
    r"extrus\w*|extruz\w*|ekstru\w*|ekstruz\w*|"
    r"υπεξαρθρ\w*|експулсир\w*"
)
_EXTRUSION_RE = _rx(EXTRUSION)

# ligament sprain / partial injury (weak on its own)
SPRAIN = (
    r"sprain\w*|esguince\w*|\bstrain\b|distorsi\w*|zerrung|verstauchung|"
    r"\bsprejn\b|\blaksite\b|elonge|"
    r"κακωσ\w*|дисторз\w*"
)
_SPRAIN_RE = _rx(SPRAIN)

# generic "injury / lesion" wording
INJURY = (
    r"\binjur\w*|\blesion\w*|\blasion\w*|lezyon\w*|lezij\w*|"
    r"grade\s*(?:ii+|2|3|iv|4)\b|grado\s*(?:ii+|2|3)\b|grad[e]?\s*(?:ii+|2|3)\b|"
    r"βλαβ\w*|лези\w*|увред\w*"
)
_INJURY_RE = _rx(INJURY)


# --- anatomy anchors -------------------------------------------------------
MENISCUS = r"menisc\w*|menisk\w*|menisq\w*|μηνισκ\w*|мениск\w*"
_MENISCUS_RE = _rx(MENISCUS)
# German writes the side into the noun.
_MED_MENISCUS_CPD_RE = _rx(r"innenmenisk\w*|innenmeniscus")
_LAT_MENISCUS_CPD_RE = _rx(r"aussenmenisk\w*|aussenmeniscus")

ACL_ANCHOR = (
    r"\bacl\b|\bacl'\w*|anterior\w* cruciate|cruciate anterior|"
    r"ligamento cruzado anterior|\blca\b|\blcao\b|"
    r"\bon capraz\w*|\banterior capraz\w*|\boncapraz\w*|\bocb\b|"
    r"prednj\w+ (?:krizn\w+|ukrizen\w+)|prednj\w+ sveza|"
    r"vorder\w* kreuzband|\bvkb\b|"
    r"voorste kruisband|"
    r"ligament croise anterieur|"
    r"προσθι\w* χιαστ\w*|"                 # el
    r"предн\w* кръст\w*"                        # bg
)
_ACL_RE = _rx(ACL_ANCHOR)

MCL_ANCHOR = (
    r"\bmcl\b|medial collateral|collateral medial|"
    r"ligamento colateral medial|complejo medial|\blcm\b|"
    r"medial\w* kollateral\w*|\bic yan bag\w*|"
    r"medijaln\w* kolateraln\w*|kolateraln\w* medijaln\w*|"
    r"innenband|mediale?s? (?:kollateralband|seitenband)|"
    r"mediaal collateraal|mediale collaterale? ligament\w*|"
    r"ligament collateral(?:e)? (?:medial|interne)|"
    r"εσω πλαγι\w*|"                                     # el
    r"медиал\w* (?:колатерал\w*|лигамен\w*)"
)
_MCL_RE = _rx(MCL_ANCHOR)

# Generic mentions ("cruciate and collateral ligaments are normal").  These can
# establish a NEGATIVE but never a POSITIVE: a positive could belong to the PCL
# or the LCL, which are not labels in this task.
GENERIC_CRUCIATE = (
    r"cruciate ligaments?|ligamentos cruzados|capraz (?:ve yan )?bag\w*|"
    r"krizn\w+ ligament\w*|ukrizen\w+ sveze|kreuzband\w*|kruisband\w*|"
    r"ligaments croises|χιαστ\w+ συνδεσμ\w*|"
    r"χιαστοι|кръстн\w* връзк\w*"
)
_GEN_CRUCIATE_RE = _rx(GENERIC_CRUCIATE)

GENERIC_COLLATERAL = (
    r"collateral ligament\w*|ligamentos colaterales|\bcolaterales\b|"
    r"\byan bag\w*|"
    r"kollateral ligaman\w*|kolateraln\w* ligament\w*|kollateralband\w*|"
    r"collaterale ligament\w*|ligaments collateraux|seitenband\w*|"
    r"πλαγι\w* συνδεσμ\w*|"
    r"колатерал\w*"
)
_GEN_COLLATERAL_RE = _rx(GENERIC_COLLATERAL)

# --- cartilage / osteoarthritis --------------------------------------------
# Split into "pathology is explicit" vs "just names cartilage".  A positive
# needs the former; a negative only needs the latter plus a negation.
OA_PATH = (
    r"\boa\b|"                       # "OA of medial compartment", "OA promjene"
    r"osteoarthr\w*|osteoartrit\w*|osteoartroz\w*|artros\w*|arthros\w*|"
    r"artrotsk\w*|artrot\w*|"
    r"gonartro\w*|gonarthro\w*|goanrtro\w*|"
    r"chondros\w*|hondros\w*|chondromalac\w*|hondromalac\w*|kondromalaz\w*|"
    r"chondropath\w*|chondropat\w*|hondropat\w*|condropat\w*|kondropat\w*|"
    r"osteophyt\w*|osteofit\w*|osteofyt\w*|"
    r"οστεοφυτ\w*|οστεοαρθρ\w*|"
    r"остеофит\w*|артроз\w*|гонартроз\w*|"
    r"kraakbeenlijden|kraakbeenverlies|knorpelschad\w*|knorpelverlust|"
    r"denudacij\w*|denudation|joint space narrowing|"
    # "osteochondral" on its own is a FRACTURE descriptor in this corpus
    # ("остеохондрална фрактура"), so it only counts when it heads a defect.
    r"osteochondral (?:defect|lesion|injury|body|bodies)|osteochondritis"
)
_OA_PATH_RE = _rx(OA_PATH)

OA_ANAT = (
    r"cartilag\w*|cartilago?s?\b|chondral|condral\w*|kondral\w*|"
    r"kikirdak\w*|kraakbeen\w*|knorpel\w*|hrskavic\w*|"
    r"eklem aralig\w*|zglobn\w* prostor\w*|gewrichtsspleet|"
    r"χονδρ\w*|хрущял\w*"
)
_OA_ANAT_RE = _rx(OA_ANAT)

OA_DAMAGE = (
    r"\bloss\b|losses|\bdefect\w*|thinning|thinned|fissur\w*|fisur\w*|"
    r"ulcer\w*|erosi\w*|erozi\w*|eroziv\w*|\bwear\b|wearing|fibrillation|"
    r"heterogene\w*|irregular\w*|delaminat\w*|damage\w*|abnorm\w*|"
    r"kayb\w*|kayip\w*|incelme\w*|dejenerasyon\w*|daral\w*|"
    r"verlies|verdunn\w*|ausgedunnt|abnutzung|verschmalerung|"
    r"stanjen\w*|reducira\w*|reducir\w*|fisure|"
    r"perdida|adelgaza\w*|pinzamiento|"
    r"λεπτυν\w*|διαβρωσ\w*|εξαλειψ\w*|"
    r"изтъняв\w*|увред\w*"
)
_OA_DAMAGE_RE = _rx(OA_DAMAGE)

# Femorotibial compartment nouns -- the side word must attach to one of these
# for Medial OA / Lateral OA.
FT_NOUN = (
    r"compartment\w*|kompart\w*|compartimento\w*|comparti?ment\w*|"
    r"condyl\w*|kondil\w*|kondul\w*|kondyl\w*|condil\w*|"
    r"plateau\w*|plato\w*|platillo\w*|meseta\w*|tibiaplateau|"
    r"femorotibial\w*|femoro-tibial\w*|tibiofemoral\w*|femorotibiaal|"
    r"joint line|jointline|eklem aralig\w*|"
    r"διαμερισμ\w*|μεσαρθρ\w*|"
    r"κονδυλ\w*|κνημιαι\w*|μηριαι\w*|"
    r"компартм\w*|кондил\w*|кондул\w*|плато"
)
_FT_NOUN_RE = _rx(FT_NOUN)

# Patellofemoral nouns.
PF_NOUN = (
    r"patell\w*|patela\w*|patele\b|rotulian\w*|rotul\w*|retropatell\w*|"
    r"femoropatell\w*|patelofemoral\w*|patellofemoral\w*|femoro-?patellar\w*|"
    r"trochlea\w*|troklea\w*|trochlear\w*|troclea\w*|trohle\w*|troklear\w*|"
    r"\bpf zglob\w*|\bfp zglob\w*|\bpf\b|"
    r"επιγονατιδ\w*|τροχιλ\w*|"
    r"пател\w*|трохле\w*"
)
_PF_NOUN_RE = _rx(PF_NOUN)

# All-compartment statements.
_TRICOMPARTMENTAL_RE = _rx(
    r"tricompartmental|tri-compartmental|three compartment\w*|all three compartm\w*|"
    r"three compartmens|compartmens|"
    r"gonartro\w*|gonarthro\w*|goanrtro\w*|panartro\w*|"
    r"her uc kompartman|uc kompartman|"
    r"гонартроз\w*"
)

# --- effusion --------------------------------------------------------------
EFFUSION = (
    r"effusion\w*|effsuion|epanchement\w*|derrame\w*|"
    r"erguss|gelenkerguss|gelenkserguss|hydrops|hidrops|"
    r"izljev\w*|izliv\w*|"
    r"efuzyon|efusion|"
    r"joint fluid|intra-?articular fluid|fluid (?:in|within) the joint|"
    r"liquido articular|liquido intraarticular|"
    r"hemartros\w*|hemarthros\w*|haemarthros\w*|"
    r"sivi (?:miktari|artisi)|sivi artis\w*|eklem(?:de|inde)? sivi|"
    r"mayii?\b|mayi artis\w*|"
    r"vocht in het gewricht|"
    r"συλλογη υγρου|υδραρθρ\w*|ενδαρθρικ\w*|"
    r"ставен излив|излив в став\w*|хеморагичен излив|ставният излив"
)
_EFFUSION_RE = _rx(EFFUSION)

# --- synovitis -------------------------------------------------------------
SYNOVITIS = (
    r"synovit\w*|sinovit\w*|synovialit\w*|reizsynovial\w*|synovitis|"
    r"synovial thickening|thickened synovi\w*|synovial hypertroph\w*|"
    r"hypertrophy of the synovium|thickening of the synovi\w*|"
    r"proliferacij\w* sinovij\w*|proliferaci\w* na sinovi\w*|"
    r"sinovij\w* zadeblj\w*|verdikking\w*[^.]{0,30}synovium|synoviale verdikking|"
    r"pachynsi tou ymena|"
    r"υμενιτ\w*|συνοβιτ\w*|"
    r"παχυνση του υμεν\w*|"
    r"синовит\w*|синовиалн\w* пролифер\w*|"
    r"пролифераци\w* на синови\w*|"
    r"pannus|panus|hoffit\w*|hoffitis"
    # NOTE: descriptive Hoffa forms ("pinzamiento ... hoffa", "fat pad
    # impingement/oedema") were tried and reverted.  Measured on the 58 gold
    # studies they raise Synovitis coverage 0.379 -> 0.466 and recall
    # 0.481 -> 0.519, but drop precision 0.765 -> 0.737 and the label's own
    # AUC 0.706 -> 0.700.  More cells, slightly worse ones -- not a win for a
    # soft-target source, whose value is the ranking it produces.
)
_SYNOVITIS_RE = _rx(SYNOVITIS)

# --- Baker's / popliteal cyst ---------------------------------------------
BAKER = (
    r"baker\w*|bakerova|baker-?zyste|bakerzyste|"
    r"popliteal cyst\w*|popliteal fossa cyst|cyst\w* popliteal\w*|"
    r"quiste\w* poplite\w*|kyste\w* poplite\w*|"
    r"poplitealn\w* cist\w*|poplitealn\w* ciste|poplitealne ciste|"
    r"popliteale cyste|poplitea?le? cyst\w*|"
    r"κυστ\w*[^.]{0,20}baker|baker[^.]{0,20}κυστ\w*|"
    r"бекеров\w*|бейкер\w*|киста на бейкер"
)
_BAKER_RE = _rx(BAKER)

# --- contusion / bone marrow oedema ---------------------------------------
CONTUSION_STRONG = (
    r"contusion\w*|contusio\b|contusiones|contuse\w*|bone bruise\w*|"
    r"kontuzyon\w*|kontuzion\w*|kontuzij\w*|kontuzion\w*|botcontusie|"
    r"bone contusion|osseous contusion|"
    r"impaction (?:injur\w*|fracture)|impactie|"
    r"μωλωπ\w*|"                                              # el: molopas (bruise)
    r"контузион\w*|контузия"
)
CONTUSION_WEAK = (
    r"bone marrow o?edema|bone marrow o?edeem|marrow o?edema|"
    r"osseous o?edema|subchondral o?edema|subchondral bone o?edema|"
    r"edema oseo|edema de la medula osea|edema oseo|"
    r"kemik iligi odem\w*|kemik odem\w*|subkondral odem\w*|"
    r"kostani edem|kostane srzi edem|edem kosti|"
    r"knochenmarkodem|knochenodem|knochenmarksodem|botoedeem|beenmergoedeem|"
    r"o?edeme osseux|o?edeme de la moelle|"
    r"οστικ\w* οιδημ\w*|οστεομυελιк\w*|οστεομυελικ\w* οιδημ\w*|"
    r"костно-?мозъчен едем|костномозъчен едем|едем в кост\w*"
)
_CONTUSION_STRONG_RE = _rx(CONTUSION_STRONG)
_CONTUSION_WEAK_RE = _rx(CONTUSION_WEAK)
# Marrow oedema explained by degeneration is not a contusion.
#
# SUBCHONDRAL oedema is the classic osteoarthritic bone-marrow lesion: it abuts
# the joint surface under damaged cartilage.  A traumatic contusion is marrow
# oedema in the cancellous bone, described with an impaction/trauma context or
# named outright ("contusion", "bone bruise").  CONTUSION_WEAK lists
# "subchondral oedema" as a cue, which is what made Contusion the joint-worst
# label for precision (0.562): 14 false positives on the 58 gold studies, the
# largest single group being "with subchondral bone edema is detected".
#
# Oedema ADJACENT TO A FRACTURE is likewise attributed to the fracture -- the
# report is describing one injury, and Fracture is already its own label.
_CONTUSION_SUPPRESS_RE = _rx(
    r"degenerativ\w*|degenerat\w*|dejenerat\w*|reactive|reaktiv\w*|"
    r"cystic change|subchondral cyst\w*|osteoarthr\w*|artros\w*|"
    r"related to overlying chondros\w*|insufficiency|"
    r"subchondral|subkondral\w*|subhondral\w*|subcondral\w*|"
    r"fracture line|adjacent bone marrow|"
    r"(?:adjacent|surrounding|peri-?)\s*(?:to\s+)?(?:the\s+)?fractur\w*"
)

# --- fracture --------------------------------------------------------------
FRACTURE = (
    r"fracture\w*|fractur\w*|fraktur\w*|fractura\w*|fracturas|"
    r"fissure fracture|"
    # Turkish k -> g consonant mutation: kirik / kirigi / kiriga are one word.
    r"kiri[kg]\w*|"
    r"prijelom\w*|prelom\w*|"
    r"fractuur\w*|breuk\b|impactiefractuur|"
    r"knocherner? ausriss|knochernem? ausriss|"
    r"καταγμ\w*|"                                        # el: katagma
    r"фрактур\w*|счупван\w*"
)
_FRACTURE_RE = _rx(FRACTURE)
# "fracture" appearing only inside these is not a fracture claim.
_FRACTURE_CONTEXT_SUPPRESS_RE = _rx(r"microtrabecular|microdokid|μικροδοκιδ\w*")

# --- "the bones are normal" ------------------------------------------------
# A negated bone/marrow statement is evidence against Fracture and Contusion
# specifically.  This is what carries the 37-study Turkish normal template
# ("Eklem kikirdaklari ve kemikler normal") and the English "Bones: Normal."
BONE_ANAT = (
    r"\bbones?\b|osseous|bone marrow|marrow signal|"
    r"kemik\w*|kostan\w*|kost\w*|kostiju|"
    r"knochen\w*|knochenmark\w*|"
    r"\bbot\b|botten|beenmerg\w*|"
    r"hueso\w*|medula osea|"
    r"\bos\b|osseux|moelle osseuse|"
    r"οστ(?:α|ων|ικ\w*|εο\w*|ουν)|μυελ\w*|"
    r"кост\w*|костномозъч\w*"
)
_BONE_ANAT_RE = _rx(BONE_ANAT)

# --- "the whole study is normal" ------------------------------------------
# ~183 studies in train.csv share their report text with another study; those
# are boilerplate NORMAL templates (a 37-study Turkish one, the Spanish
# "Sin anomalias" with and without the accent, an English one), NOT duplicate
# patients.  They are the single largest block of genuine negatives in the
# corpus and must not be dropped as duplicates.
GLOBAL_NORMAL = (
    r"sin anomalias|estudio normal|resonancia normal|sin hallazgos patologicos|"
    r"normal study|normal mri|unremarkable (?:mri|study|exam\w*)|"
    r"no (?:significant )?abnormalit\w*|conclusion:? normal|"
    r"normal sinirlarda|dogal sinirlarda|olagan sinirlarda|"
    r"uredan nalaz|bez patoloskih promjena|"
    r"unauffalliger befund|regelrechter befund|kein pathologischer befund|"
    r"geen afwijkingen|normaal onderzoek|"
    r"examen normal|pas d'anomalie|sans anomalie|"
    r"χωρισ παθολογικα ευρηματα|φυσιολογικη μελετη|"
    r"нормална находка|без патологични промени"
)
_GLOBAL_NORMAL_RE = _rx(GLOBAL_NORMAL)
W_GLOBAL_NORMAL = -2.0


# ===========================================================================
# 5. Engine
# ===========================================================================

@dataclass
class Finding:
    """Rule output for one (study, label)."""
    label: str
    prob: float                  # soft target in [0, 1]
    confidence: float            # 0..1; drives the training sample weight
    state: str                   # positive | weak_positive | negative | unmentioned
    logit: float                 # evidence in log-odds, prior excluded
    evidence: List[str] = field(default_factory=list)

    def as_tuple(self):
        return (self.prob, self.confidence, self.state)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


PRE_WINDOW = 60
POST_WINDOW = 70
SYM_WINDOW = 70


def _normal_near(clause: str, span: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Span of the normality assertion nearest to `span`, within SYM_WINDOW."""
    lo = max(0, span[0] - SYM_WINDOW)
    hi = min(len(clause), span[1] + SYM_WINDOW)
    best = None
    best_gap = None
    for m in _NEG_NORMAL_RE.finditer(clause[lo:hi]):
        cand = (m.start() + lo, m.end() + lo)
        g = _gap(span, cand)
        if best_gap is None or g < best_gap:
            best, best_gap = cand, g
    return best


def _negated(clause: str, *spans) -> Optional[str]:
    """Negation cue governing ANY of the given spans, or None.

    Callers pass the pathology cue first and the anatomy anchor LAST, because
    the negator may attach to either:

        "Lateralni menisk BEZ znakova rupture"   -> negator precedes the CUE
        "the meniscus is NOT torn"               -> negator precedes the CUE
        "Medyal ve lateral menisküs NORMAL"      -> normality follows the ANCHOR
        "NO lateral meniscal tear"               -> negator precedes both

    KNOWN LIMIT: a single clause that carries a finding for one structure and
    a normality word for another -- "complex tear of the medial meniscus and
    normal lateral meniscus" -- cancels itself out and lands on the prior.  A
    distance tie-break was tried and measured WORSE on the gold set (macro AUC
    0.8692 -> 0.8678, Lateral OA 0.822 -> 0.805), because these reports almost
    always put the two structures in separate sentences and the tie-break only
    misfired on the ones that do not.  Left as a known cancellation.
    """
    real = [s for s in spans if s is not None]
    if not real:
        return None

    # Directional negators are unambiguous; take them first.
    for span in real:
        m = _NEG_PRE_RE.search(clause[max(0, span[0] - PRE_WINDOW):span[0]])
        if m:
            return m.group(0)
        m = _NEG_POST_RE.search(clause[span[1]:span[1] + POST_WINDOW])
        if m:
            return m.group(0)

    anchor = real[-1]
    nearest = None
    for span in real:
        cand = _normal_near(clause, span)
        if cand is not None and (nearest is None
                                 or _gap(anchor, cand) < _gap(anchor, nearest)):
            nearest = cand
    if nearest is None:
        return None
    return clause[nearest[0]:nearest[1]]


def _gap(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """Character gap between two spans; 0 if they touch or overlap."""
    return max(0, b[0] - a[1], a[0] - b[1])


def _iter_side_mentions(clause: str, anchor_re, window: int = 70,
                        coord_gap: int = 28):
    """Yield ('medial'|'lateral'|'generic', span) for each anchor occurrence.

    One occurrence can be BOTH sides -- "Medial and lateral menisci are normal",
    "Medyal ve lateral menisküs normal", "Rotura de menisco interno y lateral"
    are among the commonest sentences in this corpus, and collapsing them to a
    single generic mention throws away two verdicts per report.

    But only when the two side words are COORDINATED, i.e. sit next to each
    other.  In "complex tear of the medial meniscus and normal lateral
    meniscus" both side words are inside the window of both anchors, and
    claiming both sides for both anchors would hand the lateral meniscus a tear
    it does not have.  When the side words are far apart, each anchor takes the
    nearer one.
    """
    for m in anchor_re.finditer(clause):
        lo = max(0, m.start() - window)
        hi = min(len(clause), m.end() + window)
        ctx = clause[lo:hi]
        anchor = (m.start() - lo, m.end() - lo)
        med = [x.span() for x in _MEDIAL_RE.finditer(ctx)]
        lat = [x.span() for x in _LATERAL_RE.finditer(ctx)]

        if med and lat:
            if min(_gap(a, b) for a in med for b in lat) <= coord_gap:
                yield "medial", (m.start(), m.end())
                yield "lateral", (m.start(), m.end())
                continue
            dm = min(_gap(anchor, s) for s in med)
            dl = min(_gap(anchor, s) for s in lat)
            yield ("medial" if dm <= dl else "lateral"), (m.start(), m.end())
            continue
        if med:
            yield "medial", (m.start(), m.end())
        elif lat:
            yield "lateral", (m.start(), m.end())
        else:
            yield "generic", (m.start(), m.end())


class _Accumulator:
    """Collects per-label evidence across all clauses of one report."""

    def __init__(self):
        self.pos: Dict[str, List[float]] = {k: [] for k in KNEE_LABELS}
        self.neg: Dict[str, List[float]] = {k: [] for k in KNEE_LABELS}
        self.ev: Dict[str, List[str]] = {k: [] for k in KNEE_LABELS}

    def add(self, label: str, weight: float, why: str):
        if weight > 0:
            self.pos[label].append(weight)
        elif weight < 0:
            self.neg[label].append(weight)
        if len(self.ev[label]) < 8:
            self.ev[label].append(why)

    def positive(self, label, why, hedged=False):
        self.add(label, W_WEAK if hedged else W_POS, why)

    def weak_positive(self, label, why):
        self.add(label, W_WEAK, why)

    def negative(self, label, why, weak=False):
        self.add(label, W_WEAK_NEG if weak else W_NEG, why)


def _structure_rules(acc: _Accumulator, clause: str) -> None:
    """ACL / MCL: a named ligament plus a tear or sprain cue."""
    for label, anchor_re, generic_re in (
        ("ACL", _ACL_RE, _GEN_CRUCIATE_RE),
        ("MCL", _MCL_RE, _GEN_COLLATERAL_RE),
    ):
        spans = [m.span() for m in anchor_re.finditer(clause)]
        tear = _TEAR_RE.search(clause)
        sprain = _SPRAIN_RE.search(clause) or _INJURY_RE.search(clause)
        degen = _DEGEN_RE.search(clause)
        if spans:
            for span in spans:
                cue = tear or sprain or degen
                neg = _negated(clause, cue.span() if cue else None, span)
                if tear and not neg:
                    acc.positive(label, "tear:%s" % tear.group(0),
                                 hedged=bool(_HEDGE_RE.search(clause)))
                elif (sprain or degen) and not neg:
                    acc.weak_positive(label, "sprain/degen:%s" %
                                      (sprain or degen).group(0))
                elif neg:
                    acc.negative(label, "neg:%s" % neg)
            continue
        # Generic "cruciate and collateral ligaments are normal" -> negative only.
        for m in generic_re.finditer(clause):
            neg = _negated(clause, m.span())
            if neg and not tear:
                acc.negative(label, "generic-neg:%s" % neg)


def _meniscus_rules(acc: _Accumulator, clause: str) -> None:
    mentions = list(_iter_side_mentions(clause, _MENISCUS_RE))
    for m in _MED_MENISCUS_CPD_RE.finditer(clause):
        mentions.append(("medial", m.span()))
    for m in _LAT_MENISCUS_CPD_RE.finditer(clause):
        mentions.append(("lateral", m.span()))
    if not mentions:
        return
    tear = _TEAR_RE.search(clause)
    degen = _DEGEN_RE.search(clause)
    extr = _EXTRUSION_RE.search(clause)
    hedged = bool(_HEDGE_RE.search(clause))

    for side, span in mentions:
        cue = tear or degen or extr
        neg = _negated(clause, cue.span() if cue else None, span)
        targets = ({"medial": ["Medial Meniscus"],
                    "lateral": ["Lateral Meniscus"]}.get(
                        side, ["Medial Meniscus", "Lateral Meniscus"]))
        for label in targets:
            if tear and not neg:
                acc.positive(label, "tear:%s" % tear.group(0), hedged=hedged)
            elif (degen or extr) and not neg:
                acc.weak_positive(label, "degen:%s" % (degen or extr).group(0))
            elif neg:
                acc.negative(label, "neg:%s" % neg)


def _oa_rules(acc: _Accumulator, clause: str) -> None:
    path = _OA_PATH_RE.search(clause)
    anat = _OA_ANAT_RE.search(clause)
    damage = _OA_DAMAGE_RE.search(clause)
    if not (path or anat):
        return
    positive_evidence = path or (anat and damage)
    hedged = bool(_HEDGE_RE.search(clause))

    # (a) whole-joint statements
    tri = _TRICOMPARTMENTAL_RE.search(clause)
    if positive_evidence and tri:
        span = tri.span()
        if not _negated(clause, span):
            for label in ("Medial OA", "Lateral OA", "PF OA"):
                acc.positive(label, "tricompartmental", hedged=hedged)
            return

    # (b) patellofemoral
    _cue_span = (path or anat).span()
    for m in _PF_NOUN_RE.finditer(clause):
        neg = _negated(clause, _cue_span, m.span())
        if positive_evidence and not neg:
            acc.positive("PF OA", "pf:%s" % m.group(0), hedged=hedged)
            break
        if neg and anat:
            acc.negative("PF OA", "pf-neg:%s" % neg)
            break

    # (c) femorotibial compartments.  A side word only counts if it attaches to
    # a femorotibial noun; "medial patellar facet" is PF OA, not Medial OA.
    for side_re, label in ((_MEDIAL_RE, "Medial OA"), (_LATERAL_RE, "Lateral OA")):
        for m in side_re.finditer(clause):
            lo, hi = max(0, m.start() - 45), min(len(clause), m.end() + 45)
            ctx = clause[lo:hi]
            near_pf = _PF_NOUN_RE.search(clause[m.end():min(len(clause), m.end() + 22)])
            if near_pf:
                continue                       # "medial patellar facet"
            if not _FT_NOUN_RE.search(ctx):
                continue
            neg = _negated(clause, _cue_span, m.span())
            if positive_evidence and not neg:
                acc.positive(label, "ft:%s" % m.group(0), hedged=hedged)
                break
            if neg and anat:
                acc.negative(label, "ft-neg:%s" % neg)
                break

    # (d) unsided cartilage statements: "Cartilages normal." -> all three.
    if anat and not positive_evidence:
        neg = _negated(clause, anat.span())
        if neg and not _MEDIAL_RE.search(clause) and not _LATERAL_RE.search(clause) \
                and not _PF_NOUN_RE.search(clause):
            for label in ("Medial OA", "Lateral OA", "PF OA"):
                acc.negative(label, "cartilage-normal:%s" % neg)


def _presence_rules(acc: _Accumulator, clause: str) -> None:
    """Findings whose anchor IS the pathology: effusion, synovitis, cyst, ..."""
    hedged = bool(_HEDGE_RE.search(clause))

    for label, rx in (("Effusion", _EFFUSION_RE),
                      ("Synovitis", _SYNOVITIS_RE),
                      ("Baker's", _BAKER_RE)):
        for m in rx.finditer(clause):
            neg = _negated(clause, m.span())
            if neg:
                acc.negative(label, "neg:%s" % neg)
            else:
                acc.positive(label, "%s:%s" % (label, m.group(0)), hedged=hedged)
            break

    m = _CONTUSION_STRONG_RE.search(clause)
    if m:
        neg = _negated(clause, m.span())
        if neg:
            acc.negative("Contusion", "neg:%s" % neg)
        else:
            acc.positive("Contusion", "contusion:%s" % m.group(0), hedged=hedged)
    else:
        m = _CONTUSION_WEAK_RE.search(clause)
        if m:
            neg = _negated(clause, m.span())
            if neg:
                acc.negative("Contusion", "neg:%s" % neg)
            elif _CONTUSION_SUPPRESS_RE.search(clause):
                pass                       # degenerative marrow oedema, not trauma
            else:
                acc.weak_positive("Contusion", "marrow-oedema:%s" % m.group(0))

    m = _FRACTURE_RE.search(clause)
    if m:
        neg = _negated(clause, m.span())
        if neg:
            acc.negative("Fracture", "neg:%s" % neg)
        elif _FRACTURE_CONTEXT_SUPPRESS_RE.search(clause):
            acc.weak_positive("Fracture", "microtrabecular")
        else:
            acc.positive("Fracture", "fracture:%s" % m.group(0), hedged=hedged)

    # "Bones: Normal." / "Eklem kikirdaklari ve kemikler normal." -- a negated
    # bone statement argues against Fracture and Contusion, and nothing else.
    if not _FRACTURE_RE.search(clause) and not _CONTUSION_STRONG_RE.search(clause) \
            and not _CONTUSION_WEAK_RE.search(clause):
        b = _BONE_ANAT_RE.search(clause)
        if b and _negated(clause, b.span()):
            cue = _negated(clause, b.span())
            acc.negative("Fracture", "bones-normal:%s" % cue, weak=True)
            acc.negative("Contusion", "bones-normal:%s" % cue, weak=True)


def _global_normal_rule(acc: _Accumulator, folded: str) -> None:
    """A report that declares the whole study normal negates everything.

    Guarded on "no positive evidence anywhere": "Sonuc: Diz ekleminde minimal
    mayii artisi disindan normal sinirlarda bulgular" says normal AND names a
    finding, so it must not blanket-negate.
    """
    if not _GLOBAL_NORMAL_RE.search(folded):
        return
    if any(acc.pos[l] for l in KNEE_LABELS):
        return
    for label in KNEE_LABELS:
        if not acc.neg[label]:
            acc.add(label, W_GLOBAL_NORMAL, "global-normal")


def _cross_label_rules(acc: _Accumulator) -> None:
    """One physiological coupling, and only in the direction the data supports.

    Synovitis is the label reports mention least: on the 58 gold studies the
    rules make no call for it 41 times.  Measured over those 41 unmentioned
    rows:

        report explicitly denies effusion (n=5)  -> P(synovitis) = 0.00
        report asserts effusion         (n=30)  -> P(synovitis) = 0.40
        (the label's own prior is 0.41)

    So "no fluid in the joint" is real evidence against synovitis, while "there
    is fluid" tells you nothing you did not already know.  Only the negative
    direction is applied -- adding the positive one would manufacture 30
    confident targets out of no information.
    """
    if acc.pos["Synovitis"] or acc.neg["Synovitis"]:
        return
    if acc.neg["Effusion"] and not acc.pos["Effusion"]:
        acc.negative("Synovitis", "no-effusion", weak=True)


def _state_prob(label: str, state: str, ev_logit: float,
                calibration: Optional[Dict[str, Dict[str, float]]]) -> float:
    """Map a verdict to its measured reliability, nudged by evidence strength."""
    table = calibration if calibration is not None else LABEL_STATE_PROB
    base = table.get(label, {}).get(state)
    if base is None:
        base = STATE_RELIABILITY.get(state)
    if base is None:
        base = {"positive": P_POSITIVE, "weak_positive": P_WEAK_POSITIVE,
                "weak_negative": 0.5 * (P_NEGATIVE + UNMENTIONED_PRIOR),
                "negative": P_NEGATIVE}.get(state, UNMENTIONED_PRIOR)
    # The nudge is scaled by how close the band already is to 0 or 1.  A flat
    # +/-0.03 would swamp a band whose centre is 0.008 (ACL "negative") and
    # slam every such cell into the clip floor, destroying the ranking the AUC
    # is measured on.
    scale = min(base, 1.0 - base) / 0.5
    p = base + CAL_SPREAD * math.tanh(ev_logit / 2.0) * scale
    return float(min(max(p, 0.001), 0.999))


def extract_findings(text, priors: Optional[Dict[str, float]] = None,
                     calibration: Optional[Dict[str, Dict[str, float]]] = None
                     ) -> Dict[str, Finding]:
    """Run every rule over one report and return a Finding per label."""
    priors = priors if priors is not None else LABEL_PRIORS
    folded = normalize_text(text)
    acc = _Accumulator()
    for clause in split_clauses(folded):
        if not clause:
            continue
        _structure_rules(acc, clause)
        _meniscus_rules(acc, clause)
        _oa_rules(acc, clause)
        _presence_rules(acc, clause)

    _global_normal_rule(acc, folded)
    _cross_label_rules(acc)

    out: Dict[str, Finding] = {}
    for label in KNEE_LABELS:
        prior = float(priors.get(label, UNMENTIONED_PRIOR))
        pos, neg = acc.pos[label], acc.neg[label]
        if not pos and not neg:
            out[label] = Finding(label, prior, UNMENTIONED_CONFIDENCE,
                                 "unmentioned", 0.0, [])
            continue

        # Repeated mentions of the same finding (findings section + impression)
        # are one observation, not three, so extra hits only add a third of
        # their weight.
        pos_score = 0.0
        if pos:
            top = max(pos)
            pos_score = min(top + 0.3 * (sum(pos) - top), LOGIT_CLIP)
        neg_score = 0.0
        if neg:
            bot = min(neg)
            neg_score = max(bot + 0.3 * (sum(neg) - bot), -LOGIT_CLIP)
            if pos:
                neg_score *= NEG_DISCOUNT_WHEN_POS

        ev_logit = max(-LOGIT_CLIP, min(LOGIT_CLIP, pos_score + neg_score))

        if ev_logit >= W_POS - 1e-9:
            state = "positive"
        elif ev_logit > 0:
            state = "weak_positive"
        elif ev_logit <= W_NEG + 1e-9:
            state = "negative"
        elif ev_logit < 0:
            state = "weak_negative"
        else:
            # positive and negative evidence cancelled exactly
            out[label] = Finding(label, prior, UNMENTIONED_CONFIDENCE,
                                 "unmentioned", 0.0, acc.ev[label])
            continue

        prob = _state_prob(label, state, ev_logit, calibration)
        confidence = min(1.0, abs(ev_logit) / W_POS)
        out[label] = Finding(label, float(prob), float(confidence), state,
                             float(ev_logit), acc.ev[label])
    return out


def report_soft_targets(text, priors: Optional[Dict[str, float]] = None,
                        calibration: Optional[Dict[str, Dict[str, float]]] = None
                        ) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Convenience wrapper: ({label: prob}, {label: confidence})."""
    f = extract_findings(text, priors=priors, calibration=calibration)
    return ({k: v.prob for k, v in f.items()},
            {k: v.confidence for k, v in f.items()})


def report_scores(text, priors: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Just the probabilities -- what you rank by when computing AUC."""
    return {k: v.prob for k, v in extract_findings(text, priors=priors).items()}


# ===========================================================================
# 6. DataFrame helpers (pandas is imported lazily so the rules stay testable
#    without it)
# ===========================================================================

def label_dataframe(df, text_col: str = "Report",
                    labels: Optional[Sequence[str]] = None,
                    priors: Optional[Dict[str, float]] = None,
                    calibration: Optional[Dict[str, Dict[str, float]]] = None,
                    prefix: str = ""):
    """Return a frame of soft targets + confidences aligned with ``df``.

    Columns: ``<label>`` (soft target), ``<label>__conf``, ``<label>__state``,
    plus ``report_lang``.  ``prefix`` lets a caller namespace them.
    """
    import pandas as pd

    labels = list(labels) if labels is not None else list(KNEE_LABELS)
    rows = []
    for txt in df[text_col].fillna("") if text_col in df.columns else [""] * len(df):
        found = extract_findings(txt, priors=priors, calibration=calibration)
        row = {}
        for lab in labels:
            f = found.get(lab)
            if f is None:
                f = Finding(lab, UNMENTIONED_PRIOR, 0.0, "unmentioned", 0.0, [])
            row[prefix + lab] = f.prob
            row[prefix + lab + "__conf"] = f.confidence
            row[prefix + lab + "__state"] = f.state
            row[prefix + lab + "__evidence"] = "; ".join(f.evidence)
        row[prefix + "report_lang"] = detect_language(txt)
        rows.append(row)
    return pd.DataFrame(rows, index=df.index)


# ===========================================================================
# 7. Validation against the 58 gold studies
# ===========================================================================

def _auc(y_true: Sequence[float], y_score: Sequence[float]) -> float:
    """ROC-AUC by rank, ties averaged.  No sklearn dependency."""
    pairs = sorted(zip(y_score, y_true))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    pos = sum(1 for _, y in pairs if y == 1)
    negn = n - pos
    if pos == 0 or negn == 0:
        return float("nan")
    rank_sum = sum(r for r, (_, y) in zip(ranks, pairs) if y == 1)
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * negn)


def evaluate_against_gold(gold_df, labels: Optional[Sequence[str]] = None,
                          text_col: str = "Report",
                          priors: Optional[Dict[str, float]] = None,
                          calibration: Optional[Dict[str, Dict[str, float]]] = None):
    """Per-label agreement of the rules with the expert labels.

    ``gold_df`` must have ``text_col`` plus one column per label with 0/1.
    Returns a DataFrame indexed by label with: n, prevalence, coverage (fraction
    of studies where the rules made a call), precision / recall / F1 of
    "rules say positive", the AUC of the soft score, and the empirical
    P(y=1 | unmentioned) that calibrates ``LABEL_PRIORS``.
    """
    import pandas as pd

    labels = list(labels) if labels is not None else list(KNEE_LABELS)
    findings = [extract_findings(t, priors=priors, calibration=calibration)
                for t in gold_df[text_col].fillna("")]

    rows = []
    for lab in labels:
        y = gold_df[lab].astype(float).tolist()
        states = [f[lab].state for f in findings]
        probs = [f[lab].prob for f in findings]

        pred_pos = [s in ("positive", "weak_positive") for s in states]
        pred_neg = [s in ("negative", "weak_negative") for s in states]
        unmention = [s == "unmentioned" for s in states]

        tp = sum(1 for p, t in zip(pred_pos, y) if p and t == 1)
        fp = sum(1 for p, t in zip(pred_pos, y) if p and t == 0)
        fn = sum(1 for p, t in zip(pred_pos, y) if not p and t == 1)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * prec * rec / (prec + rec) if prec and rec and (prec + rec) else float("nan")

        # strict positives only (drop hedged calls) -- the precision that matters
        strict = [s == "positive" for s in states]
        stp = sum(1 for p, t in zip(strict, y) if p and t == 1)
        sfp = sum(1 for p, t in zip(strict, y) if p and t == 0)
        sprec = stp / (stp + sfp) if (stp + sfp) else float("nan")

        # negatives: how often is a rule-asserted negative actually negative?
        ntn = sum(1 for p, t in zip(pred_neg, y) if p and t == 0)
        nfn = sum(1 for p, t in zip(pred_neg, y) if p and t == 1)
        npv = ntn / (ntn + nfn) if (ntn + nfn) else float("nan")

        n_un = sum(unmention)
        p_un = (sum(t for u, t in zip(unmention, y) if u) / n_un) if n_un else float("nan")

        rows.append(dict(
            label=lab, n=len(y), prevalence=sum(y) / len(y),
            coverage=1.0 - n_un / len(y),
            n_pos_called=sum(pred_pos), precision=prec, recall=rec, f1=f1,
            strict_precision=sprec, n_neg_called=sum(pred_neg), npv=npv,
            n_unmentioned=n_un, p_pos_given_unmentioned=p_un,
            auc=_auc(y, probs),
        ))
    return pd.DataFrame(rows).set_index("label")


def _restrict_to_folds(gold_df, folds):
    """Subset gold_df to the given fold ids (train.py writes a `fold` column).

    Pass ``folds`` to keep the calibration out of the fold you validate on.
    """
    if folds is None or "fold" not in getattr(gold_df, "columns", []):
        return gold_df
    keep = set(folds if hasattr(folds, "__iter__") else [folds])
    return gold_df[gold_df["fold"].isin(keep)]


def calibrate_priors(gold_df, labels: Optional[Sequence[str]] = None,
                     text_col: str = "Report",
                     pseudo_count: float = 8.0,
                     fallback: float = UNMENTIONED_PRIOR,
                     folds=None) -> Dict[str, float]:
    """Estimate P(y=1 | rules found no mention) per label, Beta-shrunk.

    With 58 gold studies a label may have only a handful of "unmentioned" rows,
    so the raw fraction is unusable on its own; ``pseudo_count`` pulls it back
    towards ``fallback``.  ``folds`` restricts the fit to those gold folds.
    """
    gold_df = _restrict_to_folds(gold_df, folds)
    labels = list(labels) if labels is not None else list(KNEE_LABELS)
    findings = [extract_findings(t) for t in gold_df[text_col].fillna("")]
    out = {}
    for lab in labels:
        y = gold_df[lab].astype(float).tolist()
        un = [i for i, f in enumerate(findings) if f[lab].state == "unmentioned"]
        k = sum(y[i] for i in un)
        n = len(un)
        out[lab] = round((k + pseudo_count * fallback) / (n + pseudo_count), 3)
    return out


def calibrate_states(gold_df, labels: Optional[Sequence[str]] = None,
                     text_col: str = "Report",
                     pseudo_count: float = 10.0,
                     priors: Optional[Dict[str, float]] = None,
                     folds=None) -> Dict[str, Dict[str, float]]:
    """Refit LABEL_STATE_PROB: P(y=1 | verdict) per label, Beta-shrunk + monotone.

    Use with ``folds=`` to refit on training folds only, if you would rather
    the shipped constants carried no trace of the fold you validate on.
    """
    gold_df = _restrict_to_folds(gold_df, folds)
    labels = list(labels) if labels is not None else list(KNEE_LABELS)
    priors = priors if priors is not None else LABEL_PRIORS
    findings = [extract_findings(t) for t in gold_df[text_col].fillna("")]

    table: Dict[str, Dict[str, float]] = {}
    for lab in labels:
        y = gold_df[lab].astype(float).tolist()
        row = {}
        for state, target in STATE_RELIABILITY.items():
            idx = [i for i, f in enumerate(findings) if f[lab].state == state]
            k = sum(y[i] for i in idx)
            row[state] = (k + pseudo_count * target) / (len(idx) + pseudo_count)
        # Enforce STRICT negative < weak_negative < prior < weak_positive <
        # positive.  Strict, not merely non-decreasing: if a weak_negative
        # landed exactly on the prior, "the report denies it" and "the report
        # never mentions it" would be the same target, and the rule that
        # produced the denial would be doing no work.
        prior = float(priors.get(lab, UNMENTIONED_PRIOR))
        row["weak_negative"] = min(row["weak_negative"], prior * 0.9)
        row["negative"] = min(row["negative"], row["weak_negative"] * 0.9)
        row["weak_positive"] = max(row["weak_positive"], prior * 1.1 + 0.02)
        row["positive"] = max(row["positive"], row["weak_positive"] * 1.05)
        table[lab] = {k: round(v, 3) for k, v in row.items()}
    return table


def write_label_csv(source_df, out_path: str, text_col: str = "Report",
                    labels: Optional[Sequence[str]] = None,
                    priors: Optional[Dict[str, float]] = None,
                    calibration: Optional[Dict[str, Dict[str, float]]] = None,
                    image_dir: Optional[str] = None,
                    exclude_ids: Optional[Sequence[str]] = None) -> str:
    """Materialise the soft targets as a CSV.

    Schema, which is what ``train.py``'s ``load_derived_labels`` /
    ``derived_cell_weights`` read:

        StudyInstanceUID, <label> ... , <label>__conf ... ,
        <label>__state ... , report_lang

    ``<label>``       calibrated soft target in (0, 1)
    ``<label>__conf`` per-cell confidence in [0, 1]; 0 means "the report does
                      not speak to this finding", so the cell costs the loss
                      nothing while its target still carries the prior.

    Point ``cfg.labels_csv`` at the result.  Nothing under data_subset/ is
    touched -- ``out_path`` is wherever the caller says.
    """
    import pandas as pd

    df = source_df
    df = df[df[text_col].fillna("").astype(str).str.strip() != ""].copy()
    df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
    if image_dir and os.path.isdir(image_dir):
        on_disk = {d for d in os.listdir(image_dir) if not d.startswith(".")}
        df = df[df["StudyInstanceUID"].isin(on_disk)]
    if exclude_ids:
        df = df[~df["StudyInstanceUID"].isin({str(s) for s in exclude_ids})]

    labels = list(labels) if labels is not None else list(KNEE_LABELS)
    out = label_dataframe(df, text_col=text_col, labels=labels,
                          priors=priors, calibration=calibration)
    out.insert(0, "StudyInstanceUID", df["StudyInstanceUID"].values)
    out = out.drop(columns=[c for c in out.columns if c.endswith("__evidence")])
    d = os.path.dirname(os.path.abspath(out_path))
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":  # pragma: no cover - developer harness
    import argparse as _argparse
    import csv as _csv

    _csv.field_size_limit(10 ** 9)
    import pandas as pd

    ap = _argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=None,
                    help="defaults to <repo>/data_subset")
    ap.add_argument("--write", default=None, metavar="PATH",
                    help="also write the soft-target CSV here (point "
                         "cfg.labels_csv at it). Nothing is written by default, "
                         "and never into data_subset/ unless you say so.")
    ap.add_argument("--images-only", action="store_true",
                    help="with --write, keep only studies that have pixels on disk")
    ap.add_argument("--calibrate-on-folds", default="",
                    help="refit the calibration on these gold folds, e.g. 0,1,2,3")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = args.data_dir or os.path.join(root, "data_subset")
    gold = pd.read_csv(os.path.join(data, "train_gold.csv"), engine="python")
    train = pd.read_csv(os.path.join(data, "train.csv"), engine="python")

    folds = [int(x) for x in args.calibrate_on_folds.replace(" ", "").split(",")
             if x != ""] or None
    priors = calibrate_priors(gold, folds=folds) if folds else None
    calib = calibrate_states(gold, priors=priors, folds=folds) if folds else None

    print("language distribution (all %d reports):" % len(train))
    print(train["Report"].map(detect_language).value_counts().to_string())

    print("\nper-label agreement vs the 58 gold studies:")
    with pd.option_context("display.width", 220, "display.max_columns", 50):
        rep = evaluate_against_gold(gold, priors=priors, calibration=calib)
        print(rep.round(3).to_string())
    print("\nMACRO ROC-AUC = %.4f" % rep["auc"].mean())

    print("\nsuggested LABEL_PRIORS =", calibrate_priors(gold, folds=folds))
    print("suggested LABEL_STATE_PROB =", calibrate_states(gold, folds=folds))

    if args.write:
        img = os.path.join(data, "train_images") if args.images_only else None
        p = write_label_csv(train, args.write, priors=priors, calibration=calib,
                            image_dir=img)
        print("\nwrote soft targets to", p)
