"""Derive the 12 knee targets from the free-text radiology report.

WHY
---
`train_gold.csv` carries 58 labelled studies. `train.csv` carries a report for
every one of the 649 studies whose DICOMs are on disk. The labels and the
reports describe the same findings, so the reports are 11x more supervision
sitting unused — and supervision, not architecture, is what limits this project.

WHAT THIS IS NOT
----------------
It is not a claim that parsed labels equal gold labels. It is a *measurable*
weak-supervision source: `evaluate()` scores every rule against the 58 gold
studies and reports per-label precision/recall, so the noise is quantified
before any of it reaches a loss function.

Uncertain cells are emitted as **NaN**, never as 0. The training pipeline uses
NaN-masked losses, so a NaN cell contributes exactly zero gradient — an
unparsed language or an unmentioned finding costs nothing, while a confidently
parsed one teaches. This is why the module can afford to cover some languages
well and abstain on the rest.

LANGUAGE COVERAGE
-----------------
The corpus is international: English 261, Turkish 128, Spanish 94, Greek 46,
Cyrillic 37, other-Latin 83 (of 649 on-disk studies). Rules below cover English
and Spanish in full, and use Latin/Greek/Cyrillic cognate stems for the rest
where the term is unambiguous. Anything not covered returns an all-NaN row.

NEGATION
--------
Radiology prose is dominated by explicit negation ("no evidence of a tear",
"sin signos de rotura"), so a keyword hit without negation handling is worse
than useless. Each finding is scored inside its own clause, and a negation cue
anywhere before the term in that clause flips a hit to 0.
"""

import re
import unicodedata

import numpy as np
import pandas as pd

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
           "Medial OA", "Lateral OA", "PF OA",
           "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# ── Language identification ────────────────────────────────────────────────
# Script first (Greek/Cyrillic are unambiguous), then keyword voting inside
# Latin. Only 'en' and 'es' are treated as fully covered.
_LATIN_VOTES = {
    "es": r"\b(menisco|rodilla|hallazgos|derrame|se[nñ]al|sin\s+signos|"
          r"articular|c[oó]ndilo|cart[ií]lago|rotura)\b",
    "en": r"\b(knee|meniscus|effusion|there\s+is|joint|tear|no\s+evidence|"
          r"findings|impression|cartilage)\b",
    "pt": r"\b(joelho|derrame|n[aã]o\s+h[aá]|ligamento\s+cruzado|menisco\s+medial)\b",
    "tr": r"\b(diz|menisk|eklem|y[ıi]rt[ıi]k|izlenmektedir|kemik)\b",
    # Dutch and German are close enough to English on short tokens that they
    # used to win the vote and get parsed with English rules.  A Dutch report
    # was scoring Lateral OA as a false negative purely because of that, so
    # they are listed as explicit competitors even though neither is covered.
    "nl": r"\b(kraakbeen|bevindingen|gewricht|knie|mediale|laterale|"
          r"meniscus\s+met|voorkomen)\b",
    "de": r"\b(kniegelenk|knorpel|befund|beurteilung|innenmeniskus|"
          r"aussenmeniskus|kein[e]?\s)\b",
}
FULLY_COVERED = ("en", "es")


def detect_language(text: str) -> str:
    s = str(text or "")
    counts = {}
    for ch in s[:2000]:
        if ch.isalpha():
            n = unicodedata.name(ch, "")
            k = ("greek" if "GREEK" in n else "cyrillic" if "CYRILLIC" in n
                 else "arabic" if "ARABIC" in n else "latin")
            counts[k] = counts.get(k, 0) + 1
    if not counts:
        return "none"
    script = max(counts, key=counts.get)
    if script != "latin":
        return script
    low = s.lower()
    votes = {k: len(re.findall(p, low)) for k, p in _LATIN_VOTES.items()}
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    best, best_n = ranked[0]
    runner_n = ranked[1][1] if len(ranked) > 1 else 0
    # Require a clear win.  A tie means the vocabulary is ambiguous between two
    # languages, and guessing there is how a Dutch report gets parsed with
    # English rules and silently contributes wrong labels.
    if best_n == 0 or best_n == runner_n:
        return "latin-other"
    return best


# ── Negation cues, per language family ─────────────────────────────────────
_NEG = (r"\b(no|not|without|absent|negative\s+for|unremarkable|intact|normal|"
        r"rule[sd]?\s+out|free\s+of|"
        r"sin|no\s+hay|no\s+se|ausencia|aus[eé]ncia|conservad[oa]s?|normales?|"
        r"[ıi]zlenmemektedir|yoktur|yok|"
        r"δεν|χωρ[ίι]ς|"
        r"не|няма|без)\b")
_NEG_RE = re.compile(_NEG, re.I)

# Clause splitter: negation scope in radiology prose rarely crosses these.
_CLAUSE_RE = re.compile(r"[.;:!?\n\r]|(?:\s[-–—>•]\s)|\bbut\b|\bpero\b|\bhowever\b", re.I)

# ── Finding patterns ───────────────────────────────────────────────────────
# Each target maps to (anatomy regex, pathology regex). A positive needs BOTH
# in the same clause, un-negated. Cognate stems cover non-en/es scripts.
_MENISCUS = r"menisc|men[ií]sc|menisk|μηνίσκ|µηνίσκ|мениск"
_TEAR = (r"tear|torn|rupture|ruptur|rotura|roto|desgarr|fissur|fisura|"
         r"y[ıi]rt|ρήξη|ρήξ|руптур|разкъс|maceration|degenerative\s+signal|"
         r"grade\s*(?:2|3|III|II)\b|amputaci[oó]n")
_MEDIAL = r"medial|interno|internal|έσω|медиал|i[cç]\s|mediale"
_LATERAL = r"lateral|externo|external|έξω|латерал"
_COMPARTMENT = (r"compartment|comparti|femorotibial|femorotibiaal|tibial|femoral|"
                r"condyle|c[oó]ndilo|plateau|platillo|meseta|"
                r"κνημ|κονδυλ|тибиал|бедрен")
_OA = (r"osteoarthr|arthros|arthrosis|artrosis|osteo?fit|osteophyt|osteofyt|"
       r"chondropath|condropat|"
       # cartilage damage, either word order and either language
       r"chondral\s+(?:loss|thinning|defect|ulcer|fissur|damage)|"
       r"cartilage\s+(?:loss|thinning|defect|fissur|damage|wear)|"
       r"osteochondral\s+(?:defect|lesion)|full[\s-]thickness\s+cartilage|"
       r"(?:high|low)[\s-]grade\s+cartilage|subchondral\s+(?:bone\s+)?(?:edema|cyst|sclero)|"
       r"[úu]lcera\s+condral|condral\s+focal|p[eé]rdida\s+de\s+cart[ií]lago|"
       r"adelgazamiento|degenerative\s+change|degenerativ|"
       r"kraakbeen(?:lijden|verlies)|knorpel|"
       r"οστεοφ|χόνδρ|остеофит|хрущял|"
       r"ICRS\s*Grade\s*(?:III|IV|3|4)|grade\s*(?:3|4|III|IV)\s+chondr")

_PATTERNS = {
    "ACL": (r"\bACL\b|anterior\s+cruciate|cruzado\s+anterior|\bLCA\b|"
            r"πρόσθι\w*\s+χιαστ|предна\s+кръстна", _TEAR),
    "MCL": (r"\bMCL\b|medial\s+collateral|colateral\s+(?:medial|interno)|\bLCM\b|"
            r"έσω\s+πλάγι|медиален\s+колатерал", _TEAR + r"|sprain|esguince|edema|injury|lesi[oó]n"),
    "Medial Meniscus": (rf"(?:{_MEDIAL})[\w\s]{{0,18}}(?:{_MENISCUS})|"
                        rf"(?:{_MENISCUS})[\w\s]{{0,18}}(?:{_MEDIAL})", _TEAR),
    "Lateral Meniscus": (rf"(?:{_LATERAL})[\w\s]{{0,18}}(?:{_MENISCUS})|"
                         rf"(?:{_MENISCUS})[\w\s]{{0,18}}(?:{_LATERAL})", _TEAR),
    # "lateral femoral condyle" and "cóndilo femoral lateral" are the same
    # finding; the side word appears before the anatomy in English and after it
    # in Spanish, so both orders have to match.
    "Medial OA": (rf"(?:{_MEDIAL})[\w\s]{{0,25}}(?:{_COMPARTMENT})|"
                  rf"(?:{_COMPARTMENT})[\w\s]{{0,25}}(?:{_MEDIAL})", _OA),
    "Lateral OA": (rf"(?:{_LATERAL})[\w\s]{{0,25}}(?:{_COMPARTMENT})|"
                   rf"(?:{_COMPARTMENT})[\w\s]{{0,25}}(?:{_LATERAL})", _OA),
    "PF OA": (r"patellofemoral|femoropatelar|patelofemoral|femoro-?patellar|"
              r"trochlea|tr[oó]clea|patellar\s+cartilage|cart[ií]lago\s+rotulian|"
              r"retropatellar|επιγονατιδομηρια|пателофемор|patella\w*\s+chondr", _OA),
    # Single-concept findings: the term itself is the finding.
    "Effusion": (r"effusion|derrame|joint\s+fluid|fluid\s+(?:accumulat|collect)|"
                 r"hydrarthros|hidrartros|s[ıi]v[ıi]\s+art[ıi]|αρθρικ\w*\s+υγρ|"
                 r"излив|синовиал\w*\s+течност", None),
    "Synovitis": (r"synovit|sinovit|synovial\s+(?:thicken|proliferat|hypertroph)|"
                  r"engrosamiento\s+sinovial|sinovyal|υμενίτ|синовит|hoffit", None),
    "Baker's": (r"baker|popliteal\s+cyst|quiste\s+popl[ií]te|poplit[ée]al?\s+cyst|"
                r"gastrocnemio-?semimembranos|popliteal\s+bursa|"
                r"κύστη\s+Baker|подколянна\s+киста|popliteal\s+kist", None),
    "Contusion": (r"contusion|contusi[oó]n|bone\s+(?:marrow\s+)?(?:edema|oedema|bruise)|"
                  r"edema\s+(?:[oó]seo|de\s+m[eé]dula|medular)|bone\s+marrow\s+signal|"
                  r"kemik\s+[oö]dem|οστεομυελικ\w*\s+οίδημα|οστεομυελικ|"
                  r"костномозъчен\s+едем|контузион", None),
    "Fracture": (r"fracture|fractura|fissure\s+fracture|k[ıi]r[ıi]k|"
                 r"κάταγμα|κάταγµα|фрактур|счупван|avulsion|arrancamiento", None),
}
_COMPILED = {k: (re.compile(a, re.I), re.compile(p, re.I) if p else None)
             for k, (a, p) in _PATTERNS.items()}


def _clauses(text):
    return [c for c in _CLAUSE_RE.split(str(text or "")) if c and c.strip()]


def _negated(clause, upto):
    """True if a negation cue appears before position `upto` in this clause."""
    return bool(_NEG_RE.search(clause[:upto]))


def label_report(text):
    """Return a dict target -> 1.0 / 0.0 / nan for one report.

    A finding is 1.0 when its anatomy and pathology terms co-occur in one clause
    with no preceding negation cue. It is 0.0 when the language is covered and
    no such clause exists (radiology reports are exhaustive by convention, so an
    unmentioned finding is an absent finding). It is NaN when the language is
    not covered, so the cell is masked out of the loss instead of guessed.
    """
    lang = detect_language(text)
    if lang not in FULLY_COVERED:
        return {t: np.nan for t in TARGETS}, lang

    out = {t: 0.0 for t in TARGETS}
    for clause in _clauses(text):
        for target, (anat_re, path_re) in _COMPILED.items():
            if out[target] == 1.0:
                continue
            m = anat_re.search(clause)
            if not m:
                continue
            if path_re is None:
                if not _negated(clause, m.start()):
                    out[target] = 1.0
                continue
            pm = path_re.search(clause)
            if pm and not _negated(clause, min(m.start(), pm.start())):
                out[target] = 1.0
    return out, lang


def label_frame(df, report_col="Report"):
    """Vectorised `label_report` over a dataframe. Adds a `report_lang` column."""
    rows, langs = [], []
    for text in df[report_col].astype(str):
        r, lang = label_report(text)
        rows.append(r)
        langs.append(lang)
    out = pd.DataFrame(rows, index=df.index)[TARGETS]
    out["report_lang"] = langs
    return out


# ══════════════════════════════════════════════════════════════════════════

def evaluate(gold_csv="data_subset/train_gold.csv"):
    """Score every rule against the 58 gold studies. Prints per-label PR."""
    g = pd.read_csv(gold_csv)
    pred = label_frame(g)
    covered = pred["report_lang"].isin(FULLY_COVERED)
    print(f"Gold studies: {len(g)} | language-covered: {int(covered.sum())} "
          f"({100 * covered.mean():.0f}%)")
    print(f"Language mix: {pred['report_lang'].value_counts().to_dict()}\n")

    sub_g, sub_p = g[covered], pred[covered]
    print(f"{'label':18} {'n_pos':>5} {'TP':>3} {'FP':>3} {'FN':>3} "
          f"{'prec':>6} {'rec':>6} {'agree':>6}")
    print("-" * 62)
    tot = np.zeros(3)
    for t in TARGETS:
        y = sub_g[t].values.astype(float)
        p = sub_p[t].values.astype(float)
        keep = np.isfinite(y) & np.isfinite(p)
        y, p = y[keep], p[keep]
        tp = float(((y == 1) & (p == 1)).sum())
        fp = float(((y == 0) & (p == 1)).sum())
        fn = float(((y == 1) & (p == 0)).sum())
        tot += [tp, fp, fn]
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        agree = float((y == p).mean()) if y.size else float("nan")
        print(f"{t:18} {int(y.sum()):5d} {int(tp):3d} {int(fp):3d} {int(fn):3d} "
              f"{prec:6.2f} {rec:6.2f} {agree:6.2f}")
    tp, fp, fn = tot
    print("-" * 62)
    print(f"{'MICRO':18} {'':5} {int(tp):3d} {int(fp):3d} {int(fn):3d} "
          f"{tp/(tp+fp) if tp+fp else float('nan'):6.2f} "
          f"{tp/(tp+fn) if tp+fn else float('nan'):6.2f}")
    return pred


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--gold", default="data_subset/train_gold.csv")
    ap.add_argument("--train-csv", default="data_subset/train.csv")
    ap.add_argument("--image-dir", default="data_subset/train_images")
    ap.add_argument("--out", default=None,
                    help="Write a weak-label CSV for every study with images.")
    a = ap.parse_args()

    if a.evaluate or not a.out:
        evaluate(a.gold)

    if a.out:
        import os
        tr = pd.read_csv(a.train_csv)
        on_disk = {d for d in os.listdir(a.image_dir) if not d.startswith(".")}
        tr = tr[tr["StudyInstanceUID"].astype(str).isin(on_disk)].reset_index(drop=True)
        pred = label_frame(tr)
        out = pd.concat([tr[["StudyInstanceUID"]], pred], axis=1)
        # Gold always overrides a parsed label for the studies that have one.
        gold = pd.read_csv(a.gold)
        gcols = [c for c in TARGETS if c in gold.columns]
        gmap = gold.set_index(gold["StudyInstanceUID"].astype(str))[gcols]
        n_over = 0
        for i, sid in enumerate(out["StudyInstanceUID"].astype(str)):
            if sid in gmap.index:
                out.loc[i, gcols] = gmap.loc[sid, gcols].values
                n_over += 1
        out.to_csv(a.out, index=False)
        usable = out[TARGETS].notna().any(axis=1).sum()
        print(f"\nWrote {a.out}: {len(out)} studies, {usable} with at least one "
              f"usable label ({n_over} overridden by gold).")
