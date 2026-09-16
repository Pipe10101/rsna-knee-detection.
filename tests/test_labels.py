"""Tests for src/labels.py — weak supervision from multilingual reports.

Run:  python3 -m tests.test_labels          (from the repo root)
      python3 -m pytest tests/test_labels.py

Plain asserts, no pytest required, so this also runs on the Kaggle image.
Tests that need data_subset/ skip loudly when it is absent.

Every non-English string below is copied verbatim from data_subset/train.csv.
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.labels as L                                        # noqa: E402

PASSED, FAILED, SKIPPED = [], [], []


class _Skip(Exception):
    pass


def check(name, fn):
    try:
        fn()
        PASSED.append(name)
        print(f"  PASS  {name}")
    except _Skip as e:
        SKIPPED.append((name, str(e)))
        print(f"  SKIP  {name}: {e}")
    except AssertionError as e:
        FAILED.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")
    except Exception as e:                                    # noqa: BLE001
        FAILED.append((name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}: {type(e).__name__}: {e}")


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO, "data_subset")


def state(text, label):
    return L.extract_findings(text)[label].state


def prob(text, label):
    return L.extract_findings(text)[label].prob


def is_pos(text, label):
    return state(text, label) in ("positive", "weak_positive")


def is_neg(text, label):
    return state(text, label) in ("negative", "weak_negative")


# ══════════════════════════════════════════════════════════════════════════
# 1. Normalisation
# ══════════════════════════════════════════════════════════════════════════

def test_accent_variants_unify():
    """'Sin anomalias' and 'Sin anomalias' both occur in the corpus and must
    normalise to the same string, or the second spelling silently loses its
    global-normal rule."""
    assert L.normalize_text("Sin anomalías") == L.normalize_text("Sin anomalias")
    assert L.normalize_text("Impresión") == L.normalize_text("Impresion")


def test_turkish_and_croatian_letters_fold_to_ascii():
    assert L.normalize_text("Ön çapraz bağ") == "on capraz bag"
    assert L.normalize_text("MENİSKÜS") == "meniskus"
    assert L.normalize_text("Yırtık") == "yirtik"
    assert L.normalize_text("križni") == "krizni"
    assert L.normalize_text("Koštani") == "kostani"


def test_german_sharp_s_expands():
    assert "aussenmenisk" in L.normalize_text("Außenmeniskushinterhornwurzel")


def test_greek_micro_sign_folds_to_mu_and_accents_are_stripped():
    """The Greek reports in this corpus use MICRO SIGN U+00B5 for mu, so a
    pattern written with real Greek mu would never fire without NFKD."""
    assert L.normalize_text("µηνίσκου") == L.normalize_text("μηνισκου")
    assert L.normalize_text("έσω") == "εσω"
    assert L.normalize_text("ΡΉΞΗ") == "ρηξη"


def test_final_sigma_normalises():
    assert L.normalize_text("χιαστός") == L.normalize_text("χιαστοσ")


# ══════════════════════════════════════════════════════════════════════════
# 2. Language identification
# ══════════════════════════════════════════════════════════════════════════

SAMPLES = {
    "en": "FINDINGS: No joint effusion. No popliteal cyst. The medial meniscus is normal.",
    "es": "Técnica: RMN de la rodilla. Resultados: Rotura de menisco interno. Impresión: Rotura de menisco interno.",
    "tr": "Diz eklemi içi sıvı miktarı normal. Çapraz ve yan bağlar normal. Medyal ve lateral menisküs normal.",
    "hr": "Lateralni menisk bez znakova rupture. Prednji križni ligament održanog kontinuiteta. Manja količina izljeva u zglob.",
    "de": "VKB, HKB, MCL und LCL intakt und physiologisch signalgebend. Geringer Gelenkerguss. Baker-Zyste.",
    "nl": "Bevindingen: Geen hydrops. Normale quadricepspees. Intacte voorste kruisband. Normaal voorkomen menisci.",
    "fr": "Il y a un minime épanchement articulaire. Il n'y a pas de kyste poplité. Ligament croisé antérieur : Normal.",
    "el": "ΕΥΡΗΜΑΤΑ Χωρίς ενδαρθρική συλλογή υγρού. Εντός του φυσιολογικού ελέγχονται οι µηνίσκοι.",
    "bg": "МР находка: Няма данни за ставен излив. Нормално представени менискуси. Ставният хрущял е интактен.",
}


def test_language_detection_covers_all_nine():
    for want, text in SAMPLES.items():
        got = L.detect_language(text)
        assert got == want, f"{want!r} sample detected as {got!r}: {text[:60]!r}"


def test_language_detection_handles_empty_and_none():
    assert L.detect_language("") == "unknown"
    assert L.detect_language(None) == "unknown"
    assert L.detect_language("   \n ") == "unknown"


# ══════════════════════════════════════════════════════════════════════════
# 3. Multilingual POSITIVES
# ══════════════════════════════════════════════════════════════════════════

MULTILINGUAL_POSITIVES = [
    # (label, language, text)
    ("ACL", "en", "There is complete tear of the anterior cruciate ligament, at mid substance."),
    ("ACL", "es", "Rotura espesor y ancho total de la unión del tercio medio del LCA."),
    ("ACL", "tr", "Anterior çapraz bağda komplet rüptüre sekonder bütünlük kaybı izlenmiştir."),
    ("ACL", "hr", "Potpuna ruptura prednjeg križnog ligamenta."),
    ("ACL", "de", "Anamnestisch bek. VKB Ruptur mit wohl ligamentärer Ruptur des anteromedialen Bündels."),
    ("ACL", "nl", "Volledige scheur van de voorste kruisband, onregelmatig voorkomend, discontinu."),
    ("ACL", "el", "Οιδηµατώδης απεικόνιση του πρόσθιου χιαστού συνδέσµου µε ασυνέχεια των ινών του στο πλαίσιο ρήξης."),
    ("ACL", "bg", "МР данни за руптура на предната кръстна връзка."),

    ("MCL", "en", "Medial collateral ligament (MCL): Grade 2 injury."),
    ("MCL", "es", "Rotura parcial del LCM."),
    ("MCL", "tr", "Medial kollateral ligamanda komplet rüptür izlenmiştir."),
    ("MCL", "de", "Komplette Ruptur des medialen Kollateralbandes."),

    ("Medial Meniscus", "en", "There is longitudinal vertical tear at the body and posterior horn of the medial meniscus."),
    ("Medial Meniscus", "es", "Rotura de menisco interno."),
    ("Medial Meniscus", "tr", "Medial menisküs posterior hornda radial yırtık hattı izlenmiştir."),
    ("Medial Meniscus", "hr", "Kompleksna ruptura stražnjeg roga medijalnog meniska."),
    ("Medial Meniscus", "de", "Innenmeniskus mit horizontalem Riss Pars intermedia bis ins Hinterhorn ziehend."),
    ("Medial Meniscus", "bg", "МР данни за руптура на задния рог на медиалния менискус."),

    ("Lateral Meniscus", "en", "Horizontal tear at anterior horn of the lateral meniscus is noted."),
    ("Lateral Meniscus", "es", "Rotura del cuerno anterior del menisco lateral con mínima extrusión."),
    ("Lateral Meniscus", "hr", "Vidi se radijalna ruptura prednjeg roga lateralnog meniska."),

    ("PF OA", "es", "Condropatía rotuliana."),
    ("PF OA", "hr", "Hondromalacija III° obje fasete patele i trohleje femura."),
    ("PF OA", "de", "Retropatellar ausgedehnte Chondropathie."),

    ("Medial OA", "en", "Full thickness cartilage loss along the medial femoral condyle and medial tibial plateau."),
    ("Medial OA", "es", "Artrosis femorotibial medial."),

    ("Effusion", "en", "Small joint effusion."),
    ("Effusion", "es", "Leve derrame articular."),
    ("Effusion", "tr", "Diz eklemi içi sıvı miktarı hafif artmış."),
    ("Effusion", "hr", "Manja količina izljeva u zglob."),
    ("Effusion", "de", "Geringer Gelenkerguss."),
    ("Effusion", "nl", "Matige hydrops."),
    ("Effusion", "el", "Συλλογή υγρού παρατηρείται ενδαρθρικά."),
    ("Effusion", "bg", "МР данни за ставен излив."),

    ("Synovitis", "en", "Small joint effusion with synovial thickening compatible with synovitis."),
    ("Synovitis", "es", "Derrame con sinovitis."),
    ("Synovitis", "tr", "Diz ekleminde hafif sinovit."),
    ("Synovitis", "de", "Reizsynovialitis."),

    ("Baker's", "en", "There is popliteal cyst measuring 21 x 17 x 35 mm."),
    ("Baker's", "es", "Quiste de Baker."),
    ("Baker's", "tr", "Popliteal fossada 54x15 mm çapta Baker kisti ile uyumlu görünüm mevcuttur."),
    ("Baker's", "hr", "Bakerova cista dimenzija 25x29x53 mm."),
    ("Baker's", "de", "Baker-Zyste; diese bis 2 x 1,5 x CC 3,5 cm Ausdehnung."),
    ("Baker's", "bg", "Наличие на малка еднокамерна Бекерова киста в поплитеалната ямка."),

    ("Contusion", "en", "There is mild residual bone contusion in the weightbearing aspect of the lateral femoral condyle."),
    ("Contusion", "es", "Contusiones óseas femorotibiales."),
    ("Contusion", "tr", "Tibyal plato posterolateral köşesinde kemik kontüzyonu mevcut."),
    ("Contusion", "bg", "Контузионен костно-мозъчен едем се визуализира и в латералния феморален кондил."),

    ("Fracture", "en", "Subchondral insufficiency fracture at the medial margin of the medial tibial plateau."),
    ("Fracture", "es", "Pequeña fractura subcortical regional."),
    ("Fracture", "tr", "Femur lateral kondilde subartiküler nondeplase fraktür izlenmiştir."),
    ("Fracture", "hr", "Prikazuje se prijelom lateralnog tibijalnog platoa."),
    ("Fracture", "nl", "Impactiefractuur op centraal dragende deel van de laterale femorale condylen."),
    ("Fracture", "el", "Μικροδοκιδώδη κατάγµατα ανεπάρκειας έσω κνηµιαίου κονδύλου."),
    ("Fracture", "bg", "МР данни за фрактура на тибията в областта на интеркондиларните еминенции."),
]


def test_multilingual_positives():
    bad = []
    for label, lang, text in MULTILINGUAL_POSITIVES:
        if not is_pos(text, label):
            bad.append(f"{label}/{lang}: state={state(text, label)} :: {text[:70]}")
    assert not bad, "positives not detected:\n    " + "\n    ".join(bad)


def test_positive_soft_target_exceeds_the_prior():
    """A soft target only helps if a detected finding outranks an undetected
    one; that ordering is the whole basis of the AUC number."""
    for label, _lang, text in MULTILINGUAL_POSITIVES:
        assert prob(text, label) > L.LABEL_PRIORS[label], \
            f"{label}: {prob(text, label):.3f} <= prior {L.LABEL_PRIORS[label]}"


# ══════════════════════════════════════════════════════════════════════════
# 4. Multilingual NEGATIVES / negation handling
# ══════════════════════════════════════════════════════════════════════════

MULTILINGUAL_NEGATIVES = [
    # English: negator precedes
    ("Effusion", "en", "No joint effusion. No popliteal cyst."),
    ("Baker's", "en", "No joint effusion. No popliteal cyst."),
    ("Medial Meniscus", "en", "In the medial compartment, the meniscus is not torn."),
    ("ACL", "en", "Anterior cruciate ligament (ACL): Normal with both bundles intact."),
    ("Fracture", "en", "No acute fracture or bone bruise."),
    ("Contusion", "en", "No acute fracture or bone bruise."),

    # Spanish
    ("Medial Meniscus", "es", "Menisco medial de morfología y señal conservada, sin signos de rotura."),
    ("ACL", "es", "Ligamentos cruzados y colaterales dentro de límites normales."),
    ("MCL", "es", "Ligamentos cruzados y colaterales dentro de límites normales."),
    ("Baker's", "es", "No hay quistes poplíteos."),

    # Turkish: the negator FOLLOWS the finding
    ("Effusion", "tr", "Diz eklemi içi sıvı miktarı normal."),
    ("ACL", "tr", "Çapraz ve yan bağlar normal."),
    ("MCL", "tr", "Çapraz ve yan bağlar normal."),
    ("Medial Meniscus", "tr", "Medyal ve lateral menisküs normal."),
    ("Lateral Meniscus", "tr", "Medyal ve lateral menisküs normal."),
    ("Lateral Meniscus", "tr", "Lateral menisküste yırtık izlenmedi."),

    # Croatian
    ("Lateral Meniscus", "hr", "Lateralni menisk bez znakova degeneracije ili rupture."),
    ("Baker's", "hr", "Bez znakova poplitealne ciste."),

    # German / Dutch / French
    ("ACL", "de", "VKB, HKB, MCL und LCL intakt und physiologisch signalgebend."),
    ("Effusion", "nl", "Geen hydrops."),
    ("Medial Meniscus", "nl", "Normaal voorkomen menisci."),
    ("Baker's", "fr", "Il n'y a pas de kyste poplité."),

    # Greek / Bulgarian
    ("Effusion", "el", "Χωρίς ενδαρθρική συλλογή υγρού."),
    ("Effusion", "bg", "Няма данни за ставен излив."),
    ("Medial Meniscus", "bg", "Нормално представени менискуси."),
]


def test_multilingual_negatives():
    bad = []
    for label, lang, text in MULTILINGUAL_NEGATIVES:
        if not is_neg(text, label):
            bad.append(f"{label}/{lang}: state={state(text, label)} :: {text[:70]}")
    assert not bad, "negatives not detected:\n    " + "\n    ".join(bad)


def test_negative_soft_target_is_below_the_prior_but_never_zero():
    for label, _lang, text in MULTILINGUAL_NEGATIVES:
        p = prob(text, label)
        assert p < L.LABEL_PRIORS[label], f"{label}: {p:.3f} >= prior"
        assert p > 0.0, f"{label}: a rule-derived negative must not be a hard 0"


def test_trailing_qualifier_does_not_negate():
    """'... fracture ... WITHOUT articular surface collapse' is a fracture.

    A symmetric negation window read the trailing 'without' as denying the
    fracture; that is one of the 58 gold studies, and it flipped the label."""
    t = ("Subchondral insufficiency fracture at the medial margin of the medial "
         "tibial plateau without articular surface collapse.")
    assert is_pos(t, "Fracture"), state(t, "Fracture")


def test_section_header_binds_to_its_answer():
    """'Fractures :\\nAucune.' must be a NEGATIVE, not a bare mention of a
    fracture followed by an orphan 'Aucune'."""
    assert is_neg("CONSTATATIONS :\n\nFractures :\nAucune.\n", "Fracture")
    assert is_neg("Joint effusion: None.\nBaker cyst: None.", "Baker's")
    assert is_neg("Medial collateral ligament (MCL):\nNormal.", "MCL")


def test_run_together_sentences_are_still_split():
    """Turkish reports omit the space after a full stop; without a split the
    ACL rupture leaks into the MCL sentence and mislabels the MCL."""
    t = ("Anterior çapraz bağda komplet rüptüre sekonder bütünlük kaybı "
         "izlenmiştir.Medial kollateral ligamende incelme izlenmiştir.")
    assert is_pos(t, "ACL")
    assert state(t, "MCL") != "positive", state(t, "MCL")


def test_contrastive_conjunction_ends_the_negation_scope():
    t = ("Eklem kıkırdakları ve kemikler patellofemoral eklem dejenerasyonu ile "
         "uyumlu fokal kıkırdak kayıpları dışında normal.")
    assert is_pos(t, "PF OA"), state(t, "PF OA")


def test_hedged_finding_is_weaker_than_an_asserted_one():
    asserted = "Complete tear of the anterior cruciate ligament."
    hedged = "Suspect grade 2 injury of the anterior cruciate ligament, cannot be excluded."
    assert prob(asserted, "ACL") > prob(hedged, "ACL")


# ══════════════════════════════════════════════════════════════════════════
# 5. Laterality and structure discrimination — precision over recall
# ══════════════════════════════════════════════════════════════════════════

def test_coordinated_sides_negate_both_menisci():
    for t in ("Normal medial and lateral menisci. No evidence of tears.",
              "Medyal ve lateral menisküs normal."):
        assert is_neg(t, "Medial Meniscus"), t
        assert is_neg(t, "Lateral Meniscus"), t


def test_a_medial_tear_does_not_leak_to_the_lateral_meniscus():
    t = ("In the medial compartment, there is a radial tear of the medial meniscus. "
         "The lateral compartment, the meniscus is not torn.")
    assert is_pos(t, "Medial Meniscus")
    assert is_neg(t, "Lateral Meniscus")


def test_pcl_does_not_trigger_acl():
    t = "Interstitial high intensity and laxity in the PCL, suspected complete tear."
    assert state(t, "ACL") != "positive", state(t, "ACL")
    t2 = "Arka çapraz bağda komplet yırtık izlenmiştir."
    assert state(t2, "ACL") != "positive", state(t2, "ACL")


def test_lcl_does_not_trigger_mcl():
    t = "There is grade II ligamentous sprain of the lateral collateral ligament."
    assert state(t, "MCL") != "positive", state(t, "MCL")


def test_medial_patellar_facet_is_pf_oa_not_medial_oa():
    """'medial patellar facet' names the patella, not the medial compartment."""
    t = "There is mild cartilage thinning and surface irregularity at the medial patellar facet."
    assert is_pos(t, "PF OA"), state(t, "PF OA")
    assert state(t, "Medial OA") != "positive", state(t, "Medial OA")


def test_tricompartmental_sets_all_three_oa_labels():
    t = "Tricompartmental osteoarthritis with marginal osteophytes and cartilage loss."
    for lab in ("Medial OA", "Lateral OA", "PF OA"):
        assert is_pos(t, lab), f"{lab}: {state(t, lab)}"


def test_generic_ligament_mention_can_only_produce_a_negative():
    """A positive on 'the cruciate ligaments are abnormal' could be the PCL,
    which is not a label here, so a generic anchor is negative-only."""
    t = "There is a tear of the cruciate ligaments."
    assert state(t, "ACL") != "positive", state(t, "ACL")


# ══════════════════════════════════════════════════════════════════════════
# 6. Unmentioned -> calibrated prior, never 0
# ══════════════════════════════════════════════════════════════════════════

def test_unmentioned_maps_to_the_calibrated_prior_not_zero():
    """The load-bearing rule of this whole module.  Measured on the 58 gold
    studies, a finding the report never mentions is still present 12.6% of the
    time (95% CI [0.082, 0.170]); writing 0 there fabricates negatives."""
    t = "Technique: MRI of the knee. Medial meniscus tear."
    f = L.extract_findings(t)
    for lab in ("Baker's", "Fracture", "Synovitis"):
        assert f[lab].state == "unmentioned", f"{lab}: {f[lab].state}"
        assert f[lab].prob == L.LABEL_PRIORS[lab], \
            f"{lab}: {f[lab].prob} != prior {L.LABEL_PRIORS[lab]}"
        assert f[lab].prob > 0.0, f"{lab}: unmentioned collapsed to a hard 0"


def test_unmentioned_confidence_matches_the_consumer_contract():
    """train.py's derived_cell_weights() multiplies the per-cell confidence
    straight into the loss weight, on the documented assumption that an
    unmentioned cell comes back with confidence 0 and therefore costs nothing.
    The TARGET stays at the prior; only the weight is zeroed."""
    f = L.extract_findings("Technique: MRI of the knee. Medial meniscus tear.")
    assert f["Baker's"].confidence == L.UNMENTIONED_CONFIDENCE
    assert L.UNMENTIONED_CONFIDENCE == 0.0
    assert f["Medial Meniscus"].confidence > 0.0


def test_every_prior_is_strictly_between_zero_and_one():
    for lab in L.KNEE_LABELS:
        p = L.LABEL_PRIORS[lab]
        assert 0.0 < p < 1.0, f"{lab}: prior {p}"


def test_empty_report_gives_every_label_its_prior():
    for text in ("", None, "   "):
        f = L.extract_findings(text)
        assert set(f) == set(L.KNEE_LABELS)
        for lab in L.KNEE_LABELS:
            assert f[lab].state == "unmentioned"
            assert f[lab].prob == L.LABEL_PRIORS[lab]


def test_global_normal_template_negates_everything():
    """~183 studies share their report text; those are boilerplate NORMAL
    templates, not duplicate patients, and they are genuine negatives."""
    t = "Técnica: RMN de la rodilla. Resultados: Sin anomalías. Impresión: Sin anomalías"
    f = L.extract_findings(t)
    for lab in L.KNEE_LABELS:
        assert f[lab].state in ("negative", "weak_negative"), f"{lab}: {f[lab].state}"
        assert f[lab].prob < L.LABEL_PRIORS[lab]
    # accent-free spelling must behave identically
    t2 = "Técnica: RMN de la rodilla. Resultados: Sin anomalias. Impresión: Sin anomalias."
    assert L.extract_findings(t2)["ACL"].state in ("negative", "weak_negative")


def test_global_normal_is_blocked_by_any_positive_finding():
    t = ("Sonuç: Diz ekleminde minimal mayii artışı dışından normal sınırlarda bulgular. "
         "Medial menisküste yırtık mevcuttur.")
    assert is_pos(t, "Medial Meniscus"), state(t, "Medial Meniscus")


# ══════════════════════════════════════════════════════════════════════════
# 7. Output contract
# ══════════════════════════════════════════════════════════════════════════

def test_probabilities_and_confidences_are_in_range():
    texts = [t for _, _, t in MULTILINGUAL_POSITIVES + MULTILINGUAL_NEGATIVES]
    texts += ["", "Técnica: RMN de la rodilla. Resultados: Sin anomalías."]
    for t in texts:
        probs, confs = L.report_soft_targets(t)
        assert set(probs) == set(L.KNEE_LABELS)
        for lab in L.KNEE_LABELS:
            assert 0.0 < probs[lab] < 1.0, f"{lab}: {probs[lab]} on {t[:50]!r}"
            assert 0.0 <= confs[lab] <= 1.0, f"{lab}: conf {confs[lab]}"


def test_calibration_table_is_monotone():
    """negative <= weak_negative <= prior <= weak_positive <= positive.
    A non-monotone band would rank a denied finding above a hedged one."""
    for lab in L.KNEE_LABELS:
        row = L.LABEL_STATE_PROB[lab]
        seq = [row["negative"], row["weak_negative"], L.LABEL_PRIORS[lab],
               row["weak_positive"], row["positive"]]
        assert all(seq[i] <= seq[i + 1] + 1e-9 for i in range(4)), f"{lab}: {seq}"


def test_no_torch_or_sklearn_dependency():
    """This module runs at data-prep time on the Kaggle image; it must not drag
    in a heavyweight import, and its AUC helper must not need sklearn."""
    src = open(os.path.join(REPO, "src", "labels.py")).read()
    for banned in ("import torch", "import sklearn", "from sklearn"):
        assert banned not in src, f"src/labels.py must not {banned}"


def test_auc_helper_matches_a_known_value():
    assert abs(L._auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) - 0.75) < 1e-9
    assert abs(L._auc([0, 1], [0.5, 0.5]) - 0.5) < 1e-9      # all ties -> 0.5


# ══════════════════════════════════════════════════════════════════════════
# 8. Integration hook in src/kaggle_data.py
# ══════════════════════════════════════════════════════════════════════════

def _fake_cfg(**kw):
    c = types.SimpleNamespace(
        data_dir=DATA, use_report_labels=True, report_csv="train.csv",
        report_text_col="Report", report_label_weight=0.35,
        report_min_confidence=0.10, report_labels_require_images=False,
        report_label_max_studies=0, report_label_calibrate_on_folds="",
        train_csv="train_gold.csv",
    )
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _gold_and_reports():
    import pandas as pd
    gold = pd.DataFrame({
        "StudyInstanceUID": ["GOLD1", "GOLD2"],
        "fold": [0, 1],
    })
    for lab in L.KNEE_LABELS:
        gold[lab] = 1.0                          # deliberately all-positive
    reports = pd.DataFrame({
        "StudyInstanceUID": ["GOLD1", "REP1", "REP2"],
        "Report": [
            "Técnica: RMN de la rodilla. Resultados: Sin anomalías.",   # would say 0
            "There is complete tear of the anterior cruciate ligament. No joint effusion.",
            "Diz eklemi içi sıvı miktarı normal. Çapraz ve yan bağlar normal.",
        ],
    })
    return gold, reports


def test_hook_is_a_no_op_when_the_flag_is_off():
    import numpy as np
    import src.kaggle_data as kd
    gold, reports = _gold_and_reports()
    cfg = _fake_cfg(use_report_labels=False)
    out, w = kd.attach_report_label_rows(gold, None, cfg, L.KNEE_LABELS,
                                         report_df=reports)
    assert len(out) == len(gold), "flag OFF must not add rows"
    assert w.shape == (len(gold), len(L.KNEE_LABELS))
    assert np.allclose(w, 1.0)


def test_hook_never_overwrites_a_gold_label():
    """GOLD1 is all-positive AND has a 'Sin anomalías' report.  The report must
    not be able to touch it: gold studies are excluded outright."""
    import src.kaggle_data as kd
    gold, reports = _gold_and_reports()
    out, w = kd.attach_report_label_rows(gold, None, _fake_cfg(), L.KNEE_LABELS,
                                         report_df=reports)
    assert (out["StudyInstanceUID"] == "GOLD1").sum() == 1, "gold study duplicated"
    g1 = out[out["StudyInstanceUID"] == "GOLD1"].iloc[0]
    for lab in L.KNEE_LABELS:
        assert float(g1[lab]) == 1.0, f"gold label {lab} was overwritten"
    # ... and its loss weights are still 1.0
    idx = int(out.index[out["StudyInstanceUID"] == "GOLD1"][0])
    assert (w[idx] == 1.0).all(), "gold cell weights were down-weighted"


def test_hook_adds_report_rows_that_can_never_be_validated_on():
    import src.kaggle_data as kd
    gold, reports = _gold_and_reports()
    out, w = kd.attach_report_label_rows(gold, None, _fake_cfg(), L.KNEE_LABELS,
                                         report_df=reports)
    added = out[out["StudyInstanceUID"].isin(["REP1", "REP2"])]
    assert len(added) == 2, out["StudyInstanceUID"].tolist()
    assert (added["fold"] == -1).all(), "a report row must never join a val fold"
    assert bool(added["is_report"].all())
    assert not bool(out[out["StudyInstanceUID"].str.startswith("GOLD")]["is_report"].any())


def test_hook_weights_report_rows_below_gold():
    import numpy as np
    import src.kaggle_data as kd
    gold, reports = _gold_and_reports()
    cfg = _fake_cfg()
    out, w = kd.attach_report_label_rows(gold, None, cfg, L.KNEE_LABELS,
                                         report_df=reports)
    n_gold = len(gold)
    assert np.allclose(w[:n_gold], 1.0)
    assert w[n_gold:].max() <= cfg.report_label_weight + 1e-6, \
        "a report cell outweighed a gold cell"
    assert w[n_gold:].max() > 0.0, "every report cell was filtered out"


def test_hook_soft_targets_reach_the_frame_and_respect_the_prior():
    import src.kaggle_data as kd
    gold, reports = _gold_and_reports()
    out, w = kd.attach_report_label_rows(gold, None, _fake_cfg(), L.KNEE_LABELS,
                                         report_df=reports)
    rep1 = out[out["StudyInstanceUID"] == "REP1"].iloc[0]
    assert float(rep1["ACL"]) > 0.5, "asserted ACL tear did not become a high target"
    assert float(rep1["Effusion"]) < 0.2, "denied effusion did not become a low target"
    # unmentioned in REP1 -> the calibrated prior, not 0 ...
    assert abs(float(rep1["Baker's"]) - L.LABEL_PRIORS["Baker's"]) < 1e-6
    assert float(rep1["Baker's"]) > 0.0
    # ... carried at weight 0, so the prior is present but costs nothing
    r = int(out.index[out["StudyInstanceUID"] == "REP1"][0])
    j = list(L.KNEE_LABELS).index("Baker's")
    assert w[r, j] == 0.0, "an unmentioned cell reached the loss with weight"
    assert w[r, list(L.KNEE_LABELS).index("ACL")] > 0.0


def test_hook_respects_exclude_ids():
    import src.kaggle_data as kd
    gold, reports = _gold_and_reports()
    out, _ = kd.attach_report_label_rows(gold, None, _fake_cfg(), L.KNEE_LABELS,
                                         report_df=reports, exclude_ids=["REP1"])
    assert "REP1" not in set(out["StudyInstanceUID"])
    assert "REP2" in set(out["StudyInstanceUID"])


def test_config_defaults_keep_report_labels_off():
    from src.config import Config
    c = Config()
    assert c.use_report_labels is False, \
        "report labels must default OFF so no other run changes"
    assert hasattr(c, "report_label_weight") and 0 < c.report_label_weight <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# 9. Real data — the number that actually matters
# ══════════════════════════════════════════════════════════════════════════

def _load_gold():
    path = os.path.join(DATA, "train_gold.csv")
    if not os.path.exists(path):
        raise _Skip("data_subset/train_gold.csv not present")
    import csv
    import pandas as pd
    csv.field_size_limit(10 ** 9)
    return pd.read_csv(path, engine="python")


def test_macro_auc_against_the_58_gold_studies():
    gold = _load_gold()
    rep = L.evaluate_against_gold(gold)
    macro = float(rep["auc"].mean())
    print(f"        macro ROC-AUC vs gold = {macro:.4f}")
    assert macro > 0.80, f"macro AUC regressed to {macro:.4f}"


def test_no_label_is_worse_than_chance():
    gold = _load_gold()
    rep = L.evaluate_against_gold(gold)
    bad = rep[rep["auc"] < 0.55]
    assert bad.empty, "labels at/below chance:\n" + bad[["auc"]].to_string()


def test_rule_asserted_negatives_are_reliable():
    """Precision over recall: the rules are allowed to stay silent, but a
    negative they DO assert has to be trustworthy, because it is the closest
    thing to a hard 0 that reaches the loss."""
    gold = _load_gold()
    rep = L.evaluate_against_gold(gold)
    called = rep[rep["n_neg_called"] >= 5]
    assert not called.empty
    assert (called["npv"] >= 0.85).all(), \
        "unreliable negatives:\n" + called[["n_neg_called", "npv"]].to_string()


def test_soft_targets_are_calibrated_on_the_gold_set():
    """mean(soft target) should land near the true prevalence for each label;
    that is what makes the numbers usable as targets and not just as a ranking."""
    gold = _load_gold()
    for lab in L.KNEE_LABELS:
        mean_p = sum(L.extract_findings(t)[lab].prob for t in gold["Report"]) / len(gold)
        prev = float(gold[lab].mean())
        assert abs(mean_p - prev) < 0.10, \
            f"{lab}: mean soft {mean_p:.3f} vs prevalence {prev:.3f}"


def test_whole_corpus_runs_without_raising():
    path = os.path.join(DATA, "train.csv")
    if not os.path.exists(path):
        raise _Skip("data_subset/train.csv not present")
    import csv
    import pandas as pd
    csv.field_size_limit(10 ** 9)
    df = pd.read_csv(path, engine="python")
    assert df["Report"].notna().all(), "a report went missing"
    for t in df["Report"].head(400):
        f = L.extract_findings(t)
        assert len(f) == 12


def main():
    print("\n=== src/labels.py tests ===")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and isinstance(fn, types.FunctionType):
            check(name[5:], fn)
    print(f"\n{len(PASSED)} passed, {len(FAILED)} failed, {len(SKIPPED)} skipped")
    for n, e in FAILED:
        print(f"  - {n}: {e}")
    for n, e in SKIPPED:
        print(f"  ~ {n}: {e}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
