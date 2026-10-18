import os
import sys
import json
import pytest


# Ensure root AegisParse directory is on sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import app  
import ontology_assistant  


# ==============================================================================
# 1. SINGLE SOURCE OF TRUTH JSON INTEGRITY TESTS
# ==============================================================================
def test_configuration_json_files_valid():
    """Verifies that all 3 core JSON knowledge files exist and are non-empty."""
    assert isinstance(app.APP_CONFIG, dict) and len(app.APP_CONFIG) > 0
    assert isinstance(app.LEXICONS, dict) and len(app.LEXICONS) >= 10

    with open(os.path.join(ROOT_DIR, "seed_ontology.json"), "r", encoding="utf-8") as f:
        seed = json.load(f)
    assert len(seed) >= 190
    assert "LOINC:10351-5" in seed  # HIV-1 Viral Load
    assert "LOINC:2077-6" in seed   # Sweat Chloride


# ==============================================================================
# 2. CASE-SENSITIVE SI UNIT & VALUE/UNIT DECOUPLING TESTS
# ==============================================================================
def test_case_sensitive_si_unit_normalization():
    """
    Critical Clinical Guard:
    Uppercase 'G/L' (Giga/Liter = 10*9/L for WBC/Platelets) must NEVER collapse
    into lowercase 'g/L' (grams/Liter for Hemoglobin/Albumin).
    """
    assert app.normalize_unit("G/L") == "10*9/l"
    assert app.normalize_unit("g/L") == "g/l"
    assert app.get_unit_dimension("G/L") == "COUNT_VOL"
    assert app.get_unit_dimension("g/L") == "MASS_VOL"


def test_decouple_squashed_value_unit_and_scientific_notation():
    """
    Verifies that merged value+unit strings split cleanly while scientific
    notation values ('5.4 x 10^6') remain intact.
    """
    val1, unit1 = app.decouple_squashed_value_unit("110 U/L", None)
    assert val1 == "110"
    assert unit1 == "u/l"

    val2, unit2 = app.decouple_squashed_value_unit("1:320 Títulos", None)
    assert val2 == "1:320"
    assert unit2 == "titer"

    # Scientific notation must not be split into a bogus unit
    val3, unit3 = app.decouple_squashed_value_unit("5.4 x 10^6", "10*6/uL")
    assert val3 == "5.4 x 10^6"
    assert unit3 == "10*6/uL"


# ==============================================================================
# 3. 30-LANGUAGE BIDI RTL, HOMOGLYPH & CJK KANJI LEXICON TESTS
# ==============================================================================
def test_visual_rtl_arabic_reversal_restoration():
    """Restores visually reversed Arabic laboratory strings to logical order."""
    reversed_arabic = "(LDH) ميزنإ"
    fixed = app.fix_visual_rtl_reversal(reversed_arabic)
    assert fixed == "(LDH) إنزيم"


def test_homoglyph_normalization_in_short_acronyms():
    """Normalizes Cyrillic/Greek lookalike characters inside short lab acronyms."""
    # Cyrillic 'С', 'Р', 'Т' lookalikes in 'СРТ'
    cleaned = app.normalize_homoglyphs_in_acronyms("СRP")
    assert cleaned == "CRP"


def test_cjk_pure_kanji_japanese_transliteration():
    """
    Verifies Bug 1 Fix: Pure-Kanji Japanese test names ('白血球数', '中性脂肪')
    match Chinese Unicode range [\\u4E00-\\u9FFF] but still resolve Japanese roots.
    """
    translit_wbc = app.transliterate_via_lexicon("白血球数")
    assert "Leukocytes" in translit_wbc

    translit_tg = app.transliterate_via_lexicon("中性脂肪")
    assert "Triglycerides" in translit_tg


def test_nordic_and_european_specimen_prefix_stripping():
    """Verifies S-, P-, B-, U-, fP-, fS-, Liq- prefixes are stripped into candidates."""
    cands = app.extract_candidate_keys("S-Karbamatsepiini")
    assert "karbamatsepiini" in cands

    cands_fasting = app.extract_candidate_keys("fP-Glukos")
    assert "glukos" in cands_fasting


# ==============================================================================
# 4. 10-STAGE CLINICAL SUBTYPE & SPECIMEN SYSTEM GUARD TESTS
# ==============================================================================
@pytest.fixture(scope="module")
def mapper_no_vecs(monkeypatch_module=None):
    """Instantiates Phase2OntologyMapper without calling Ollama embeddings."""
    orig_embed = app.get_embedding
    app.get_embedding = lambda text: None
    with open(os.path.join(ROOT_DIR, "seed_ontology.json"), "r", encoding="utf-8") as f:
        ontology_data = json.load(f)
    mapper = app.Phase2OntologyMapper(ontology_data)
    app.get_embedding = orig_embed
    return mapper


def test_specimen_system_axis_guards(mapper_no_vecs):
    """
    Ensures Urine, CSF, Stool, Sweat, and Sputum analytes NEVER collide with
    Serum/Blood LOINC targets.
    """
    # 1. Urine Glucose vs Serum Glucose
    assert mapper_no_vecs._violates_subtype_guard("Glukoza u urinu", "glucose") is True
    assert mapper_no_vecs._violates_subtype_guard("Glukoza u urinu", "glucose in urine") is False

    # 2. CSF Glucose vs Serum Glucose
    assert mapper_no_vecs._violates_subtype_guard("Glukose (spinalvæske)", "glucose") is True
    assert mapper_no_vecs._violates_subtype_guard("Glukose (spinalvæske)", "glucose in cerebral spinal fluid") is False

    # 3. Sweat Chloride vs Serum Chloride
    assert mapper_no_vecs._violates_subtype_guard("Svedtest (Klorid)", "klorid") is True
    assert mapper_no_vecs._violates_subtype_guard("Svedtest (Klorid)", "svedtest (klorid)") is False

    # 4. Stool Elastase vs Serum
    assert mapper_no_vecs._violates_subtype_guard("Elastáza 1 v stolici", "protein") is True


def test_clinical_subtype_guards(mapper_no_vecs):
    """
    Verifies clinical subtype isolation:
    - HbA1c / HbS / HbA2 vs generic Hemoglobin
    - Direct / Indirect Bilirubin vs Total Bilirubin
    - Pathogen-specific IgM/IgG vs Total IgM/IgG
    - Autoimmune AChR Receptor Ab must NOT be blocked by genetic mutation guard
    """
    assert mapper_no_vecs._violates_subtype_guard("Hemoglobin A1c", "hemoglobin") is True
    assert mapper_no_vecs._violates_subtype_guard("Direct Bilirubin", "total bilirubin") is True
    assert mapper_no_vecs._violates_subtype_guard("Dengue IgM", "igm") is True
    assert mapper_no_vecs._violates_subtype_guard("Factor V Leiden (מוטציה)", "factor v") is True
    # Bug 5 Fix verification: 'receptor' in AChR must not trigger mutation guard
    assert mapper_no_vecs._violates_subtype_guard(
        "AChR-antikroppar (Acetylcholinreceptor)", "achr-antikroppar"
    ) is False


def test_non_laboratory_observation_exclusion():
    """Verifies radiology, ECG/echocardiography, and vital signs are flagged as non-lab."""
    assert app.is_non_laboratory_observation("Chest X-Ray (RTG)") is True
    assert app.is_non_laboratory_observation("Brain MRI (Kaudatusatrofi)") is True
    assert app.is_non_laboratory_observation("Ecocardiograma (LVEF)") is True
    assert app.is_non_laboratory_observation("Blood Pressure", "mmHg") is True
    assert app.is_non_laboratory_observation("Body Mass Index (BMI)", "kg/m2") is True
    assert app.is_non_laboratory_observation("Hemoglobin (HGB)", "g/dL") is False


# ==============================================================================
# 5. ONTOLOGY ASSISTANT 6-AXIS BONUS & PHASE 3 EVALUATOR TESTS
# ==============================================================================
def test_ontology_assistant_6axis_bonus():
    """Verifies 6-Axis LOINC property and specimen system re-ranking bonuses."""
    # Mass/volume unit ('mg/dL') should boost [Mass/volume] and penalize [Moles/volume]
    b_mass = ontology_assistant.compute_loinc_6axis_bonus(
        "Creatinine [Mass/volume] in Serum or Plasma", "mg/dL", "Creatinine"
    )
    b_molar = ontology_assistant.compute_loinc_6axis_bonus(
        "Creatinine [Moles/volume] in Serum or Plasma", "mg/dL", "Creatinine"
    )
    assert b_mass > b_molar

    # CSF query should boost 'in Cerebral spinal fluid' and penalize 'in Serum'
    b_csf = ontology_assistant.compute_loinc_6axis_bonus(
        "Glucose [Moles/volume] in Cerebral spinal fluid", "mmol/L", "Glukose (spinalvæske)"
    )
    b_ser = ontology_assistant.compute_loinc_6axis_bonus(
        "Glucose [Moles/volume] in Serum or Plasma", "mmol/L", "Glukose (spinalvæske)"
    )
    assert b_csf > b_ser


def test_phase3_evaluator_excludes_non_lab_from_denominator():
    """Verifies NON_LAB_OBSERVATION rows do not penalize laboratory LOINC mapping accuracy."""
    sample_payload = {
        "patient_name": " Maria Papadopoulou ",
        "patient_id": "MRN-10492",
        "drawn_time": "2026-10-15",
        "clinic": "Athens General",
        "physician": "Dr. Nikolaos",
        "lab_name": "Central Labs",
        "clinical_impressions": "Normal hematology and chemistry panel.",
        "tests": [
            {
                "raw_extracted_name": "Hemoglobin",
                "result_value": "14.2",
                "unit": "g/dL",
                "loinc_code": "LOINC:718-7",
                "similarity_score": 1.0,
                "match_type": "EXACT_KEYWORD"
            },
            {
                "raw_extracted_name": "Chest X-Ray",
                "result_value": "Normal",
                "unit": None,
                "loinc_code": "NON_LAB_OBSERVATION",
                "similarity_score": 0.0,
                "match_type": "EXCLUDED_NON_LAB"
            }
        ]
    }
    metrics = app.Phase3Evaluator.evaluate(sample_payload, "test_patient")
    summary = metrics["evaluation_summary"]
    assert summary["total_tests_extracted"] == 2
    assert summary["total_lab_tests_evaluated"] == 1
    assert summary["total_non_lab_excluded"] == 1
    assert summary["phase_2_loinc_mapping_accuracy_pct"] == 100.0


