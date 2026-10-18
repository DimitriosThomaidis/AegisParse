import os
import glob
import json
import logging
import re
import unicodedata
import numpy as np
import ollama
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel
from celery import Celery

# RTL & BiDi Correction Libraries
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    BIDI_SUPPORT = True
except ImportError:
    BIDI_SUPPORT = False

# Enterprise Vector Search (FAISS)
try:
    import faiss
    FAISS_SUPPORT = True
except ImportError:
    FAISS_SUPPORT = False

# ==============================================================================
# 1. CONFIGURATION & PATHS (SINGLE SOURCE OF TRUTH VIA config_rules.json)
# ==============================================================================
BASE_DIR = os.getenv("AEGIS_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.getenv("AEGIS_IMAGES_DIR", os.path.join(BASE_DIR, "Images"))
JSON_DIR = os.getenv("AEGIS_JSON_DIR", os.path.join(BASE_DIR, "Json_Outputs"))
ONTOLOGY_FILE = os.path.join(BASE_DIR, "seed_ontology.json")
CONFIG_RULES_FILE = os.path.join(BASE_DIR, "config_rules.json")
LEXICONS_FILE = os.path.join(BASE_DIR, "clinical_lexicons.json")

os.makedirs(JSON_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S"
)

MODEL_VISION = os.getenv("AEGIS_MODEL_VISION", "qwen3.5:4b")
MODEL_EMBEDDING = os.getenv("AEGIS_MODEL_EMBEDDING", "nomic-embed-text")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=180.0)

CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "memory://")
celery_app = Celery("aegisparse", broker=CELERY_BROKER_URL)
celery_app.conf.task_always_eager = (CELERY_BROKER_URL == "memory://")

def load_json(filepath: str) -> dict:
    if not os.path.exists(filepath):
        logging.warning(f"Configuration file not found: {filepath}")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data if isinstance(data, dict) else {}

APP_CONFIG = load_json(CONFIG_RULES_FILE)
LEXICONS = load_json(LEXICONS_FILE)

# Dynamically loaded rule sets from config_rules.json
SEMANTIC_NOISE_CFG = APP_CONFIG.get("semantic_noise", {})
if not isinstance(SEMANTIC_NOISE_CFG, dict):
    SEMANTIC_NOISE_CFG = {}

STATUS_NOISE_PATTERNS: List[str] = SEMANTIC_NOISE_CFG.get("status_noise_patterns", [
    r"\s*\[[A-Za-z]{1,4}\]\s*$",
    r"\s*[↓↑]\s*$"
])
HOMOGLYPH_MAP: Dict[str, str] = SEMANTIC_NOISE_CFG.get("homoglyphs", {})
NON_ACRONYM_WORDS: set = {
    w.lower() for w in SEMANTIC_NOISE_CFG.get("non_analyte_parentheticals", [])
}
HEADER_BLACKLIST: set = {h.lower() for h in APP_CONFIG.get("HEADER_EXACT_NAMES", [])}
CASE_SENSITIVE_TOKEN_MAP: Dict[str, str] = APP_CONFIG.get("case_sensitive_token_map", {})
SPECIMEN_SYSTEM_MARKERS: Dict[str, List[str]] = APP_CONFIG.get("specimen_system_markers", {})
UNIT_DIMENSIONS_RAW = APP_CONFIG.get("unit_dimensions", {})
UNIT_DIMENSIONS_MAP: Dict[str, set] = {
    dim: {u.lower() for u in units}
    for dim, units in (UNIT_DIMENSIONS_RAW.items() if isinstance(UNIT_DIMENSIONS_RAW, dict) else [])
    if isinstance(units, list)
}
ACRONYM_EXPANSIONS_RAW = APP_CONFIG.get("clinical_acronym_expansions", {})
CLINICAL_ACRONYM_EXPANSIONS: Dict[str, str] = {
    k.lower(): re.sub(r"\s+", " ", re.sub(r"[^\w\s\-\+/]", " ", v.lower(), flags=re.UNICODE)).strip()
    for k, v in (ACRONYM_EXPANSIONS_RAW.items() if isinstance(ACRONYM_EXPANSIONS_RAW, dict) else [])
    if isinstance(v, str)
}

# Non-laboratory clinical observations (Radiology, Cardiology, Vitals/Anthropometrics)
DEFAULT_NON_LAB_MARKERS = [
    r"\bx\-?ray\b", r"\bradiograf", r"\bröntgen\b", r"\brtg\b", r"\bрентген", r"\bακτινογραφία",
    r"\bmri\b", r"\birm\b", r"\bмагнитно\-резонанс", r"\bμαγνητική\s+τομογραφία", r"\bkaudatusatrofi",
    r"\bct\s+scan\b", r"\btomograf", r"\bкомпьютерная\s+томограф", r"\bαξονική\s+τομογραφία",
    r"\bultrasound\b", r"\becograf", r"\bechograf", r"\béchographie\b", r"\busg\b", r"\bузи\b", r"\bυπερηχογράφημα",
    r"\bechocardiogra", r"\becocardiogra", r"\bεχοκαρδιογράφημα", r"\blvef\b", r"\bejection\s+fraction\b",
    r"\becg\b", r"\bekg\b", r"\bэлектрокардиограф", r"\bηλεκτροκαρδιογράφημα",
    r"\bblood\s+pressure\b", r"\bciśnienie\s+tętnicze\b", r"\btensión\s+arterial\b", r"\bpressione\s+arteriosa\b",
    r"\bαρτηριακή\s+πίεση", r"\bартериальное\s+давление", r"\b혈압\b", r"\b血压\b", r"\b血圧\b",
    r"לחץ\s+דם", r"לץח\s+דם", r"סיסטול",
    r"\bbody\s+mass\s+index\b", r"\bbmi\b", r"\bδείκτης\s+μάζας\s+σώματος\b", r"\bимт\b",
    r"\bwaist\s+circumference\b", r"\bobwód\s+talii\b", r"\bπερίμετρος\s+μέσης"
]
NON_LAB_CLINICAL_PATTERNS: List[str] = list(dict.fromkeys(
    APP_CONFIG.get("non_lab_clinical_markers", []) + DEFAULT_NON_LAB_MARKERS
))

# Dynamically build non-Latin script regex from clinical_lexicons.json
NON_LATIN_SCRIPT_REGEX = re.compile(
    "|".join(
        data["script_regex"]
        for lang, data in LEXICONS.items()
        if isinstance(data, dict) and lang != "extended_latin" and data.get("script_regex")
    ) or r"[\u0370-\u03FF\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]"
)

# ==============================================================================
# 2. SCHEMAS (COMPACT REQUIRED FIELDS FOR PHASE 0 & PHASE 1)
# ==============================================================================
class GatekeeperSchema(BaseModel):
    is_medical_lab_report: bool
    verdict: str
    detected_content: str

class LabTest(BaseModel):
    raw_extracted_name: str
    result_value: str
    unit: str | None
    reference_range: str | None

class MedicalReportSchema(BaseModel):
    patient_name: str | None
    patient_id: str | None
    drawn_time: str | None
    clinic: str | None
    physician: str | None
    lab_name: str | None
    clinical_impressions: str | None
    tests: List[LabTest]

# ==============================================================================
# 3. 30-LANGUAGE NORMALIZATION, HOMOGLYPH & BIDI ENGINE
# ==============================================================================
def normalize_homoglyphs_in_acronyms(text: str) -> str:
    """Uses config_rules.json homoglyphs to normalize Cyrillic/Greek lookalikes in short tokens."""
    if not HOMOGLYPH_MAP:
        return text
    def fix_token(tok: str) -> str:
        clean_tok = tok.strip("()[]")
        if 1 <= len(clean_tok) <= 6 and re.search(r"[A-Za-z0-9]", clean_tok):
            return "".join(HOMOGLYPH_MAP.get(ch, ch) for ch in tok)
        return tok
    return " ".join(fix_token(t) for t in text.split())

def fix_visual_rtl_reversal(text: str) -> str:
    """Restores logical character order for visually reversed Arabic/Hebrew strings."""
    if not text or not re.search(r"[\u0590-\u05FF\u0600-\u06FF]", text):
        return text

    arabic_words = re.findall(r"[\u0600-\u06FF]+", text)
    if arabic_words:
        reversed_markers = sum(
            1 for w in arabic_words
            if (len(w) >= 4 and w.endswith("لا"))
            or (len(w) >= 2 and w.startswith("ة"))
            or w in {"ميزنإ", "ريغ", "في", "نم", "عم", "ىلإ"}
        )
        if reversed_markers >= 1 and not any(w.startswith("ال") and len(w) >= 4 for w in arabic_words):
            m_prefix = re.match(r"^(\([A-Za-z0-9\s\-\+/]+\))\s*(.+)$", text)
            m_suffix = re.match(r"^(.+?)\s*(\([A-Za-z0-9\s\-\+/]+\))$", text)
            if m_prefix:
                return f"{m_prefix.group(1)} {m_prefix.group(2)[::-1].strip()}"
            if m_suffix:
                return f"{m_suffix.group(1)[::-1].strip()} {m_suffix.group(2)}"
            if re.search(r"[A-Za-z]", text):
                tokens = text.split()
                fixed_tokens = [tok[::-1] if re.search(r"[\u0600-\u06FF]", tok) else tok for tok in tokens]
                return " ".join(fixed_tokens)
            return text[::-1].strip()
    return text

def fix_bidi_and_unicode(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    cleaned = unicodedata.normalize("NFKC", str(text)).strip()
    if not cleaned or cleaned.lower() in {"null", "none", "n/a", "not visible"}:
        return None
    return fix_visual_rtl_reversal(cleaned)

def detect_script_language(text: str) -> str:
    for lang, data in LEXICONS.items():
        if not isinstance(data, dict):
            continue
        regex = data.get("script_regex", "")
        if regex and re.search(regex, text):
            return lang
    return ""

def transliterate_via_lexicon(text: str) -> str:
    """
    Transliterates non-Latin or extended-Latin strings using clinical_lexicons.json.
    Merges Japanese and Chinese native_roots when CJK script is detected so pure-Kanji
    Japanese test names ('白血球数', '好中球', '中性脂肪') never bypass Japanese roots.
    """
    lang = detect_script_language(text)
    if not lang or lang not in LEXICONS or not isinstance(LEXICONS[lang], dict):
        return text
    char_map = LEXICONS[lang].get("alphabet_map", {})
    transliterated = "".join([char_map.get(ch, ch) for ch in text]) if isinstance(char_map, dict) and char_map else text

    native_roots = dict(LEXICONS[lang].get("native_roots", {}))
    if lang in {"chinese", "japanese"}:
        other_cjk = "japanese" if lang == "chinese" else "chinese"
        if isinstance(LEXICONS.get(other_cjk), dict):
            native_roots.update(LEXICONS[other_cjk].get("native_roots", {}))

    if isinstance(native_roots, dict) and native_roots:
        is_unspaced = lang in {"chinese", "japanese", "thai", "korean"}
        for native_word, english_term in sorted(native_roots.items(), key=lambda x: len(x[0]), reverse=True):
            if is_unspaced:
                if native_word.lower() in transliterated.lower():
                    transliterated = re.sub(rf"(?i){re.escape(native_word)}", f" {english_term} ", transliterated)
            else:
                pat = rf"(?i)(?<!\w){re.escape(native_word)}(?!\w)"
                if re.search(pat, transliterated):
                    transliterated = re.sub(pat, f" {english_term} ", transliterated)
    return re.sub(r"\s+", " ", transliterated).strip()

def normalize_unit(unit_str: Optional[str]) -> Optional[str]:
    """
    Normalizes clinical measurement units while preserving case-sensitive SI hematology
    distinctions (e.g. 'G/L' = 10*9/L Giga/L vs. 'g/L' = grams/L) and bilingual units ('滴度 (Titer)').
    """
    if not unit_str:
        return None
    u = fix_bidi_and_unicode(unit_str)
    if not u:
        return None
    u_clean = u.strip()
    if u_clean.lower() in {"-", "—", "", "none", "null", "n/a", "ลบ"}:
        return None
    u_stripped = u_clean
    if (u_stripped.startswith("(") and u_stripped.endswith(")")) or (u_stripped.startswith("[") and u_stripped.endswith("]")):
        u_stripped = u_stripped[1:-1].strip()

    # 1. Check case-sensitive SI unit map first (distinguishes 'G/L' from 'g/L')
    if isinstance(CASE_SENSITIVE_TOKEN_MAP, dict) and u_stripped in CASE_SENSITIVE_TOKEN_MAP:
        return CASE_SENSITIVE_TOKEN_MAP[u_stripped]

    # 2. Check lowercase token_map
    u_lower = u_stripped.lower().replace("μ", "u").replace("µ", "u")
    u_lower_compact = re.sub(r"\s+", " ", u_lower).strip()
    token_map = APP_CONFIG.get("token_map", {})
    if isinstance(token_map, dict):
        if u_lower_compact in token_map:
            return token_map[u_lower_compact]
        u_no_space = u_lower_compact.replace(" ", "")
        if u_no_space in token_map:
            return token_map[u_no_space]
    return u_clean

def clean_test_name(raw_name: str) -> str:
    cleaned = fix_bidi_and_unicode(raw_name) or ""
    cleaned = re.sub(r"^>\s*", "", cleaned).strip()
    for pat in STATUS_NOISE_PATTERNS:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = normalize_homoglyphs_in_acronyms(cleaned)
    cleaned = cleaned.replace("(aPTTT)", "(aPTT)")
    return cleaned

def decouple_squashed_value_unit(val_str: str, unit_str: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Splits merged value+unit strings ('110 U/L', '3.5 10^3/μL', '1:320 Títulos')
    while preserving scientific notation values ('5.4 x 10^6', '4.5 x 10^5').
    """
    clean_v = fix_bidi_and_unicode(val_str) or ""
    clean_u = normalize_unit(unit_str)

    clean_v = re.sub(r"\s+\-$", "", clean_v).strip()

    for pat in STATUS_NOISE_PATTERNS:
        clean_v = re.sub(pat, "", clean_v, flags=re.IGNORECASE).strip()

    match = re.match(
        r"^([<>]?\s*[\d\.,\+\-]+(?:\s*[x×\*]\s*10\^\d+)?)\s+((?!x\s*10\^)(?:10[\^\*]\d+/[a-zA-Zµμ]+|[a-zA-Z%µμ/\u0400-\u04FF][a-zA-Z0-9%µμ/\^\*\s\-\(\)\.\u0400-\u04FF]*))$",
        clean_v,
        flags=re.IGNORECASE
    )
    if match:
        extracted_num = match.group(1).strip()
        extracted_u = normalize_unit(match.group(2).strip())
        clean_v = extracted_num
        if not clean_u:
            clean_u = extracted_u
    elif re.search(r"\s+(?:titer|títulos|titulos)$", clean_v, flags=re.IGNORECASE):
        clean_v = re.sub(r"\s+(?:titer|títulos|titulos)$", "", clean_v, flags=re.IGNORECASE).strip()
        if not clean_u:
            clean_u = "titer"

    return (clean_v if clean_v else "REQUIRES_MANUAL_REVIEW"), clean_u

def is_non_laboratory_observation(raw_name: str, unit: Optional[str] = None) -> bool:
    """
    Detects non-laboratory clinical rows (radiology imaging, echocardiography, ECG,
    vital signs, or anthropometrics) so they are marked as NON_LAB_OBSERVATION rather
    than penalizing laboratory LOINC accuracy.
    """
    if not raw_name:
        return False
    norm = raw_name.lower().strip()
    u_low = (unit or "").lower().strip()
    if u_low in {"bpm", "kg/m2", "kg/m²", "cm", "m"}:
        return True
    for pat in NON_LAB_CLINICAL_PATTERNS:
        if re.search(pat, norm, flags=re.IGNORECASE):
            return True
    return False

# ==============================================================================
# 4. PHASE 0: VISUAL OOD GATEKEEPER (YES / NO MEDICAL LAB REPORT VERIFIER)
# ==============================================================================
def phase_0_verify_medical_lab_image(img_path: str) -> Dict[str, Any]:
    """
    Fast binary visual gatekeeper using qwen3.5:4b (think=False).
    Checks whether the uploaded image is a genuine medical laboratory/diagnostic report
    (in any language/layout) vs. an out-of-distribution image (dog, person, food, car,
    landscape, shopping receipt, or non-laboratory photo).
    """
    gatekeeper_prompt = (
        "Examine this image carefully.\n"
        "Is this image a medical laboratory report, clinical pathology report, or diagnostic test results document "
        "(in any language, paper printout, or digital screen)?\n"
        "RULES:\n"
        "- Answer YES if it contains medical laboratory or diagnostic test names and values/results.\n"
        "- Answer NO if it is a photo of an animal (such as a dog or cat), a person/selfie, nature/landscape, "
        "food, vehicle, everyday object, or a non-medical document (such as a restaurant bill or store receipt).\n"
        "Reply on a single line in this exact format:\n"
        "YES | <brief description>\n"
        "or\n"
        "NO | <brief description of what is in the image>"
    )
    try:
        response = ollama_client.chat(
            model=MODEL_VISION,
            messages=[{"role": "user", "content": gatekeeper_prompt, "images": [img_path]}],
            think=False,
            options={"temperature": 0.0, "num_predict": 40, "num_ctx": 2048}
        )
        raw_reply = response["message"].get("content", "").strip()
        first_line = raw_reply.split("\n")[0].strip()

        parts = [p.strip() for p in first_line.split("|", 1)]
        verdict_token = parts[0].upper()
        reason = parts[1] if len(parts) > 1 else first_line

        if re.search(r"\bNO\b", verdict_token) and not re.search(r"\bYES\b", verdict_token):
            return {
                "is_medical_lab_report": False,
                "verdict": "NO",
                "detected_content": reason
            }
        return {
            "is_medical_lab_report": True,
            "verdict": "YES",
            "detected_content": reason
        }
    except Exception as e:
        logging.warning(f"Phase 0 Gatekeeper warning on {os.path.basename(img_path)}: {e}. Defaulting to allow Phase 1.")
        return {
            "is_medical_lab_report": True,
            "verdict": "YES",
            "detected_content": "Gatekeeper fallback (allowed)"
        }

# ==============================================================================
# 5. PHASE 1: 3-STAGE VISION EXTRACTION + GENERALIZABLE TABLE ALIGNMENT REPAIR
# ==============================================================================
def extract_json_from_text(raw_text: str) -> Optional[Dict[str, Any]]:
    if not raw_text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except Exception:
            pass
    return None

def parse_xml_markdown_fallback(xml_text: str) -> Dict[str, Any]:
    record = {
        "patient_name": None, "patient_id": None, "drawn_time": None,
        "clinic": None, "physician": None, "lab_name": None,
        "clinical_impressions": None, "tests": []
    }

    def get_tag(tag: str) -> str:
        m = re.search(f"<{tag}>(.*?)</{tag}>", xml_text, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    meta_block = get_tag("metadata") or xml_text
    tests_block = get_tag("lab_tests") or xml_text
    nar_block = get_tag("clinical_impressions")
    if nar_block and nar_block.lower() not in {"none", "null", "not visible", "n/a"}:
        record["clinical_impressions"] = nar_block

    synonyms_map = APP_CONFIG.get("field_synonyms", {})
    if not isinstance(synonyms_map, dict):
        synonyms_map = {}

    for line in meta_block.split("\n"):
        if not line.strip().startswith("|") or "---" in line:
            continue
        parts = [p.strip().replace("**", "") for p in line.split("|")][1:-1]
        if len(parts) >= 2:
            k, v = parts[0].lower().strip(), parts[1].strip()
            if not v or v.lower() in {"field", "value", "none", "null", "n/a", "-", "not visible"}:
                continue
            for field_key, syns in synonyms_map.items():
                if field_key in record and record.get(field_key) is None and isinstance(syns, list) and any(s.lower() in k for s in syns):
                    record[field_key] = v
                    break

    for line in tests_block.split("\n"):
        if not line.strip().startswith("|") or "---" in line:
            continue
        cols = [c.strip().replace("**", "") for c in line.split("|")][1:-1]
        if len(cols) < 2:
            continue
        r_name, r_val = cols[0].strip(), cols[1].strip()
        if r_name.lower() in HEADER_BLACKLIST or len(r_name) < 2:
            continue
        if r_name.lower() in {"field", "patient name", "name", "physician", "clinic", "date", "id"}:
            continue
        r_unit = cols[2].strip() if len(cols) > 2 else None
        r_ref = cols[3].strip() if len(cols) > 3 else None
        record["tests"].append({
            "raw_extracted_name": r_name,
            "result_value": r_val,
            "unit": r_unit,
            "reference_range": r_ref
        })

    return record

def phase_1_extract_vision(img_path: str) -> Dict[str, Any]:
    prompt = """Analyze this medical laboratory report image carefully.
Return a valid JSON object with these exact keys:
{
  "patient_name": "Full name of the patient or null",
  "patient_id": "Patient ID / MRN / File number or null",
  "drawn_time": "Collection or report date/time or null",
  "clinic": "Clinic, department, or hospital name or null",
  "physician": "Ordering or attending physician name or null",
  "lab_name": "Laboratory facility or panel header name or null",
  "clinical_impressions": "Verbatim clinical notes, synthesis, interpretation, or diagnostic summary at the bottom of the report, or null if none exists on the page",
  "tests": [
    {
      "raw_extracted_name": "Exact test name as written on the report (keep any parenthetical abbreviations if printed on the image)",
      "result_value": "Pure numeric or qualitative result value ONLY (do not include units or H/L/ERR status flags)",
      "unit": "Measurement unit ONLY (e.g., mg/dL, U/L, %, mmol/L, Titer) or null",
      "reference_range": "Reference range interval or null"
    }
  ]
}
CRITICAL RULES:
1. Extract EVERY single laboratory test row in the table from top to bottom (do not skip the first or last row).
2. NEVER extract table column headers (such as 'Test', 'Metric', 'Parameter', 'בדיקה', 'التحليل') as a test row.
3. Keep all original languages verbatim as printed on the report. Do NOT translate test names or clinical notes.
4. Output ONLY the raw JSON object."""

    # Attempt 1: Unconstrained JSON with think=False
    try:
        response = ollama_client.chat(
            model=MODEL_VISION,
            messages=[{"role": "user", "content": prompt, "images": [img_path]}],
            think=False,
            options={"temperature": 0.0, "num_predict": 4096, "num_ctx": 8192}
        )
        content = response["message"].get("content", "")
        raw_dict = extract_json_from_text(content)
        if raw_dict and isinstance(raw_dict.get("tests"), list) and len(raw_dict["tests"]) > 0:
            return raw_dict
    except Exception as e:
        logging.warning(f"Attempt 1 (Unconstrained JSON, think=False) warning on {os.path.basename(img_path)}: {e}")

    # Attempt 2: Constrained Schema with think=False
    try:
        response = ollama_client.chat(
            model=MODEL_VISION,
            messages=[{"role": "user", "content": prompt, "images": [img_path]}],
            format=MedicalReportSchema.model_json_schema(),
            think=False,
            options={"temperature": 0.0, "num_predict": 4096, "num_ctx": 8192}
        )
        parsed = MedicalReportSchema.model_validate_json(response["message"]["content"])
        if len(parsed.tests) > 0:
            return parsed.model_dump()
    except Exception as e2:
        logging.warning(f"Attempt 2 (Schema JSON, think=False) warning on {os.path.basename(img_path)}: {e2}")

    # Attempt 3: XML-Markdown Table Fallback with think=False
    xml_prompt = (
        "Transcribe this medical report into XML blocks:\n"
        "<metadata>\n| Field | Value |\n(Patient Name, Patient ID, Date, Clinic, Physician, Lab Name)\n</metadata>\n"
        "<lab_tests>\n| Test Name | Result | Unit | Reference Range |\n(All lab test rows)\n</lab_tests>\n"
        "<clinical_impressions>\n(Verbatim clinical notes/summary at the bottom)\n</clinical_impressions>"
    )
    try:
        response = ollama_client.chat(
            model=MODEL_VISION,
            messages=[{"role": "user", "content": xml_prompt, "images": [img_path]}],
            think=False,
            options={"temperature": 0.0, "num_predict": 4096, "num_ctx": 8192}
        )
        return parse_xml_markdown_fallback(response["message"].get("content", ""))
    except Exception as e3:
        logging.error(f"All Phase 1 attempts failed on {img_path}: {e3}")
        return {
            "patient_name": None, "patient_id": None, "drawn_time": None,
            "clinic": None, "physician": None, "lab_name": None,
            "clinical_impressions": None, "tests": []
        }

def verify_and_repair_table_alignment(img_path: str, raw_tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not raw_tests:
        return raw_tests

    first_name = clean_test_name(raw_tests[0].get("raw_extracted_name", "")).lower()
    has_header_shift = first_name in HEADER_BLACKLIST

    has_corrupted_row = False
    for t in raw_tests:
        r_name = clean_test_name(t.get("raw_extracted_name", ""))
        r_val = str(t.get("result_value") or "").strip()
        if r_val.upper() in {"-", "", "L", "H", "A", "HIGH", "LOW", "NORM", "ERR: H", "ERR: L"} or re.search(r"^\(?1:\d+\)?$", r_name):
            has_corrupted_row = True
            break

    if not has_header_shift and not has_corrupted_row:
        return raw_tests

    logging.info(f"Table alignment anomaly detected on {os.path.basename(img_path)}. Running surgical table verification...")
    repair_prompt = (
        "Look ONLY at the laboratory results table in this image.\n"
        "List every test row from top to bottom as a markdown table with 4 columns:\n"
        "| Test Name | Result Value | Unit | Reference Range |\n"
        "RULES:\n"
        "1. Do NOT include the table header row.\n"
        "2. Make sure each Test Name is paired with the exact Result Value, Unit, and Reference Range on the same horizontal row.\n"
        "3. Do NOT put status flags (like H, L, ERR: H) in the Result Value column."
    )
    try:
        resp = ollama_client.chat(
            model=MODEL_VISION,
            messages=[{"role": "user", "content": repair_prompt, "images": [img_path]}],
            think=False,
            options={"temperature": 0.0, "num_predict": 2048}
        )
        repaired = parse_xml_markdown_fallback(f"<lab_tests>\n{resp['message'].get('content', '')}\n</lab_tests>")
        rep_tests = repaired.get("tests", [])
        if len(rep_tests) >= max(1, len(raw_tests) - 1):
            return rep_tests
    except Exception as e:
        logging.warning(f"Surgical table verification warning: {e}")

    return raw_tests

def post_process_and_heal(img_path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    for k in ["patient_name", "patient_id", "drawn_time", "clinic", "physician", "lab_name", "clinical_impressions"]:
        payload[k] = fix_bidi_and_unicode(payload.get(k))

    if payload.get("patient_name"):
        payload["patient_name"] = re.sub(
            r"\s*\((?:Männlich|Weiblich|Male|Female|M|F|ذكر|أنثى|남|여|男|女)[^)]*\)",
            "",
            payload["patient_name"],
            flags=re.IGNORECASE
        ).strip()

    p_name = payload.get("patient_name") or ""
    if re.match(r"^(?:Dr\.|Prof\.|د\.|Д-р|Лікар)\s*", p_name, flags=re.IGNORECASE) or p_name.endswith(" .د"):
        if not payload.get("physician") or len(p_name) > len(payload.get("physician") or ""):
            payload["physician"] = p_name
        payload["patient_name"] = None

    if payload.get("patient_name") and payload.get("patient_name") == payload.get("patient_id"):
        if not payload["patient_name"].lower().startswith("patient"):
            payload["patient_name"] = None
        else:
            m_num = re.search(r"(\d{3,})", payload["patient_name"])
            if m_num:
                payload["patient_id"] = f"MRN-{m_num.group(1)}"

    p_id = payload.get("patient_id") or ""
    if p_id.lower().startswith("patient "):
        m_id = re.search(r"(\d{3,})", p_id)
        if m_id:
            payload["patient_id"] = f"MRN-{m_id.group(1)}"

    clinic_val = payload.get("clinic") or ""
    if not payload.get("patient_id") and re.match(r"^[A-Z]{2,5}\-\d{2,6}(?:\-[A-Z0-9]{1,4})?$", clinic_val):
        payload["patient_id"] = clinic_val
        payload["clinic"] = None

    if "|" in clinic_val:
        parts = [p.strip() for p in clinic_val.split("|") if p.strip()]
        if len(parts) == 2:
            if not payload.get("physician") and re.search(r"(?:Dr\.|Dra\.|Prof\.|BS\.|د\.|Δρ\.)", parts[1], re.I):
                payload["clinic"] = parts[0]
                payload["physician"] = parts[1]
            elif payload.get("physician") and parts[1] == payload.get("physician"):
                payload["clinic"] = parts[0]

    if not payload.get("physician") and clinic_val.startswith("Dr. "):
        m_doc = re.match(r"^(Dr\.\s+[A-Za-z\-]+)\s+(.+)$", clinic_val)
        if m_doc:
            payload["physician"] = m_doc.group(1).strip()
            payload["clinic"] = m_doc.group(2).strip()

    if payload.get("patient_name") and payload.get("physician") == payload.get("patient_name"):
        payload["physician"] = None

    raw_tests = [t for t in payload.get("tests", []) if isinstance(t, dict)]
    raw_tests = verify_and_repair_table_alignment(img_path, raw_tests)

    cleaned_tests = []
    for t in raw_tests:
        raw_name = clean_test_name(t.get("raw_extracted_name", ""))
        if not raw_name or raw_name.lower() in HEADER_BLACKLIST:
            continue
        val, unit = decouple_squashed_value_unit(str(t.get("result_value", "")), t.get("unit"))
        ref = fix_bidi_and_unicode(t.get("reference_range"))

        if val.upper() in {"-", "", "L", "H", "A", "ERR: H", "ERR: L"}:
            val = "REQUIRES_MANUAL_REVIEW"

        cleaned_tests.append({
            "raw_extracted_name": raw_name,
            "result_value": val,
            "unit": unit,
            "reference_range": ref
        })
    payload["tests"] = cleaned_tests

    if not payload.get("patient_name"):
        try:
            resp = ollama_client.chat(
                model=MODEL_VISION,
                messages=[{
                    "role": "user",
                    "content": "Look at the patient information section of this medical report. Reply ONLY with the patient's full name (do NOT output the doctor's name or ID code).",
                    "images": [img_path]
                }],
                think=False,
                options={"temperature": 0.0, "num_predict": 60}
            )
            rescued_name = fix_bidi_and_unicode(resp["message"].get("content"))
            if rescued_name and len(rescued_name) < 65 and not re.match(r"^(?:Dr\.|د\.)", rescued_name, re.I):
                payload["patient_name"] = rescued_name
        except Exception:
            pass

    if not payload.get("physician"):
        try:
            resp = ollama_client.chat(
                model=MODEL_VISION,
                messages=[{
                    "role": "user",
                    "content": "Look at this medical report for the ordering or attending physician/doctor name (often near the top or bottom signature). Reply ONLY with the doctor's name, or NONE if not present.",
                    "images": [img_path]
                }],
                think=False,
                options={"temperature": 0.0, "num_predict": 60}
            )
            rescued_doc = fix_bidi_and_unicode(resp["message"].get("content"))
            if rescued_doc and rescued_doc.upper() != "NONE" and len(rescued_doc) < 65:
                payload["physician"] = rescued_doc
        except Exception:
            pass

    nar = payload.get("clinical_impressions") or ""
    if not nar or len(nar) < 15 or nar.upper().startswith("NPI:"):
        try:
            resp = ollama_client.chat(
                model=MODEL_VISION,
                messages=[{
                    "role": "user",
                    "content": "Look at the bottom of this report for any clinical notes, synthesis, interpretation, or diagnostic summary. Transcribe it verbatim in its original language. If there is no clinical note section on the page, reply ONLY with NONE.",
                    "images": [img_path]
                }],
                think=False,
                options={"temperature": 0.0, "num_predict": 600}
            )
            rescued_nar = fix_bidi_and_unicode(resp["message"].get("content"))
            if rescued_nar and rescued_nar.upper() != "NONE" and "none visible" not in rescued_nar.lower():
                payload["clinical_impressions"] = rescued_nar
            else:
                payload["clinical_impressions"] = None
        except Exception:
            pass

    return payload

# ==============================================================================
# 6. PHASE 2: 5-STAGE 30-LANGUAGE ONTOLOGY MAPPER (CONFIG-DRIVEN)
# ==============================================================================
def get_unit_dimension(unit: Optional[str]) -> str:
    """Resolves unit dimensionality dynamically using config_rules.json."""
    if not unit:
        return "UNKNOWN"
    u_raw = unit.strip()
    if (u_raw.startswith("(") and u_raw.endswith(")")) or (u_raw.startswith("[") and u_raw.endswith("]")):
        u_raw = u_raw[1:-1].strip()
    if isinstance(CASE_SENSITIVE_TOKEN_MAP, dict) and u_raw in CASE_SENSITIVE_TOKEN_MAP:
        u = CASE_SENSITIVE_TOKEN_MAP[u_raw].lower()
    else:
        u = u_raw.lower().replace("μ", "u").replace("µ", "u")
        u = re.sub(r"\s+", " ", u).strip()
        token_map = APP_CONFIG.get("token_map", {})
        if isinstance(token_map, dict):
            u = token_map.get(u, token_map.get(u.replace(" ", ""), u))

    for dim_name, unit_set in UNIT_DIMENSIONS_MAP.items():
        if u in unit_set:
            return dim_name

    for cluster in APP_CONFIG.get("UNIT_EQUIVALENCE_CLUSTERS", []):
        if isinstance(cluster, list):
            cluster_low = {c.lower() for c in cluster if isinstance(c, str)}
            if u in cluster_low:
                for dim_name, unit_set in UNIT_DIMENSIONS_MAP.items():
                    if cluster_low.intersection(unit_set):
                        return dim_name
    return "UNKNOWN"

def is_purely_latin_query(text: str) -> bool:
    latin_chars = re.findall(r"[a-zA-Z]", text)
    has_non_latin = bool(NON_LATIN_SCRIPT_REGEX.search(text))
    return len(latin_chars) >= 2 and not has_non_latin

def get_embedding(text: str) -> Optional[np.ndarray]:
    if not text or not is_purely_latin_query(text):
        return None
    try:
        resp = ollama_client.embeddings(model=MODEL_EMBEDDING, prompt=text)
        vec = np.array(resp["embedding"], dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    except Exception:
        return None

def extract_candidate_keys(raw_name: str) -> List[str]:
    norm = (clean_test_name(raw_name) or "").lower().strip()
    candidates = []

    # 1. Full cleaned string
    full_clean = re.sub(r"[^\w\s\-\+/]", " ", norm, flags=re.UNICODE)
    full_clean = re.sub(r"\s+", " ", full_clean).strip()
    if full_clean:
        candidates.append(full_clean)
        if full_clean in CLINICAL_ACRONYM_EXPANSIONS:
            candidates.append(CLINICAL_ACRONYM_EXPANSIONS[full_clean])
    full_no_hyphen = full_clean.replace("-", " ").strip()
    if full_no_hyphen and full_no_hyphen != full_clean:
        candidates.append(full_no_hyphen)

    # 1b. Strip Nordic/Finnish/European specimen prefixes (S-, P-, B-, U-, fP-, fS-, fB-, Liq-)
    prefix_stripped = re.sub(r"^(?:fp|fs|fb|s|p|b|u|liq|csv)\s*\-\s*", "", full_clean, flags=re.IGNORECASE).strip()
    if prefix_stripped and prefix_stripped != full_clean and len(prefix_stripped) >= 2:
        candidates.append(prefix_stripped)
        if prefix_stripped in CLINICAL_ACRONYM_EXPANSIONS:
            candidates.append(CLINICAL_ACRONYM_EXPANSIONS[prefix_stripped])

    # 2. Strip config_rules.json semantic_noise patterns
    noise_stripped = norm
    for pat in SEMANTIC_NOISE_CFG.get("noise_patterns", []):
        noise_stripped = re.sub(pat, " ", noise_stripped, flags=re.IGNORECASE)
    noise_stripped = re.sub(r"[^\w\s\-]", " ", noise_stripped, flags=re.UNICODE)
    noise_stripped = re.sub(r"\s+", " ", noise_stripped).strip()
    if noise_stripped and len(noise_stripped) >= 2:
        candidates.append(noise_stripped)
        if noise_stripped in CLINICAL_ACRONYM_EXPANSIONS:
            candidates.append(CLINICAL_ACRONYM_EXPANSIONS[noise_stripped])

    # 3. Parenthetical acronyms or English terms
    for pm in re.findall(r"[\(\[（]([^)\]）]+)[\)\]）]", norm):
        clean_pm = re.sub(r"[^\w\s\-\+/]", " ", pm, flags=re.UNICODE).strip()
        if clean_pm and clean_pm not in NON_ACRONYM_WORDS and len(clean_pm) >= 1:
            candidates.append(clean_pm)
            if clean_pm in CLINICAL_ACRONYM_EXPANSIONS:
                candidates.append(CLINICAL_ACRONYM_EXPANSIONS[clean_pm])

    # 4. Text outside parentheses (preserving hyphens so 'Anti-HBs' stays intact)
    outside = re.sub(r"[\(\[（][^)\]）]*[\)\]）]", " ", norm)
    clean_out_hyph = re.sub(r"[^\w\s\-]", " ", outside, flags=re.UNICODE)
    clean_out_hyph = re.sub(r"\s+", " ", clean_out_hyph).strip()
    if clean_out_hyph and len(clean_out_hyph) >= 1:
        candidates.append(clean_out_hyph)
        if clean_out_hyph in CLINICAL_ACRONYM_EXPANSIONS:
            candidates.append(CLINICAL_ACRONYM_EXPANSIONS[clean_out_hyph])
    clean_out = clean_out_hyph.replace("-", " ").strip()
    if clean_out and clean_out != clean_out_hyph:
        candidates.append(clean_out)
        if clean_out in CLINICAL_ACRONYM_EXPANSIONS:
            candidates.append(CLINICAL_ACRONYM_EXPANSIONS[clean_out])

    # 4b. Noise-stripped text outside parentheses
    out_noise_stripped = clean_out_hyph
    for pat in SEMANTIC_NOISE_CFG.get("noise_patterns", []):
        out_noise_stripped = re.sub(pat, " ", out_noise_stripped, flags=re.IGNORECASE)
    out_noise_stripped = re.sub(r"\s+", " ", out_noise_stripped).strip()
    if out_noise_stripped and len(out_noise_stripped) >= 2 and out_noise_stripped not in candidates:
        candidates.append(out_noise_stripped)
        if out_noise_stripped in CLINICAL_ACRONYM_EXPANSIONS:
            candidates.append(CLINICAL_ACRONYM_EXPANSIONS[out_noise_stripped])

    # 5. Mixed script Latin token extraction
    has_non_latin = bool(NON_LATIN_SCRIPT_REGEX.search(norm))
    if has_non_latin:
        latin_tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9\-\+]{1,18}\b", outside)
        if latin_tokens:
            joined_lat = " ".join(t.lower() for t in latin_tokens)
            if joined_lat not in NON_ACRONYM_WORDS:
                candidates.append(joined_lat)
                if joined_lat in CLINICAL_ACRONYM_EXPANSIONS:
                    candidates.append(CLINICAL_ACRONYM_EXPANSIONS[joined_lat])

    # 6. Lexicon-grounded transliteration candidate
    if has_non_latin or detect_script_language(clean_out):
        translit = transliterate_via_lexicon(clean_out)
        if translit and translit.lower() != clean_out:
            candidates.append(translit.lower())

    # 7. Reversed RTL string check
    if re.search(r"[\u0590-\u05FF\u0600-\u06FF]", clean_out):
        candidates.append(clean_out[::-1])
        if BIDI_SUPPORT:
            try:
                reshaped = get_display(arabic_reshaper.reshape(clean_out))
                candidates.append(unicodedata.normalize("NFKC", reshaped).lower().strip())
            except Exception:
                pass

    return list(dict.fromkeys([c for c in candidates if c]))

def enforce_critical_token_fidelity(raw_name: str, resolved_name: str) -> str:
    """
    Prevents LLM translation drift on critical immunoglobulin classes (IgM vs IgG vs IgA)
    and lipoprotein fractions (VLDL vs LDL).
    """
    r_low = raw_name.lower()
    res = resolved_name
    if "igm" in r_low and "igg" not in r_low and "igg" in res.lower():
        res = re.sub(r"(?i)\bigg\b", "IgM", res)
    elif "igg" in r_low and "igm" not in r_low and "igm" in res.lower():
        res = re.sub(r"(?i)\bigm\b", "IgG", res)
    if "vldl" in r_low and "vldl" not in res.lower() and "very low" not in res.lower():
        res = "VLDL Cholesterol"
    return res

def batch_translate_to_english(foreign_names: List[str]) -> Dict[str, str]:
    if not foreign_names:
        return {}
    prompt = (
        "Translate each medical laboratory test name below (from any of the 30 global languages: European, Slavic, Greek, Arabic, Hebrew, Hindi, Thai, Vietnamese, Chinese, Japanese, Korean, Turkish, Indonesian) into its standard English clinical analyte name.\n"
        "CRITICAL RULES:\n"
        "1. Preserve specimen qualifiers (e.g., 'in CSF', 'in Urine', 'in 24h Urine', 'in Stool', 'in Sweat', 'in Sputum').\n"
        "2. Preserve clinical subfractions ('Free', 'Direct', 'Indirect', 'IgM', 'IgG', 'IgA', 'High Sensitivity').\n"
        "Examples:\n"
        "Ουρία Αίματος => Blood Urea Nitrogen\n"
        "Σάκχαρο => Glucose\n"
        "Γλυκόζη Νηστείας => Glucose\n"
        "Χοληρυθρίνη Άμεση => Direct Bilirubin\n"
        "Κρεατινίνη => Creatinine\n"
        "Γαλακτική αφυδρογονάση (LDH) => Lactate Dehydrogenase\n"
        "Κρεατίνης Κινάση (CPK) => Creatine Kinase\n"
        "Σίδηρος (Fe) => Iron\n"
        "Κάλιο Ορού (K) => Potassium\n"
        "Φερριτίνη Ορού => Ferritin\n"
        "Κορεσμός Τρανσφερρίνης => Transferrin Saturation\n"
        "Viteza de sedimentare a hematiilor (VSH) => Erythrocyte Sedimentation Rate\n"
        "Plaquettes => Platelets\n"
        "Željezo (Fe) => Iron\n"
        "البيليروبين الكلي => Total Bilirubin\n"
        "صميم البروتين الشحمي B => Apolipoprotein B\n\n"
        "Output ONLY lines in the exact format: ORIGINAL_TERM => ENGLISH_TERM\n\n"
        "Terms to translate:\n" + "\n".join(foreign_names)
    )
    translations = {}
    try:
        resp = ollama_client.chat(
            model=MODEL_VISION,
            messages=[{"role": "user", "content": prompt}],
            think=False,
            options={"temperature": 0.0, "num_predict": 1024}
        )
        content = resp["message"].get("content", "")
        for line in content.split("\n"):
            if "=>" in line:
                parts = line.split("=>", 1)
                src = parts[0].strip().strip("-*•\"'")
                eng = parts[1].strip().strip("\"'")
                if src and eng:
                    translations[src] = enforce_critical_token_fidelity(src, eng)
    except Exception as e:
        logging.warning(f"Translation bridge warning: {e}")
    return translations

class Phase2OntologyMapper:
    _instance = None

    @classmethod
    def get_instance(cls, ontology_path: str):
        if cls._instance is None:
            cls._instance = cls(load_json(ontology_path))
        return cls._instance

    def __init__(self, ontology: Dict[str, Dict[str, Any]]):
        self.ontology = ontology
        self.metadata = []
        self.exact_map: Dict[str, List[Dict[str, Any]]] = {}
        vectors = []

        logging.info("Building Phase 2 Hybrid Ontology Index (seed_ontology.json + Latin FAISS)...")
        for loinc, data in ontology.items():
            if not isinstance(data, dict):
                continue
            canonical = data.get("canonical_name", loinc)
            allowed_dims = {get_unit_dimension(u) for u in data.get("allowed_units", []) if get_unit_dimension(u) != "UNKNOWN"}
            base_dim = next(iter(allowed_dims), "UNKNOWN")

            triggers = list(set(data.get("triggers", []) + [canonical]))
            for trigger in triggers:
                norm_t = (fix_bidi_and_unicode(trigger) or "").lower().strip()
                clean_t = re.sub(r"[^\w\s\-\+/]", " ", norm_t, flags=re.UNICODE)
                clean_t = re.sub(r"\s+", " ", clean_t).strip()
                if not clean_t:
                    continue

                payload = {
                    "loinc_code": loinc,
                    "canonical_name": canonical,
                    "dimension": base_dim,
                    "allowed_dims": allowed_dims,
                    "trigger": clean_t
                }
                self.exact_map.setdefault(clean_t, []).append(payload)
                no_hyph = clean_t.replace("-", " ").strip()
                if no_hyph != clean_t:
                    self.exact_map.setdefault(no_hyph, []).append(payload)

                vec = get_embedding(clean_t)
                if vec is not None:
                    vectors.append(vec)
                    self.metadata.append(payload)

        if vectors and FAISS_SUPPORT:
            self.index = faiss.IndexFlatIP(len(vectors[0]))
            self.index.add(np.array(vectors, dtype=np.float32))
            logging.info(f"Seed FAISS index ready with {len(vectors)} verified Latin trigger vectors.")
        else:
            self.index = None

    @staticmethod
    def _is_dim_compatible(ext_dim: str, payload: Dict[str, Any]) -> bool:
        if ext_dim == "UNKNOWN":
            return True
        allowed = payload.get("allowed_dims", set())
        if not allowed:
            return True
        return ext_dim in allowed

    def _violates_subtype_guard(self, query: str, target: str) -> bool:
        q, t = query.lower(), target.lower()

        # 1. Genetic mutations vs standard serum proteins (removed 'receptor' so AChR Ab is never blocked)
        mutation_markers = [
            "mutation", "mutasyonu", "mutacja", "mutácia", "genmutatie", "genetski",
            "ةرفط", "طفرة", "يتازرو", "وراثي", "מוטציה", "allele", "allel"
        ]
        if any(w in q for w in mutation_markers) and not any(w in t for w in mutation_markers):
            return True

        # 2. Viral / bacterial / parasite specific antibodies must not collide with Total IgG/IgA/IgM/IgE
        pathogen_ab_markers = [
            "vca", "ea-iga", "ebv", "eb病毒", "dengue", "chikungunya", "zikv", "zika",
            "brucella", "borrelia", "leptospira", "trypanosoma", "cruzi", "rujeolă",
            "rubella", "measles", "hiv", "vih", "cmv", "cytomegalovirus", "toxoplasma",
            "treponema", "syphilis", "parvovirus", "varicella", "herpes", "hsv",
            "mumps", "helicobacter", "chlamydia", "mycoplasma", "sars-cov-2", "mers",
            "orientia", "malaria", "pneumocystis", "shiga"
        ]
        if any(w in q for w in pathogen_ab_markers) and not any(w in t for w in pathogen_ab_markers):
            return True

        # 3. ANA Pattern / Model ANA vs Antinuclear Antibodies (ANA)
        if ("pattern" in q or "model" in q) and "pattern" not in t and "model" not in t:
            return True

        # 4. Hemoglobin variants (HbA, HbS, HbF, HbA2, HbA1c) vs generic Hemoglobin
        if "hbsag" not in q and "anti-hbs" not in q and "anti hbs" not in q:
            has_hb_variant_q = bool(
                re.search(r"\b(?:hgb|hb|hemoglobin|hemoglobina|hémoglobine|hämoglobin|המוגלובין|המולובין)[\s\-]+(?:a1c|a2|a|s|f)\b", q)
                or re.search(r"\bhb(?:a1c|a2|s|f)\b", q)
            )
            has_hb_variant_t = bool(
                re.search(r"\b(?:a1c|a2|a|s|f)\b", t)
                or re.search(r"\bhb(?:a1c|a2|s|f)\b", t)
            )
            if has_hb_variant_q and not has_hb_variant_t:
                return True

        # 5. Lipoprotein fractions (LDL / HDL / VLDL) vs Total Cholesterol
        for lipo in ["vldl", "ldl", "hdl"]:
            if re.search(rf"\b{lipo}\b", q) and not re.search(rf"\b{lipo}\b", t):
                return True

        # 6. Direct / Indirect Bilirubin vs Total Bilirubin
        if ("direct" in q or "diretta" in q or "direta" in q or "άμεση" in q or "αμεση" in q or "مباشر" in q or "прямий" in q or "прямой" in q or "直接" in q or "प्रत्यक्ष" in q) and not any(w in t for w in ["direct", "diretta", "direta", "άμεση", "αμεση", "مباشر"]):
            return True
        if ("indirect" in q or "غير المباشر" in q or "لمياشر" in q or "ريغ" in q or "έμμεση" in q or "непрямий" in q or "间接" in q) and not any(w in t for w in ["indirect", "غير المباشر", "لمياشر", "έμμεση"]):
            return True

        # 7. Urine Albumin/Creatinine or Protein/Creatinine ratio vs Serum Creatinine/Albumin
        if ("albumin/creatinine" in q or "ratio" in q or "acr" in q or "upcr" in q or "tỷ lệ" in q) and not any(w in t for w in ["ratio", "albumin/creatinine", "microalbumin", "acr", "upcr"]):
            return True

        # 8. LOINC Timing Axis Guard: Spot/Random Urine vs 24h Urine
        if ("spot" in q or "random" in q) and ("24h" in t or "24 hour" in t):
            return True

        # 9. LOINC System Axis Guard: Urine, CSF, Stool, Sweat & Sputum must not collide with Blood/Serum targets
        urine_markers = SPECIMEN_SYSTEM_MARKERS.get("urine", ["urine", "urin", "urina", "idrar", "בשתן"])
        csf_markers = SPECIMEN_SYSTEM_MARKERS.get("csf", ["csf", "spinalvæske", "liquor", "likvor", "dnt"])
        stool_markers = SPECIMEN_SYSTEM_MARKERS.get("stool", ["stool", "fecal", "faecal", "fäkales", "fækal", "stolici", "feci"])
        sweat_markers = SPECIMEN_SYSTEM_MARKERS.get("sweat", ["sweat", "svedtest", "pote", "potný", "potny"])
        sputum_markers = SPECIMEN_SYSTEM_MARKERS.get("sputum", ["sputum", "قشع", "喀痰", "객담", "เสมหะ"])

        q_is_urine = any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", q) for m in urine_markers)
        t_is_urine = any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", t) for m in urine_markers + ["microalbumin"])
        if q_is_urine and not t_is_urine:
            return True

        q_is_csf = any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", q) for m in csf_markers)
        t_is_csf = any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", t) for m in csf_markers)
        if q_is_csf and not t_is_csf:
            return True

        q_is_stool = any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", q) for m in stool_markers)
        t_is_stool = any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", t) for m in stool_markers + ["calprotectin", "elastase", "shiga"])
        if q_is_stool and not t_is_stool:
            return True

        q_is_sweat = any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", q) for m in sweat_markers)
        t_is_sweat = any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", t) for m in sweat_markers)
        if q_is_sweat and not t_is_sweat:
            return True

        q_is_sputum = any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", q) for m in sputum_markers)
        t_is_sputum = any(re.search(rf"(?<!\w){re.escape(m)}(?!\w)", t) for m in sputum_markers + ["mers-cov", "pseudomonas"])
        if q_is_sputum and not t_is_sputum:
            return True

        # 10. Free vs Total hormone subtypes
        if ("free" in q or "wolna" in q or "fritt" in q or "ελεύθερη" in q or "свободный" in q or "вільний" in q or "游离" in q or "חופשי" in q) and not any(w in t for w in ["free", "ft4", "ft3", "wolna", "fritt", "ελεύθερη", "חופשי"]):
            return True
        return False

    @staticmethod
    def _sparse_jaccard(s1: str, s2: str) -> float:
        t1, t2 = set(s1.lower().split()), set(s2.lower().split())
        if not t1 or not t2:
            return 0.0
        return len(t1.intersection(t2)) / float(len(t1.union(t2)))

    def _try_match_candidates(self, candidates: List[str], full_context_name: str, ext_dim: str) -> Optional[Dict[str, Any]]:
        for cand in candidates:
            if cand in self.exact_map:
                for hit in self.exact_map[cand]:
                    if not self._violates_subtype_guard(full_context_name, hit["trigger"]):
                        if self._is_dim_compatible(ext_dim, hit):
                            return {
                                "loinc_code": hit["loinc_code"],
                                "canonical_name": hit["canonical_name"],
                                "similarity_score": 1.0,
                                "match_type": "EXACT_KEYWORD"
                            }

        for cand in candidates:
            if not is_purely_latin_query(cand):
                continue
            best_j, best_hit = 0.0, None
            for trig, payload_list in self.exact_map.items():
                if not is_purely_latin_query(trig):
                    continue
                score = self._sparse_jaccard(cand, trig)
                if score >= 0.75 and score > best_j:
                    for payload in payload_list:
                        if not self._violates_subtype_guard(full_context_name, trig):
                            if self._is_dim_compatible(ext_dim, payload):
                                best_j, best_hit = score, payload
                                break
            if best_hit:
                return {
                    "loinc_code": best_hit["loinc_code"],
                    "canonical_name": best_hit["canonical_name"],
                    "similarity_score": round(best_j, 4),
                    "match_type": "SPARSE_OVERLAP"
                }

        if self.index is not None:
            for cand in candidates:
                vec = get_embedding(cand)
                if vec is None:
                    continue
                k_search = min(3, len(self.metadata))
                distances, indices = self.index.search(np.array([vec], dtype=np.float32), k_search)
                best_sim, best_hit = 0.0, None
                for sim, idx in zip(distances[0], indices[0]):
                    hit = self.metadata[idx]
                    sim_f = float(sim)
                    if sim_f >= 0.86 and sim_f > best_sim:
                        if not self._violates_subtype_guard(full_context_name, hit["trigger"]):
                            if self._is_dim_compatible(ext_dim, hit):
                                best_sim, best_hit = sim_f, hit
                if best_hit:
                    return {
                        "loinc_code": best_hit["loinc_code"],
                        "canonical_name": best_hit["canonical_name"],
                        "similarity_score": round(best_sim, 4),
                        "match_type": "BRIDGED_VECTOR"
                    }
        return None

    def map_tests(self, tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unmapped_indices = []

        for idx, t in enumerate(tests):
            raw_name = t.get("raw_extracted_name", "")
            ext_dim = get_unit_dimension(t.get("unit"))
            t.update({
                "loinc_code": "UNMAPPED",
                "canonical_name": raw_name,
                "similarity_score": 0.0,
                "match_type": "NONE"
            })

            candidates = extract_candidate_keys(raw_name)
            match_res = self._try_match_candidates(candidates, raw_name, ext_dim)
            if match_res:
                t.update(match_res)
            elif is_non_laboratory_observation(raw_name, t.get("unit")):
                t.update({
                    "loinc_code": "NON_LAB_OBSERVATION",
                    "canonical_name": raw_name,
                    "similarity_score": 0.0,
                    "match_type": "EXCLUDED_NON_LAB"
                })
            else:
                unmapped_indices.append(idx)

        if unmapped_indices:
            to_translate = [tests[i]["raw_extracted_name"] for i in unmapped_indices if tests[i].get("raw_extracted_name")]
            translations = batch_translate_to_english(to_translate)

            for i in unmapped_indices:
                raw_name = tests[i].get("raw_extracted_name", "")
                eng_name = translations.get(raw_name)
                if eng_name:
                    if is_non_laboratory_observation(eng_name, tests[i].get("unit")):
                        tests[i].update({
                            "loinc_code": "NON_LAB_OBSERVATION",
                            "canonical_name": raw_name,
                            "similarity_score": 0.0,
                            "match_type": "EXCLUDED_NON_LAB"
                        })
                        continue
                    ext_dim = get_unit_dimension(tests[i].get("unit"))
                    eng_candidates = extract_candidate_keys(eng_name)
                    combined_context = f"{raw_name} {eng_name}"
                    match_res = self._try_match_candidates(eng_candidates, combined_context, ext_dim)
                    if match_res:
                        tests[i].update(match_res)

        return tests

# ==============================================================================
# 7. PHASE 3: NULL-SAFE EVALUATION METRICS ENGINE
# ==============================================================================
class Phase3Evaluator:
    @staticmethod
    def evaluate(enriched_data: Dict[str, Any], base_filename: str) -> Dict[str, Any]:
        tests = enriched_data.get("tests", [])
        total_tests = len(tests)

        meta_keys = ["patient_name", "patient_id", "drawn_time", "clinic", "physician", "lab_name"]
        populated_meta = sum(1 for k in meta_keys if (enriched_data.get(k) or "").strip() not in {"", "-", "null", "None"})
        metadata_completeness_pct = round((populated_meta / len(meta_keys)) * 100, 2)

        narrative_str = (enriched_data.get("clinical_impressions") or "").strip()
        has_narrative = bool(narrative_str and narrative_str.lower() not in {"-", "none", "null"})

        tests_with_valid_values = sum(
            1 for t in tests if (t.get("result_value") or "").strip() not in {"", "-", "REQUIRES_MANUAL_REVIEW"}
        )
        tests_with_units = sum(1 for t in tests if (t.get("unit") or "").strip() not in {"", "-"})

        exact_matches = sum(1 for t in tests if t.get("match_type") == "EXACT_KEYWORD")
        sparse_matches = sum(1 for t in tests if t.get("match_type") == "SPARSE_OVERLAP")
        vector_matches = sum(1 for t in tests if t.get("match_type") == "BRIDGED_VECTOR")
        excluded_non_lab_matches = sum(1 for t in tests if t.get("loinc_code") == "NON_LAB_OBSERVATION")
        unmapped_matches = sum(1 for t in tests if t.get("loinc_code") == "UNMAPPED")
        mapped_count = exact_matches + sparse_matches + vector_matches

        # Laboratory-specific denominator excludes non-lab imaging/vitals rows
        total_lab_tests = max(0, total_tests - excluded_non_lab_matches)
        mapping_accuracy_pct = (
            round((mapped_count / total_lab_tests) * 100, 2)
            if total_lab_tests > 0
            else (100.0 if total_tests > 0 else 0.0)
        )
        text_recon_pct = round((tests_with_valid_values / total_tests) * 100, 2) if total_tests > 0 else 0.0

        scores = [
            t.get("similarity_score", 0.0)
            for t in tests
            if t.get("loinc_code") not in {"UNMAPPED", "NON_LAB_OBSERVATION"}
        ]
        mean_similarity_score = round(float(np.mean(scores)), 4) if scores else 0.0

        overall_pipeline_accuracy = round(
            (text_recon_pct * 0.4) + (mapping_accuracy_pct * 0.4) + (metadata_completeness_pct * 0.2), 2
        )

        detected_problems = []
        if total_tests == 0:
            detected_problems.append("CRITICAL: Zero laboratory tests extracted (Phase 1 collapse).")
        if metadata_completeness_pct < 50.0:
            detected_problems.append("Low metadata completeness (<50%).")
        if not has_narrative:
            detected_problems.append("Clinical impressions narrative is empty or missing.")
        if unmapped_matches > 0:
            detected_problems.append(f"{unmapped_matches} unmapped laboratory analyte(s) queued for ontology_assistant.py expansion.")

        return {
            "filename": base_filename,
            "evaluation_summary": {
                "total_tests_extracted": total_tests,
                "total_lab_tests_evaluated": total_lab_tests,
                "total_tests_mapped": mapped_count,
                "total_tests_unmapped": unmapped_matches,
                "total_non_lab_excluded": excluded_non_lab_matches,
                "phase_1_text_reconciliation_pct": text_recon_pct,
                "phase_1_metadata_completeness_pct": metadata_completeness_pct,
                "phase_2_loinc_mapping_accuracy_pct": mapping_accuracy_pct,
                "overall_pipeline_accuracy_pct": overall_pipeline_accuracy,
                "mean_mapping_similarity_score": mean_similarity_score,
                "has_clinical_narrative": has_narrative
            },
            "match_type_breakdown": {
                "exact_keyword_count": exact_matches,
                "sparse_overlap_count": sparse_matches,
                "bridged_vector_count": vector_matches,
                "excluded_non_lab_count": excluded_non_lab_matches,
                "unmapped_count": unmapped_matches
            },
            "data_quality_checks": {
                "tests_with_numeric_or_valid_values": tests_with_valid_values,
                "tests_with_units": tests_with_units,
                "populated_metadata_fields": populated_meta,
                "total_metadata_fields_evaluated": len(meta_keys)
            },
            "diagnosed_problems": detected_problems
        }

# ==============================================================================
# 8. PIPELINE EXECUTION (PHASE 0 -> PHASE 1 -> PHASE 2 -> PHASE 3)
# ==============================================================================
@celery_app.task(name="process_medical_document")
def process_document_task(file_path: str):
    base_filename = os.path.splitext(os.path.basename(file_path))[0]
    logging.info(f"\n==========================================")
    logging.info(f"Processing AegisParse Pipeline: {base_filename}")
    logging.info(f"==========================================")

    # PHASE 0: Visual Out-of-Distribution (OOD) Gatekeeper
    gatekeeper_res = phase_0_verify_medical_lab_image(file_path)
    if not gatekeeper_res.get("is_medical_lab_report", True):
        reason = gatekeeper_res.get("detected_content", "Non-medical or non-laboratory image")
        logging.warning(
            f"[PHASE 0 REJECTED] '{base_filename}' is NOT a medical laboratory report "
            f"(Verdict: {gatekeeper_res.get('verdict')} | Detected: {reason}). Skipping pipeline."
        )
        rejected_file = os.path.join(JSON_DIR, f"{base_filename}_0_rejected.json")
        with open(rejected_file, "w", encoding="utf-8") as f:
            json.dump({
                "filename": base_filename,
                "is_medical_lab_report": False,
                "status": "REJECTED_NON_MEDICAL_LAB_IMAGE",
                "gatekeeper_verdict": gatekeeper_res.get("verdict", "NO"),
                "detected_content": reason
            }, f, indent=4, ensure_ascii=False)
        return

    logging.info(f"[PHASE 0 PASSED] Verified medical laboratory report: {gatekeeper_res.get('detected_content')}")

    mapper = Phase2OntologyMapper.get_instance(ONTOLOGY_FILE)

    raw_payload = phase_1_extract_vision(file_path)
    healed_payload = post_process_and_heal(file_path, raw_payload)
    healed_payload["tests"] = mapper.map_tests(healed_payload.get("tests", []))
    evaluation_metrics = Phase3Evaluator.evaluate(healed_payload, base_filename)

    enriched_file = os.path.join(JSON_DIR, f"{base_filename}_2_enriched.json")
    with open(enriched_file, "w", encoding="utf-8") as f:
        json.dump(healed_payload, f, indent=4, ensure_ascii=False)

    evaluation_file = os.path.join(JSON_DIR, f"{base_filename}_3_evaluation.json")
    with open(evaluation_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_metrics, f, indent=4, ensure_ascii=False)

    logging.info(
        f"Completed {base_filename}: "
        f"{evaluation_metrics['evaluation_summary']['total_tests_extracted']} tests extracted, "
        f"{evaluation_metrics['evaluation_summary']['phase_2_loinc_mapping_accuracy_pct']}% LOINC mapped, "
        f"Overall Score: {evaluation_metrics['evaluation_summary']['overall_pipeline_accuracy_pct']}%."
    )

if __name__ == "__main__":
    Phase2OntologyMapper.get_instance(ONTOLOGY_FILE)
    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    target_files = sorted({
        f for ext in image_extensions
        for f in glob.glob(os.path.join(IMAGES_DIR, ext))
    })
    logging.info(f"Found {len(target_files)} target images in {IMAGES_DIR}.")
    for f in target_files:
        process_document_task.delay(f)