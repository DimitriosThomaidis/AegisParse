import os
import glob
import json
import logging
import re
import unicodedata
import ollama
import torch
import faiss
import numpy as np
from typing import Optional, Dict, List, Any
from transformers import AutoTokenizer, AutoModel

# ==============================================================================
# CONFIGURATION & PATHS (SINGLE SOURCE OF TRUTH VIA config_rules.json)
# ==============================================================================
BASE_DIR = os.getenv("AEGIS_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
JSON_DIR = os.getenv("AEGIS_JSON_DIR", os.path.join(BASE_DIR, "Json_Outputs"))
PENDING_FILE = os.path.join(BASE_DIR, "pending_approvals.json")
SEED_ONTOLOGY_FILE = os.path.join(BASE_DIR, "seed_ontology.json")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "loinc_index.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "loinc_metadata.json")
LEXICONS_FILE = os.path.join(BASE_DIR, "clinical_lexicons.json")
CONFIG_RULES_FILE = os.path.join(BASE_DIR, "config_rules.json")

MODEL_TRANSLATOR = os.getenv("AEGIS_MODEL_VISION", "qwen3.5:4b")
MODEL_FALLBACK_TRANSLATOR = os.getenv("AEGIS_MODEL_FALLBACK", "medgemma:4b")
SAPBERT_MODEL = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")

ollama_client = ollama.Client(host=OLLAMA_HOST, timeout=180.0)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

# ==============================================================================
# 1. DYNAMIC CONFIG & LEXICON LOADER (100% SYNCHRONIZED WITH config_rules.json)
# ==============================================================================
def load_json(filepath: str) -> dict:
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    logging.warning(f"Configuration file {filepath} not found.")
    return {}

LEXICONS = load_json(LEXICONS_FILE)
APP_CONFIG = load_json(CONFIG_RULES_FILE)

SEMANTIC_NOISE_CFG = APP_CONFIG.get("semantic_noise", {})
if not isinstance(SEMANTIC_NOISE_CFG, dict):
    SEMANTIC_NOISE_CFG = {}

NON_ANALYTE_PARENS: set = {
    w.lower() for w in SEMANTIC_NOISE_CFG.get("non_analyte_parentheticals", [
        "high", "low", "normal", "abnormal", "ifa", "lba", "elisa", "iha", "pcr", "real-time pcr",
        "allele 1", "allele 2", "allel 1", "allel 2", "arn", "rna", "dna", "igg/iga", "microscopic",
        "calc", "nih", "calculated", "ifcc", "hplc", "igm", "igg", "iga", "j3", "j21",
        "מוטציה", "פעילות", "חופשי", "وراثي", "يتازرو", "سنة"
    ])
}

PARENTHETICAL_QUALIFIER_BLACKLIST: set = NON_ANALYTE_PARENS | {
    "liquor", "likvor", "spinalvæske", "spinalvaeske", "csf", "lcr", "dnt",
    "serum", "sérum", "soro", "suero", "siero", "surowica", "plasma", "blood", "sangre", "sangue", "sang",
    "urin", "urina", "urine", "orina", "idrar", "spot", "24h", "stolici", "stolica", "stool", "feci",
    "sputum", "potný test", "potny test", "serologie", "mielogram", "sistema bactec", "striscio di sangue",
    "exudat faringian", "pcr exudat faringian", "sangre periférica", "sang périphérique", "ecocardiograma",
    "kaudatusatrofi", "f508del", "typ 2", "who type iii"
}

ACRONYM_EXPANSIONS_RAW = APP_CONFIG.get("clinical_acronym_expansions", {})
CLINICAL_ACRONYM_EXPANSIONS: Dict[str, str] = {
    k.lower(): v
    for k, v in (ACRONYM_EXPANSIONS_RAW.items() if isinstance(ACRONYM_EXPANSIONS_RAW, dict) else [])
    if isinstance(v, str)
}

# Official LOINC 6-Axis search terms override display names for maximum SapBERT cosine similarity
FALLBACK_ACRONYM_EXPANSIONS: Dict[str, str] = {
    "afp": "Alpha-1-fetoprotein",
    "alpha-fetoprotein, afp": "Alpha-1-fetoprotein",
    "cea": "Carcinoembryonic Ag",
    "ca 125": "Cancer Ag 125",
    "amh": "Mullerian inhibiting substance",
    "prl": "Prolactin",
    "acth": "Corticotropin",
    "dhea-s": "Dehydroepiandrosterone sulfate",
    "nt-probnp": "Natriuretic peptide.B prohormone N-Terminal",
    "hs-ctnt": "Troponin T.cardiac",
    "ipth": "Parathyrin.intact",
    "saa": "Serum amyloid A",
    "il-6": "Interleukin 6",
    "nfl": "Neurofilament light polypeptide",
    "achr": "Acetylcholine receptor Ab",
    "achr-ak": "Acetylcholine receptor Ab",
    "musk": "Muscle specific kinase Ab",
    "musk-ak": "Muscle specific kinase Ab",
    "homa-ir": "Homeostasis model assessment",
    "pivka-ii": "Acarboxyprothrombin",
    "vldl cholesterol cal": "Cholesterol in VLDL [Mass/volume] in Serum or Plasma",
    "estimated average glucose (eag)": "Glucose mean value [Mass/volume] in Blood Estimated from glycated hemoglobin",
    "direct antiglobulin test (dat/coombs)": "Direct antiglobulin test [Interpretation] in Red Blood Cells",
    "dengue igm": "Dengue virus IgM Ab [Presence] in Serum",
    "dengue igg": "Dengue virus IgG Ab [Presence] in Serum",
    "amônia sérica": "Ammonia [Moles/volume] in Plasma",
    "ammonio (ammoniemia)": "Ammonia [Moles/volume] in Plasma",
    "zellzahl (liquor)": "Leukocytes [#/volume] in Cerebral spinal fluid",
    "leukocytter (spinalvæske)": "Leukocytes [#/volume] in Cerebral spinal fluid",
    "tế bào học (csf wbc)": "Leukocytes [#/volume] in Cerebral spinal fluid",
    "glukose (spinalvæske)": "Glucose [Moles/volume] in Cerebral spinal fluid",
    "glucose dnt": "Glucose [Moles/volume] in Cerebral spinal fluid",
    "laktat (liquor)": "Lactate [Moles/volume] in Cerebral spinal fluid",
    "total protein (spinalvæske)": "Protein [Mass/volume] in Cerebral spinal fluid",
    "Δ4-ανδροσενδιόνη": "Androstenedione [Mass/volume] in Serum or Plasma",
    "δ4-ανδροσενδιόνη": "Androstenedione [Mass/volume] in Serum or Plasma",
    "anticorpi igm rujeolă (serologie)": "Measles virus IgM Ab [Units/volume] in Serum",
    "virus rujeolic (pcr exudat faringian)": "Measles virus RNA [Presence] in Specimen by NAA with probe detection"
}
for k_fb, v_fb in FALLBACK_ACRONYM_EXPANSIONS.items():
    CLINICAL_ACRONYM_EXPANSIONS[k_fb.lower()] = v_fb

UNIT_DIMENSIONS_RAW = APP_CONFIG.get("unit_dimensions", {})
UNIT_DIMENSIONS_MAP: Dict[str, set] = {
    dim: {u.lower() for u in units}
    for dim, units in (UNIT_DIMENSIONS_RAW.items() if isinstance(UNIT_DIMENSIONS_RAW, dict) else [])
    if isinstance(units, list)
}
CASE_SENSITIVE_TOKEN_MAP: Dict[str, str] = (
    APP_CONFIG.get("case_sensitive_token_map", {})
    if isinstance(APP_CONFIG.get("case_sensitive_token_map"), dict)
    else {}
)
TOKEN_MAP: Dict[str, str] = (
    APP_CONFIG.get("token_map", {})
    if isinstance(APP_CONFIG.get("token_map"), dict)
    else {}
)
SPECIMEN_SYSTEM_MARKERS: Dict[str, List[str]] = (
    APP_CONFIG.get("specimen_system_markers", {})
    if isinstance(APP_CONFIG.get("specimen_system_markers"), dict)
    else {}
)

DEFAULT_NON_LAB_MARKERS = [
    r"\bx\-?ray\b", r"\bradiograf", r"\bröntgen\b", r"\brtg\b", r"\bрентген", r"\bακτινογραφία",
    r"\bmri\b", r"\birm\b", r"\bмагнитно\-резонанс", r"\bμαγνητική\s+τομογραφία", r"\bkaudatusatrofi",
    r"\bct\s+scan\b", r"\btomograf", r"\bкомпьютерная\s+томограф", r"\bαξονική\s+τομογραφία",
    r"\bultrasound\b", r"\becograf", r"\bechograf", r"\béchographie\b", r"\busg\b", r"\bузи\b", r"\bυπερηχογράφημα",
    r"\bechocardiogra", r"\becocardiogra", r"\bεχοκαρδιογράφημα", r"\blvef\b", r"\bejection\s+fraction\b",
    r"\becg\b", r"\bekg\b", r"\bэлектрокардиограф", r"\bηλεκτροκαρδιογράφημα",
    r"\bblood\s+pressure\b", r"\bciśnienie\s+tętnicze\b", r"\btensión\s+arterial\b", r"\bpressione\s+arteriosa\b",
    r"\bαρτηριακή\s+πίεση", r"\bартериальное\s+давление", r"\b혈압\b", r"\b血压\b", r"\b血圧\b",
    r"\bbody\s+mass\s+index\b", r"\bbmi\b", r"\bδείκτης\s+μάζας\s+σώματος\b", r"\bимт\b",
    r"\bwaist\s+circumference\b", r"\bobwód\s+talii\b", r"\bπερίμετρος\s+μέσης"
]
NON_LAB_CLINICAL_PATTERNS: List[str] = APP_CONFIG.get("non_lab_clinical_markers", DEFAULT_NON_LAB_MARKERS)

NON_LATIN_SCRIPT_REGEX = re.compile(
    "|".join(
        data["script_regex"]
        for lang, data in LEXICONS.items()
        if isinstance(data, dict) and lang != "extended_latin" and data.get("script_regex")
    ) or r"[\u0370-\u03FF\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]"
)

def is_non_laboratory_observation(raw_name: str, unit: Optional[str] = None) -> bool:
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

def fix_visual_rtl_reversal(text: str) -> str:
    """Flips visually reversed Arabic/Hebrew strings back into logical order before transliteration."""
    if not text:
        return ""
    cleaned = unicodedata.normalize("NFKC", str(text)).strip()
    arabic_words = re.findall(r"[\u0600-\u06FF]+", cleaned)
    if arabic_words:
        reversed_markers = sum(
            1 for w in arabic_words
            if (len(w) >= 4 and w.endswith("لا"))
            or (len(w) >= 2 and w.startswith("ة"))
            or w in {"ميزنإ", "ريغ", "في", "نم", "عم"}
        )
        if reversed_markers >= 1 and not any(w.startswith("ال") and len(w) >= 4 for w in arabic_words):
            m_prefix = re.match(r"^(\([A-Za-z0-9\s\-\+/]+\))\s*(.+)$", cleaned)
            m_suffix = re.match(r"^(.+?)\s*(\([A-Za-z0-9\s\-\+/]+\))$", cleaned)
            if m_prefix:
                return f"{m_prefix.group(1)} {m_prefix.group(2)[::-1].strip()}"
            if m_suffix:
                return f"{m_suffix.group(1)[::-1].strip()} {m_suffix.group(2)}"
            if re.search(r"[A-Za-z]", cleaned):
                tokens = cleaned.split()
                return " ".join([tok[::-1] if re.search(r"[\u0600-\u06FF]", tok) else tok for tok in tokens])
            return cleaned[::-1].strip()
    return cleaned

def detect_language(text: str) -> str:
    for lang, data in LEXICONS.items():
        if not isinstance(data, dict):
            continue
        regex = data.get("script_regex", "")
        if regex and re.search(regex, text):
            return lang
    return ""

def transliterate_letter_by_letter(text: str, lang: str) -> str:
    lang_data = LEXICONS.get(lang, {})
    if not isinstance(lang_data, dict):
        return text
    char_map = lang_data.get("alphabet_map", {})
    if not isinstance(char_map, dict) or not char_map:
        return text
    return "".join([char_map.get(ch, ch) for ch in text])

def apply_native_roots_safely(text: str, lang: str) -> str:
    """
    Replaces native clinical roots with English equivalents.
    Merges Japanese and Chinese native_roots when CJK script is detected so pure-Kanji
    Japanese test names ('白血球数', '赤血球数', '好中球', '中性脂肪') are always resolved.
    """
    lang_data = LEXICONS.get(lang, {})
    if not isinstance(lang_data, dict):
        return text
    native_roots = dict(lang_data.get("native_roots", {}))
    if lang in {"chinese", "japanese"}:
        other_cjk = "japanese" if lang == "chinese" else "chinese"
        if isinstance(LEXICONS.get(other_cjk), dict):
            native_roots.update(LEXICONS[other_cjk].get("native_roots", {}))
    if not native_roots:
        return text

    result = text
    is_unspaced_script = lang in {"chinese", "japanese", "thai", "korean"}
    for native_word, english_term in sorted(native_roots.items(), key=lambda item: len(item[0]), reverse=True):
        if is_unspaced_script:
            if native_word.lower() in result.lower():
                result = re.sub(rf"(?i){re.escape(native_word)}", f" {english_term} ", result)
        else:
            pattern = rf"(?i)(?<!\w){re.escape(native_word)}(?!\w)"
            if re.search(pattern, result):
                result = re.sub(pattern, f" {english_term} ", result)
    return re.sub(r"\s+", " ", result).strip()

# ==============================================================================
# 2. LOCAL SEMANTIC ENGINE INITIALIZATION (LAZY-LOADED MULTILINGUAL SAPBERT)
# ==============================================================================
_sapbert_tokenizer = None
_sapbert_model = None
_loinc_index = None
_loinc_metadata = None

def _init_sapbert_and_faiss():
    """Lazily initializes SapBERT and FAISS only when unmapped lab items exist."""
    global _sapbert_tokenizer, _sapbert_model, _loinc_index, _loinc_metadata
    if _sapbert_tokenizer is not None and _loinc_index is not None:
        return

    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(METADATA_PATH):
        logging.critical("FATAL: Missing local FAISS index. Run build_loinc_vector_db.py first.")
        raise FileNotFoundError(f"Missing {FAISS_INDEX_PATH} or {METADATA_PATH}")

    logging.info("Initializing offline Multilingual SapBERT Semantic Engine...")
    _sapbert_tokenizer = AutoTokenizer.from_pretrained(SAPBERT_MODEL)
    _sapbert_model = AutoModel.from_pretrained(SAPBERT_MODEL)
    _loinc_index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        _loinc_metadata = json.load(f)

def get_sapbert_embedding(text: str) -> np.ndarray:
    _init_sapbert_and_faiss()
    inputs = _sapbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = _sapbert_model(**inputs)
    vec = outputs.last_hidden_state[:, 0, :].numpy()
    faiss.normalize_L2(vec)
    return vec

def get_existing_seed_codes() -> set:
    if not os.path.exists(SEED_ONTOLOGY_FILE):
        return set()
    with open(SEED_ONTOLOGY_FILE, "r", encoding="utf-8") as f:
        return set(json.load(f).keys())

# ==============================================================================
# 3. 30-LANGUAGE INTELLIGENT ROUTER, ACRONYM EXPANDER & FIDELITY-GUARDED TRANSLATOR
# ==============================================================================
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
        res = "Cholesterol in VLDL [Mass/volume] in Serum or Plasma"

    return res

def resolve_query_string(raw_name: str, unit: Optional[str] = None) -> str:
    norm_raw = fix_visual_rtl_reversal(raw_name)
    norm_lower = norm_raw.lower().strip()

    # Path 0A: Exact match in clinical_acronym_expansions
    if norm_lower in CLINICAL_ACRONYM_EXPANSIONS:
        expanded = CLINICAL_ACRONYM_EXPANSIONS[norm_lower]
        logging.info(f"Clinical Acronym Expansion: '{raw_name}' -> '{expanded}'")
        return expanded

    # Path 0A-Prefix: Check after stripping Nordic/Finnish/European specimen prefixes (S-, P-, B-, U-, fP-, fS-)
    stripped_prefix = re.sub(r"^(?:fp|fs|fb|s|p|b|u|liq|csv)\s*\-\s*", "", norm_lower, flags=re.IGNORECASE).strip()
    if stripped_prefix in CLINICAL_ACRONYM_EXPANSIONS:
        expanded = CLINICAL_ACRONYM_EXPANSIONS[stripped_prefix]
        logging.info(f"Prefix-Stripped Acronym Expansion: '{raw_name}' -> '{expanded}'")
        return expanded

    # Path 0A-Allergen: WHO/IUIS recombinant molecular allergens (rBet v 1, rPhl p 1, rDer p 1, rFel d 1, rCan f 1, rAra h 2, rMal d 1)
    m_allergen = re.search(r"\b(r?[A-Z][a-z]{2,3}\s+[a-z]\s+\d+)\b", norm_raw)
    if m_allergen:
        allergen_code = m_allergen.group(1)
        expanded_alg = f"{allergen_code} IgE Ab [Units/volume] in Serum"
        logging.info(f"Molecular Allergen Expansion: '{raw_name}' -> '{expanded_alg}'")
        return expanded_alg

    outside_text = re.sub(r"[\(\[（][^)\]）]*[\)\]）]", " ", norm_raw).strip()
    outside_has_non_latin = bool(NON_LATIN_SCRIPT_REGEX.search(outside_text))

    # Path 0B: Check parenthetical terms
    for paren_match in re.findall(r"[\(\[（]([^)\]）]+)[\)\]）]", norm_raw):
        clean_p = paren_match.strip()
        clean_p_low = clean_p.lower()
        if clean_p_low in PARENTHETICAL_QUALIFIER_BLACKLIST:
            continue
        if clean_p_low in CLINICAL_ACRONYM_EXPANSIONS:
            expanded = CLINICAL_ACRONYM_EXPANSIONS[clean_p_low]
            logging.info(f"Parenthetical Acronym Expansion: '{raw_name}' -> '{expanded}'")
            return expanded
        if outside_has_non_latin and re.match(r"^[A-Za-z][A-Za-z0-9\s\-\+/]{4,}$", clean_p):
            logging.info(f"Bilingual Non-Latin Fast-Path (Parenthetical): '{raw_name}' -> '{clean_p}'")
            return clean_p

    # Path 0C: Check English prefix before non-Latin scripts (stripping any opening parenthesis)
    prefix_match = re.match(
        r"^([A-Za-z][A-Za-z0-9\s\-/]{1,}?)\s*[\(\[（]?\s*[\u0590-\u05FF\u0600-\u06FF\u0900-\u097F\u0E00-\u0E7F\u3040-\u30FF\u4E00-\u9FFF\uAC00-\uD7AF]",
        norm_raw
    )
    if prefix_match:
        clean_pref = prefix_match.group(1).strip(" ()[]")
        pref_low = clean_pref.lower()
        if pref_low in CLINICAL_ACRONYM_EXPANSIONS:
            return CLINICAL_ACRONYM_EXPANSIONS[pref_low]
        if len(clean_pref) >= 5 and pref_low not in PARENTHETICAL_QUALIFIER_BLACKLIST:
            logging.info(f"Bilingual English Fast-Path (Prefix): '{raw_name}' -> '{clean_pref}'")
            return clean_pref

    lang = detect_language(norm_raw)
    transliterated = norm_raw
    if lang and isinstance(LEXICONS.get(lang), dict):
        transliterated = transliterate_letter_by_letter(norm_raw, lang)
        transliterated = apply_native_roots_safely(transliterated, lang)
        logging.info(f"Lexicon Transliteration ({lang}): '{norm_raw}' -> '{transliterated}'")

    unit_context = f" (Measurement Unit: {unit})" if unit else ""
    prompt = (
        "You are an expert clinical laboratory LOINC terminology translator.\n"
        "Translate the given medical laboratory test name into its standard, full English clinical analyte name.\n"
        "RULES:\n"
        "1. Translate accurately from ANY language (Greek, Arabic, Hebrew, Ukrainian, Russian, Chinese, Japanese, Korean, Hindi, Thai, Vietnamese, Turkish, Finnish, Romanian, Dutch, French, German, Croatian, Polish, Czech, Slovak, Hungarian, Indonesian, Swedish, Danish, Norwegian, Italian, Spanish, Portuguese).\n"
        "2. Expand medical abbreviations into full English names (e.g., 'LDH' -> 'Lactate dehydrogenase', 'CPK' -> 'Creatine kinase', 'VSH' -> 'Erythrocyte sedimentation rate', 'Χοληρυθρίνη Άμεση' -> 'Bilirubin.direct', 'S-Karbamatsepiini' -> 'Carbamazepine', 'S-Levetirasaami' -> 'Levetiracetam', 'Charge Virale VIH-1 (ARN)' -> 'HIV 1 RNA viral load').\n"
        "3. Preserve specimen types if specified in the test name (e.g., 'Zellzahl (Liquor)' -> 'Leukocytes in Cerebral spinal fluid', 'Gesamteiweiß (Liquor)' -> 'Protein in Cerebral spinal fluid', 'CHLORIDY V POTE' -> 'Chloride in Sweat', 'Glucose DNT' -> 'Glucose in Cerebral spinal fluid').\n"
        "4. NEVER change immunoglobulin classes (keep IgM as IgM, IgG as IgG, IgA as IgA) and NEVER substitute a different pathogen, antibody, or analyte.\n"
        "5. Output ONLY the concise English clinical laboratory test name on a single line.\n\n"
        f"Original Test Name: '{norm_raw}'{unit_context}\n"
        f"Phonetic/Lexicon Hint: '{transliterated}'\n\n"
        "Standard English LOINC Analyte Name:"
    )

    for model_name in [MODEL_TRANSLATOR, MODEL_FALLBACK_TRANSLATOR]:
        try:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": 0.0, "num_predict": 60}
            }
            if "qwen3" in model_name:
                kwargs["think"] = False
            response = ollama_client.chat(**kwargs)
            resolved = response["message"]["content"].strip().split("\n")[0].strip()
            if ":" in resolved:
                resolved = resolved.split(":")[-1].strip()
            resolved = resolved.replace('"', "").replace("'", "").replace("*", "")
            resolved = re.sub(r"\s*\(\s*", " (", resolved)
            resolved = re.sub(r"\s*\)\s*", ") ", resolved)
            clean_res = re.sub(r"\s+", " ", resolved).strip()
            if clean_res:
                if clean_res.lower() in CLINICAL_ACRONYM_EXPANSIONS:
                    clean_res = CLINICAL_ACRONYM_EXPANSIONS[clean_res.lower()]
                clean_res = enforce_critical_token_fidelity(norm_raw, clean_res)
                logging.info(f"Resolved Clinical Term ({model_name}): '{norm_raw}' -> '{clean_res}'")
                return clean_res
        except Exception as e:
            logging.warning(f"Translator {model_name} warning on '{norm_raw}': {e}")

    return transliterated

# ==============================================================================
# 4. 6-AXIS LOINC PROPERTY & SYSTEM RE-RANKED CANDIDATE RETRIEVAL
# ==============================================================================
def normalize_assistant_unit(unit: Optional[str]) -> str:
    """Preserves case-sensitive SI 'G/L' (10*9/L) vs 'g/L' before lowercasing."""
    if not unit:
        return ""
    u_raw = unit.strip()
    if (u_raw.startswith("(") and u_raw.endswith(")")) or (u_raw.startswith("[") and u_raw.endswith("]")):
        u_raw = u_raw[1:-1].strip()
    if u_raw in CASE_SENSITIVE_TOKEN_MAP:
        return CASE_SENSITIVE_TOKEN_MAP[u_raw].lower()
    u_low = u_raw.lower().replace("μ", "u").replace("µ", "u")
    u_low = re.sub(r"\s+", " ", u_low).strip()
    return TOKEN_MAP.get(u_low, TOKEN_MAP.get(u_low.replace(" ", ""), u_low))

def compute_loinc_6axis_bonus(loinc_name: str, unit: Optional[str], query_context: str) -> float:
    bonus = 0.0
    name_low = loinc_name.lower()
    ctx_low = query_context.lower()
    u_low = normalize_assistant_unit(unit)

    # 1. Property Axis Alignment (across all 7 unit dimensions in config_rules.json)
    if u_low in UNIT_DIMENSIONS_MAP.get("MASS_VOL", set()):
        if "[mass/volume]" in name_low:
            bonus += 0.08
        elif "[moles/volume]" in name_low:
            bonus -= 0.04
    elif u_low in UNIT_DIMENSIONS_MAP.get("MOLAR_VOL", set()):
        if "[moles/volume]" in name_low:
            bonus += 0.08
        elif "[mass/volume]" in name_low:
            bonus -= 0.04
    elif u_low in UNIT_DIMENSIONS_MAP.get("COUNT_VOL", set()):
        if "[#/volume]" in name_low or "[entiticvolume]" in name_low or "[entitic mass]" in name_low:
            bonus += 0.08
    elif u_low in UNIT_DIMENSIONS_MAP.get("ACTIVITY_VOL", set()):
        if "[enzymatic activity/volume]" in name_low:
            bonus += 0.08
        elif "[units/volume]" in name_low:
            bonus += 0.06
    elif u_low in UNIT_DIMENSIONS_MAP.get("PRESSURE", set()):
        if "[partial pressure]" in name_low or "pressure" in name_low:
            bonus += 0.08
    elif u_low == "titer":
        if "[titer]" in name_low:
            bonus += 0.08
    elif u_low in {"ug/g", "mcg/g", "mg/kg"}:
        if "in stool" in name_low:
            bonus += 0.12

    # 2. System Axis Alignment
    stool_words = SPECIMEN_SYSTEM_MARKERS.get("stool", ["stool", "fecal", "faecal", "fäkales", "fækal", "stolici", "feci", "кал", "粪便"])
    urine_words = SPECIMEN_SYSTEM_MARKERS.get("urine", ["urine", "urin", "urina", "idrar", "virtsa", "harn", "моча", "сеча", "소변", "尿", "بول", "שתן", "בשתן"])
    csf_words = SPECIMEN_SYSTEM_MARKERS.get("csf", ["csf", "spinalvæske", "liquor", "likvor", "lcr", "dnt", "뇌척수액", "脑脊液", "髄液"])
    sweat_words = SPECIMEN_SYSTEM_MARKERS.get("sweat", ["sweat", "svedtest", "pote", "potný", "potny"])
    sputum_words = SPECIMEN_SYSTEM_MARKERS.get("sputum", ["sputum", "قشع", "喀痰", "객담", "เสมหะ"])

    is_stool = any(w in ctx_low for w in stool_words) or u_low in {"ug/g", "mcg/g"}
    is_urine = any(w in ctx_low for w in urine_words) or u_low in {"mg/g", "mg/24h", "g/24h"} or bool(re.match(r"^u\-", ctx_low))
    is_csf = any(w in ctx_low for w in csf_words) or bool(re.match(r"^(?:liq|csv)\-", ctx_low))
    is_sweat = any(w in ctx_low for w in sweat_words)
    is_sputum = any(w in ctx_low for w in sputum_words)

    if is_stool:
        if "in stool" in name_low:
            bonus += 0.12
        elif "in serum" in name_low or "in plasma" in name_low:
            bonus -= 0.06
    elif is_urine:
        if "in urine" in name_low or "in 24 hour urine" in name_low:
            bonus += 0.12
            if ("24" in ctx_low or u_low in {"mg/24h", "g/24h"}) and "24 hour" in name_low:
                bonus += 0.06
        elif "in serum" in name_low or "in plasma" in name_low:
            bonus -= 0.06
    elif is_csf:
        if "in cerebral spinal fluid" in name_low or "in csf" in name_low:
            bonus += 0.12
        elif "in serum" in name_low or "in blood" in name_low:
            bonus -= 0.06
    elif is_sweat:
        if "in sweat" in name_low:
            bonus += 0.14
    elif is_sputum:
        if "in sputum" in name_low or "in respiratory" in name_low:
            bonus += 0.12
    else:
        if any(s in name_low for s in ["in serum or plasma", "in serum", "in blood", "in plasma"]):
            bonus += 0.05
        exotic_specimens = [
            "in stone", "in water", "in pericardial fluid", "in amniotic fluid",
            "in peritoneal", "in drain", "in tear", "in semen", "in saliva",
            "in synovial fluid", "in dbs", "in vitre", "in sweat", "post mortem",
            "in breast milk", "in hair", "in nail", "in food", "in vomitus"
        ]
        if any(ex in name_low for ex in exotic_specimens):
            bonus -= 0.10

    return bonus

def search_local_loinc_candidates(
    search_query: str,
    unit: Optional[str],
    raw_name: str,
    existing_seeds: set,
    top_k: int = 5
) -> list:
    _init_sapbert_and_faiss()
    query_vector = get_sapbert_embedding(search_query)
    distances, indices = _loinc_index.search(query_vector, 25)

    scored_candidates = []
    combined_ctx = f"{raw_name} {search_query}"

    for i in range(25):
        idx = str(indices[0][i])
        raw_score = float(distances[0][i])
        if raw_score > 0.40 and idx in _loinc_metadata:
            meta = _loinc_metadata[idx]
            loinc_code = meta["loinc"]
            name = meta["name"]
            adj_score = raw_score + compute_loinc_6axis_bonus(name, unit, combined_ctx)
            # Only boost & recommend existing seed codes when raw semantic confidence is high (>= 0.70)
            if loinc_code in existing_seeds and raw_score >= 0.70:
                adj_score += 0.05
            scored_candidates.append((adj_score, raw_score, loinc_code, name))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)

    results = []
    for adj_score, raw_score, loinc_code, name in scored_candidates[:top_k]:
        if loinc_code in existing_seeds and raw_score >= 0.70:
            results.append(f"{loinc_code} ({name}) [ALREADY IN SEED - RECOMMENDED (Confidence: {raw_score:.2f})]")
        else:
            results.append(f"{loinc_code} ({name}) [Confidence: {raw_score:.2f}]")

    return results

# ==============================================================================
# 5. EXECUTION PIPELINE
# ==============================================================================
def run_assistant():
    logging.info("Starting AegisParse Config-Driven 30-Language Ontology Assistant...")
    unmapped_items: Dict[str, Dict[str, Any]] = {}

    for file_path in sorted(glob.glob(os.path.join(JSON_DIR, "*_2_enriched.json"))):
        base_file = os.path.basename(file_path).replace("_2_enriched.json", "")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for test in json.load(f).get("tests", []):
                    if test.get("loinc_code") == "UNMAPPED":
                        raw_name = test.get("raw_extracted_name", "").strip()
                        unit_val = test.get("unit")
                        if not raw_name or is_non_laboratory_observation(raw_name, unit_val):
                            continue
                        if raw_name not in unmapped_items:
                            unmapped_items[raw_name] = {
                                "unit": unit_val,
                                "count": 1,
                                "files": [base_file]
                            }
                        else:
                            unmapped_items[raw_name]["count"] += 1
                            if not unmapped_items[raw_name]["unit"] and unit_val:
                                unmapped_items[raw_name]["unit"] = unit_val
                            if base_file not in unmapped_items[raw_name]["files"]:
                                unmapped_items[raw_name]["files"].append(base_file)
        except Exception as e:
            logging.warning(f"Error reading {file_path}: {e}")

    if not unmapped_items:
        logging.info(f"No unmapped laboratory items found in {JSON_DIR}.")
        return

    existing_seeds = get_existing_seed_codes()
    review_dossier = []

    for raw_name, info in unmapped_items.items():
        unit = info["unit"]
        resolved_search_term = resolve_query_string(raw_name, unit)
        if is_non_laboratory_observation(resolved_search_term, unit):
            logging.info(f"Skipping non-laboratory clinical observation: '{raw_name}' ('{resolved_search_term}')")
            continue

        logging.info(f"Querying FAISS (6-Axis Re-Ranked) for: '{resolved_search_term}' [Unit: {unit}]")
        candidates = search_local_loinc_candidates(resolved_search_term, unit, raw_name, existing_seeds)

        review_dossier.append({
            "raw_extracted_name": raw_name,
            "extracted_unit": unit,
            "occurrence_count": info["count"],
            "source_reports": info["files"],
            "resolved_search_term": resolved_search_term,
            "sapbert_semantic_candidates": list(dict.fromkeys(candidates)) if candidates else ["NO VALID LAB CODES FOUND"]
        })

    with open(PENDING_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"total_unmapped_items": len(review_dossier), "operator_review_queue": review_dossier},
            f,
            indent=4,
            ensure_ascii=False
        )

    logging.info("\n" + "=" * 80)
    logging.info("ASSISTANT RUN COMPLETE - OPERATOR DOSSIER READY")
    logging.info("=" * 80)
    logging.info("HUMAN-IN-THE-LOOP REQUIRED ACTIONS:")
    logging.info(f"1. Open the file: {PENDING_FILE}")
    logging.info("2. Review the 'operator_review_queue'.")
    logging.info("3. For each valid laboratory test:")
    logging.info("   - Select the most accurate LOINC code from 'sapbert_semantic_candidates'.")
    logging.info("   - Ensure the mass/volume units match your local laboratory standards.")
    logging.info(f"4. Open the file: {SEED_ONTOLOGY_FILE}")
    logging.info("5. Copy your selected LOINC code and format it as a new block at the bottom of the file.")
    logging.info("6. Save seed_ontology.json. The main extraction pipeline will now map this automatically.")
    logging.info("=" * 80 + "\n")

if __name__ == "__main__":
    run_assistant()