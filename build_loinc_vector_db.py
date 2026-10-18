import os
import csv
import json
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ==============================================================================
# CONFIGURATION
# ==============================================================================
BASE_DIR = os.getenv("AEGIS_BASE_DIR", os.path.dirname(os.path.abspath(__file__)))
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "loinc_index.faiss")
METADATA_PATH = os.path.join(BASE_DIR, "loinc_metadata.json")
LOCAL_LOINC_CSV = os.getenv(
    "AEGIS_LOINC_CSV",
    os.path.join(BASE_DIR, "Loinc_2.82", "LoincTable", "Loinc.csv")
)

MODEL_NAME = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"

_tokenizer = None
_model = None

def get_embedding(text: str) -> np.ndarray:
    """Converts a text string into a 768-dimensional semantic vector."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print("Loading SapBERT Semantic Engine...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = _model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()[0]

def build_database():
    if not os.path.exists(LOCAL_LOINC_CSV):
        print(f"ERROR: Cannot find local LOINC file at {LOCAL_LOINC_CSV}")
        print("Please ensure the Loinc_2.82/LoincTable/Loinc.csv file exists.")
        return

    print("Reading local LOINC database...")
    valid_loincs = []

    try:
        with open(LOCAL_LOINC_CSV, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                loinc_class = row.get("CLASS", "")
                status = row.get("STATUS", "")

                if status == "ACTIVE" and loinc_class in ["CHEM", "HEM", "SERO", "MICRO", "TOX"]:
                    valid_loincs.append({
                        "code": row["LOINC_NUM"],
                        "name": row["LONG_COMMON_NAME"]
                    })

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"Successfully loaded {len(valid_loincs)} laboratory concepts.")
    print("Building FAISS vector index (This will take 1-3 minutes)...")

    dimension = 768  # SapBERT vector dimension
    index = faiss.IndexFlatIP(dimension)  # Cosine Similarity Index
    metadata = {}
    vectors = []

    for i, concept in enumerate(valid_loincs):
        code = concept["code"]
        name = concept["name"]

        vec = get_embedding(name)
        vectors.append(vec)
        metadata[str(i)] = {"loinc": f"LOINC:{code}", "name": name}

        if (i + 1) % 500 == 0:
            print(f" -> Processed {i + 1} / {len(valid_loincs)} concepts...")

    matrix = np.vstack(vectors).astype(np.float32)
    faiss.normalize_L2(matrix)
    index.add(matrix)

    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print(f"\nSUCCESS: Offline semantic database generated at {FAISS_INDEX_PATH}")
    print("You can now run 'ontology_assistant.py'")

if __name__ == "__main__":
    build_database()