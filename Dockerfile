FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AEGIS_BASE_DIR=/app \
    OLLAMA_HOST=http://ollama:11434

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy AegisParse scripts and Single Source of Truth JSON configurations
COPY app.py ontology_assistant.py build_loinc_vector_db.py ./
COPY config_rules.json clinical_lexicons.json seed_ontology.json ./
COPY loinc_index.faiss loinc_metadata.json ./

RUN mkdir -p /app/Images /app/Json_Outputs

CMD ["python", "app.py"]