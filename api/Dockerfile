# Backend image for Railway.
# BUILD CONTEXT MUST BE THE REPO ROOT -- the image needs api/, agent/, rag/,
# ml/, and data_generation/ as siblings, matching how agent/tools.py resolves
# them (Path(__file__).resolve().parent.parent / "ml", etc). On Railway:
# leave "Root Directory" blank; the repo-root railway.json points here via
# dockerfilePath.
#
#   docker build -f api/Dockerfile -t pip-backend .

FROM python:3.12-slim

WORKDIR /srv

# 1. Dependencies -- cached until requirements.txt changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2. Application code and everything the regeneration scripts below need.
COPY api/ ./api/
COPY agent/ ./agent/
COPY rag/ ./rag/
COPY ml/ ./ml/
COPY data_generation/ ./data_generation/
COPY knowledge_base/ ./knowledge_base/
COPY data/synthetic/.gitkeep ./data/synthetic/.gitkeep

# 3. Regenerate the three artifacts .gitignore deliberately excludes from
# version control -- synthetic CSVs, the trained model, and the Chroma
# vector index -- from their tracked, seeded sources, same as local dev.
# Order matters: inspection/incident generation and model training both
# read facility_risk_assessments.csv; build_index.py needs cwd=rag/ for its
# same-directory `from chunk_documents import ...`.
RUN python data_generation/generate_risk_assessments.py \
 && python data_generation/generate_inspection_reports.py \
 && python data_generation/generate_incident_reports.py \
 && python ml/train_final_model.py \
 && cd rag && python build_index.py

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
