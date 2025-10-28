# Dockerfile (no extension) — place at repo root
FROM python:3.11-slim

# ---- System packages ----
# libgomp1: needed by FAISS and some numpy/scikit builds (OpenMP runtime)
# ffmpeg: optional but useful for pydub / media ops
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# ---- App setup ----
WORKDIR /app

# Python deps first (better cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app (including faiss_transcripts/ if present)
COPY . .

# Streamlit must listen on 0.0.0.0:$PORT for Cloud Run/Render
ENV PORT=8080
EXPOSE 8080
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_WATCHER_TYPE=none

# Streamlit entrypoint
CMD ["bash","-lc","streamlit run recherche_streamlit.py --server.port=$PORT --server.address=0.0.0.0"]
