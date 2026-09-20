FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PORT=8000

WORKDIR /srv
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY ingest ./ingest
COPY scripts ./scripts
# Read-only data, baked in: no database service and no artifact hosting needed.
COPY data/build/radar_serve.duckdb data/build/vectors_openai.npz ./data/build/

RUN useradd -m -u 1000 user && chown -R 1000:1000 /srv
USER 1000

EXPOSE 8000
CMD ["sh","-c","uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
