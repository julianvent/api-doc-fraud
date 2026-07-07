# Document Fraud Detection API

A multi-stage identity document fraud detection system built with FastAPI.

---

## Table of Contents

- [Overview](#overview)
- [Purpose & Scope](#purpose--scope)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Quick Start Example](#quick-start-example)
- [Using the Swagger UI](#using-the-swagger-ui)
- [Environment Variables Reference](#environment-variables-reference)
- [Detailed Documentation](#detailed-documentation)

---

## Overview

This API analyzes uploaded identity documents (images or PDFs) through a multi-stage forensic pipeline and returns a risk verdict (`ACCEPT`, `REVIEW`, or `REJECT`) along with a risk score and detailed module flags.

---

## Purpose & Scope

The system is designed to assist identity verification teams in detecting fraudulent or AI-generated identity documents at scale. It is intended to be used as a backend service integrated into larger KYC (Know Your Customer) or onboarding flows.

**In scope:**
- Forensic analysis of ID documents (passports, national IDs) — images and PDFs
- Template-based layout and OCR verification
- MRZ (Machine Readable Zone) parsing and cross-field validation
- Pixel-level manipulation detection (splicing, cloning, compression artifacts)
- File metadata forensics (AI-generation traces, software signatures)
- Document template management (create, confirm, list)

**Out of scope:**
- End-user authentication or session management
- Direct integration with government databases

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        API Layer  (FastAPI)                      │
│             POST /v1/verify    ·   /v1/templates/*               │
└───────────────────────────────┬──────────────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │      Controllers       │
                    │  FraudDetection │ Tmpl │
                    └───────────┬───────────┘
                                │
          ┌─────────────────────▼─────────────────────┐
          │              Service Layer                 │
          │                                           │
          │  ┌────────────┐   ┌─────────────────────┐ │
          │  │  Metadata  │   │     Tampering        │ │
          │  │ (forensics)│   │  TruFor + DocTamper  │ │
          │  └─────┬──────┘   └──────────┬──────────┘ │
          │        │                     │             │
          │  ┌─────▼──────┐   ┌──────────▼──────────┐ │
          │  │Preprocessor│   │         OCR          │ │
          │  │(normalize) │   │  PaddleOCR + MRZ +   │ │
          │  └────────────┘   │  Template Matching   │ │
          │                   └─────────────────────┘ │
          └──────────────────┬────────────────────────┘
                             │
          ┌──────────────────▼────────────────────────┐
          │             Persistence Layer              │
          │  PostgreSQL  │  Qdrant (vectors)  │  Disk  │
          └────────────────────────────────────────────┘
                             │
          ┌──────────────────▼────────────────────────┐
          │           External Services                │
          │     Ollama (qwen2.5vl:7b · bge-m3)        │
          └────────────────────────────────────────────┘
```

### Pipeline Execution Order (fraud detection)

```
Upload → [1] Metadata forensics
        → [2] Tampering detection  (runs on raw pixels, before any processing)
        → [3] Preprocessor         (deskew, enhance, rasterize PDFs at 300 DPI)
        → [4] OCR + Template match (PaddleOCR, MRZ parse, spatial layout compare)
        → Policy engine            (weighted risk score → ACCEPT / REVIEW / REJECT)
```

**Score weights:** OCR 40% · Tampering 35% · Metadata 25%

**Hard-reject triggers:** confirmed pixel manipulation · MRZ field mismatch · identity packet mismatch · metadata suspicion ≥ 0.85 · aggregate score ≥ 0.70

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Web framework | FastAPI 0.139 · Uvicorn 0.50 |
| Data validation | Pydantic v2 |
| OCR | PaddleOCR 3.4 / PaddlePaddle 3.2 |
| Vision LLM | Ollama (`qwen2.5vl:7b`) |
| Embeddings | Ollama (`bge-m3`) |
| ML / Deep learning | PyTorch 2.11 · ONNX Runtime 1.27 · timm |
| Tampering detection | TruFor · DocTamper |
| Vector database | Qdrant (local on-disk + remote HTTP) |
| Relational database | PostgreSQL · SQLAlchemy 2 · Alembic |
| Image processing | OpenCV 4.13 · Pillow · PyMuPDF |
| MRZ parsing | mrz 0.6 |
| Fuzzy matching | RapidFuzz |
| Logging | colorlog · Sentry SDK |

---

## Installation & Setup

### Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.11 |
| PostgreSQL | 14+ |
| Ollama | latest |
| Qdrant | 1.7+ (optional — disabled by default) |

### 1. Clone the repository

```bash
git clone <repository-url>
cd api-doc-fraud
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Installation may take several minutes due to PyTorch and PaddlePaddle wheel sizes.

### 4. Set up PostgreSQL

Create the database and user:

```sql
CREATE DATABASE antifraud;
CREATE USER postgres WITH PASSWORD 'root';
GRANT ALL PRIVILEGES ON DATABASE antifraud TO postgres;
```

The default connection string is `postgresql://postgres:root@localhost:5432/antifraud`.  
Override it by setting the `DATABASE_URL` environment variable if your credentials differ.

Run database migrations:

```bash
alembic upgrade head
```

### 5. Set up Ollama

Install Ollama from [ollama.com](https://ollama.com) and pull the required models:

```bash
ollama pull qwen2.5vl:7b
ollama pull qwen2.5:7b
ollama pull bge-m3
```

Ollama must be running on `http://localhost:11434` before starting the API.

### 6. Configure environment variables (optional)

The API runs with sensible defaults out of the box. Create a `.env` file in the project root to override any value:

```env
# CORS
CORS_ALLOW_ORIGINS=http://localhost:3000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=text

# OCR engine: paddle | dots | dolphin
OCR_ENGINE=paddle

# Ollama endpoints
OLLAMA_VISION_URL=http://localhost:11434/api/generate
OLLAMA_VISION_MODEL=qwen2.5vl:7b
OLLAMA_EMBED_URL=http://localhost:11434/api/embeddings
OLLAMA_EMBED_MODEL=bge-m3

# Qdrant (disabled by default)
QDRANT_URL=http://localhost:6333
DISABLE_VECTOR_MATCH=1

# Upload limits
MAX_UPLOAD_BYTES=10485760
```

See [Environment Variables Reference](#environment-variables-reference) for the full list.

### 7. Start the server

**Development (with live reload):**
```bash
uvicorn main:app --reload
```

**Production:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

> On first startup the server warms up all ML models (tampering detectors, OCR engine). This may take 30–60 seconds depending on hardware.

---

## Quick Start Example

### Verify a document

```bash
curl -X POST http://localhost:8000/v1/verify \
  -F "file=@passport_sample.jpg"
```

**Response:**
```json
{
  "verdict": "ACCEPT",
  "risk_score": 0.12,
  "confidence": 0.91,
  "modules": {
    "metadata": {
      "verdict": "ACCEPT",
      "score": 0.05,
      "flags": []
    },
    "tampering": {
      "verdict": "ACCEPT",
      "score": 0.08,
      "flags": []
    },
    "ocr": {
      "verdict": "ACCEPT",
      "score": 0.18,
      "flags": [],
      "mrz_valid": true,
      "template_match": "passport_MEX_2020"
    }
  }
}
```

### Generate a document template

```bash
curl -X POST http://localhost:8000/v1/templates/generate \
  -F "file=@id_sample.jpg" \
  -F "mode=auto"
```

### List available templates

```bash
curl http://localhost:8000/v1/templates
```

---

## Using the Swagger UI

FastAPI auto-generates interactive API documentation. Once the server is running, navigate to:

| Interface | URL | Description |
|---|---|---|
| **Swagger UI** | `http://localhost:8000/docs` | Interactive — try endpoints directly in the browser |
| **ReDoc** | `http://localhost:8000/redoc` | Clean read-only reference |
| **OpenAPI JSON** | `http://localhost:8000/openapi.json` | Raw schema for tooling / SDK generation |

### How to test an endpoint in Swagger

1. Open `http://localhost:8000/docs` in your browser.
2. Click on any endpoint to expand it (e.g. `POST /v1/verify`).
3. Click **Try it out** in the top-right corner of the endpoint card.
4. Fill in the required parameters — for file uploads, use the file picker that appears.
5. Click **Execute**.
6. Scroll down to see the **Response body**, **Status code**, and **Response headers**.

> The Swagger UI reflects the live schema — if new endpoints are added, they appear automatically on the next server restart.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `CORS_ALLOW_ORIGINS` | `http://localhost:3000` | Comma-separated list of allowed CORS origins |
| `LOG_LEVEL` | `INFO` | App log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | `text` | Log format: `text` or `json` |
| `MAX_UPLOAD_BYTES` | `10485760` | Max upload size in bytes (default: 10 MB) |
| `OCR_ENGINE` | `paddle` | OCR backend: `paddle`, `dots`, or `dolphin` |
| `DOTS_MODEL_PATH` | `service/ocr/DotsOCR` | Path to DotsOCR model weights |
| `DOLPHIN_MODEL_PATH` | `service/ocr/hf_model` | Path to Dolphin model weights |
| `DOLPHIN_REPO_PATH` | `service/ocr/Dolphin` | Path to Dolphin repository |
| `OLLAMA_VISION_URL` | `http://localhost:11434/api/generate` | Ollama vision inference URL |
| `OLLAMA_VISION_MODEL` | `qwen2.5vl:7b` | Vision model name for document extraction |
| `OLLAMA_VISION_TIMEOUT` | `600` | Vision request timeout in seconds |
| `OLLAMA_URL` | `http://localhost:11434/api/generate` | Ollama URL for template OCR |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Text model for template OCR |
| `OLLAMA_TIMEOUT` | `600` | Ollama request timeout in seconds |
| `OLLAMA_EMBED_URL` | `http://localhost:11434/api/embeddings` | Ollama embeddings endpoint |
| `OLLAMA_EMBED_MODEL` | `bge-m3` | Embedding model for vector search |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `QDRANT_COLLECTION` | `document_templates` | Qdrant collection name |
| `MATCH_THRESHOLD` | `0.75` | Minimum vector similarity for template match |
| `DISABLE_VECTOR_MATCH` | `1` | Set to `0` to enable Qdrant vector matching |
| `TEMPLATE_SCAN_CACHE_DIR` | `tmp/template_scans` | Temporary directory for template scan cache |
| `TEMPLATE_SCAN_CACHE_TTL` | `3600` | Scan cache TTL in seconds |
| `TEMPLATE_IMAGE_MAX_SIDE` | `1024` | Max pixel side length for template images |
| `PREPROCESSOR_CONFIG` | _(built-in TOML)_ | Path to a custom preprocessor config TOML |

---

## Detailed Documentation

| Topic | Location |
|---|---|
| Preprocessor configuration knobs | [`service/preprocessor/config/defaults.toml`](service/preprocessor/config/defaults.toml) |
| Document template JSON format | [`service/ocr/templates/`](service/ocr/templates/) |
| Database schema & migrations | [`alembic/versions/`](alembic/versions/) |
| Live interactive API reference | `http://localhost:8000/docs` (server must be running) |
| Raw OpenAPI schema | `http://localhost:8000/openapi.json` (server must be running) |
