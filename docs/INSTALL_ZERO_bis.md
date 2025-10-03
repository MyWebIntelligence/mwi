# Installation Guide - Level Zero

This guide provides step-by-step installation instructions for MyWebIntelligence (MyWI) with three installation methods and three configuration levels.

## Table of Contents

- [Overview](#overview)
- [Installation Methods](#installation-methods)
  - [Local Installation](#local-installation)
  - [Docker Installation](#docker-installation)
  - [Docker Compose Installation](#docker-compose-installation)
- [Configuration Levels](#configuration-levels)
- [Interactive Setup Scripts](#interactive-setup-scripts)
- [Post-Installation](#post-installation)
- [Troubleshooting](#troubleshooting)

---

## Overview

MyWI offers **3 installation methods** with **3 configuration levels** each:

### Installation Methods

1. **Local** - Python virtual environment on your machine
2. **Docker** - Single container (manual setup)
3. **Docker Compose** - Orchestrated containers (recommended)

### Configuration Levels

1. **Basic** (`install`) - Core functionality only
2. **API** (`install api`) - install external APIs (SerpAPI, SEO Rank)
3. **LLM** (`install llm`) - Complete setup with embeddings and AI features

---

## Local Installation

### Prerequisites

- Python 3.10+
- pip (Python package installer)
- git
- Node.js 18+ (for Mercury Parser)

### Level 1: Basic Installation (`install`)

#### Step 1: Clone and Setup Environment
After 
# Clone repository
git clone https://github.com/MyWebIntelligence/mwi.git
cd mwi

# Create virtual environment
```bash

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Upgrade pip
python -m pip install -U pip setuptools wheel
```

#### Step 2: Install Dependencies

```bash
# Install base requirements
pip install -r requirements.txt

# Install NLTK data
python install.py
```

#### Step 3: Install Mercury Parser (optional but recommended)

```bash
# Install Node.js first (https://nodejs.org/)
sudo npm install -g @postlight/mercury-parser
```

#### Step 4: Run Interactive Configuration

```bash
python scripts/install-basic.py
```

**Interactive prompts:**

1. **Data location** - Where to store database and exports
   - Default (enter): `./data`
   - Example: `/Users/you/mywi_data` (macOS/Linux) or `C:/Users/You/mywi_data` (Windows)

The script generates `settings.py` with your configuration.

#### Step 5: Install Playwright (if using dynamic media)

If you enabled *Dynamic Media Extraction* during `install-basic.py`, complete these steps:

```bash
# 1. Install Playwright browsers
python install_playwright.py

# 2. (Linux) install required system libraries
sudo apt-get install libnspr4 libnss3 libdbus-1-3 libatk1.0-0 \
    libatk-bridge2.0-0 libatspi2.0-0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libasound2
```

**Docker users:**

```bash
docker compose exec mwi bash -lc "apt-get update && apt-get install -y libnspr4 libnss3 libdbus-1-3 \
    libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libasound2"
docker compose exec mwi python install_playwright.py
```

Skip this step if you left dynamic media extraction disabled.

#### Step 6: Initialize Database

```bash
# Create database schema
python mywi.py db setup

# Verify installation
python mywi.py land list
```
message final with all pprocess check
---

### Level 2: API Installation (`install api`)

**Builds on Basic installation with external API integrations.**

#### Step 1: check if Complete Basic Installation

if test 'python mywi.py land list' dont work then stop install.
and show [Level 1: Basic Installation](#level-1-basic-installation-install) first.

#### Step 2: Run API Configuration Script

```bash
python scripts/install-api.py
```

**Interactive prompts:**

**SerpAPI Configuration**
1. **Enable SerpAPI?** (Google search results)
   - Default: `no`
   - If yes:
     - API Key: `your-serpapi-key-here`
     - Get key at: https://serpapi.com/
     - Free tier: 100 searches/month

**SEO Rank Configuration**
2. **Enable SEO Rank enrichment?**
   - Default: `no`
   - If yes:
     - API Base URL: `https://seo-rank.my-addr.com/api2/sr+fb`
     - API Key: `your-seorank-key-here`
     - Request delay (seconds): `1.0`

**OpenRouter Configuration** (AI relevance filtering)
3. **Enable OpenRouter LLM gate?**
   - Default: `no`
   - If yes:
     - API Key: `your-openrouter-key-here`
     - Get key at: https://openrouter.ai/
     - Model selection:
       ```
       1. openai/gpt-4o-mini (fast, cheap)
       2. anthropic/claude-3-haiku (quality)
       3. google/gemini-1.5-flash (balanced)
       4. deepseek/deepseek-chat-v3.1 (default, economical)
       5. meta-llama/llama-3.1-8b-instruct
       6. Custom (enter model slug)
       ```
     - Timeout: `15` seconds
     - Min chars: `140`
     - Max chars: `12000`
     - Max calls per run: `500`

The script updates `settings.py` with API credentials.

#### Step 3: Test API Connections (optional)

```bash
# Test SerpAPI
python scripts/test-apis.py --serpapi

# Test SEO Rank
python scripts/test-apis.py --seorank

# Test OpenRouter
python scripts/test-apis.py --openrouter
```

---

### Level 3: LLM Installation (`install llm`)

**Complete installation with embeddings, semantic search, and NLI.**

#### Step 1: Complete API Installation

Follow [Level 2: API Installation](#level-2-api-installation-install-api) first.

#### Step 2: Install ML Dependencies

```bash
pip install -r requirements-ml.txt
```

This installs:
- `sentence-transformers` (cross-encoder NLI)
- `transformers` (HuggingFace models)
- `torch` (PyTorch backend)
- `faiss-cpu` (fast similarity search)
- `sentencepiece` (tokenization)

#### Step 3: Run LLM Configuration Script

```bash
python scripts/install-llm.py
```

**Interactive prompts:**

**Embedding Provider**
1. **Choose embedding provider:**
   ```
   1. fake (testing/development)
   2. openai (OpenAI API)
   3. mistral (Mistral AI) [DEFAULT]
   4. gemini (Google Gemini)
   5. huggingface (HuggingFace Inference)
   6. ollama (local Ollama)
   7. http (custom HTTP endpoint)
   ```

2. **Provider-specific configuration:**

   **If OpenAI:**
   - API Key: `sk-...`
   - Model: `text-embedding-3-small` (default)
   - Base URL: `https://api.openai.com/v1`

   **If Mistral:**
   - API Key: `your-mistral-key`
   - Model: `mistral-embed` (default)
   - Get key at: https://console.mistral.ai/

   **If Gemini:**
   - API Key: `your-gemini-key`
   - Model: `embedding-001`
   - Get key at: https://makersuite.google.com/app/apikey

   **If HuggingFace:**
   - API Key: `hf_...`
   - Model: `sentence-transformers/all-MiniLM-L6-v2`

   **If Ollama:**
   - Base URL: `http://localhost:11434`
   - Model: `nomic-embed-text`

   **If HTTP:**
   - Endpoint URL: `https://your-embedding-api.com/embed`
   - API Key (optional): `...`
   - Model name: `custom-model`

3. **Embedding settings:**
   - Batch size: `32`
   - Min paragraph chars: `150`
   - Max paragraph chars: `6000`
   - Similarity threshold: `0.75`
   - Similarity method: `cosine` or `cosine_lsh`

**NLI (Natural Language Inference) Configuration**

4. **NLI model selection:**
   ```
   1. MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 [DEFAULT - Multilingual]
   2. typeform/distilbert-base-uncased-mnli [Fast, English only]
   3. Custom (enter HuggingFace model name)
   ```

5. **NLI settings:**
   - Backend preference: `auto`, `transformers`, `crossencoder`, `fallback`
   - Batch size: `64`
   - Max tokens: `512`
   - Torch threads: `1` (CPU cores to use)

**ANN Backend Configuration**

6. **Similarity backend:**
   ```
   1. faiss (fast, requires faiss-cpu) [DEFAULT]
   2. bruteforce (slower, no extra deps)
   ```

7. **Search parameters:**
   - Top K neighbors: `50`
   - Entailment threshold: `0.8`
   - Contradiction threshold: `0.8`

The script updates `settings.py` with embedding and NLI configuration.

#### Step 4: Verify ML Installation

```bash
# Check environment and dependencies
python mywi.py embedding check
```

Expected output:
```
✓ Embedding provider: mistral
✓ FAISS: available
✓ Transformers: available
✓ Sentence-Transformers: available
✓ Database tables: ready
```

#### Step 5: Download NLI Models (first run)

```bash
# Models download automatically on first use, or pre-download:
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
  AutoTokenizer.from_pretrained('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'); \
  AutoModelForSequenceClassification.from_pretrained('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')"
```

---

## Docker Installation

### Prerequisites

- Docker 20.10+
- Docker Desktop (recommended)

### Level 1: Basic Docker (`install docker`)


All the step is after 

git clone https://github.com/MyWebIntelligence/mwi.git
cd mwi
and inside mwi

#### Step 1: Prepare Configuration Files

```bash
# Copy settings template
cp settings-example.py settings.py

# chose data directory
#Ask data directory or default 
```
#### Step 2: Build Docker Image

```bash
docker build -t mwi:latest .
```

#### Step 4: Run Container

```bash
docker run -dit \
  --name mwi \
  -v [/pat/to/data]:/app/data \
  mwi:latest
```

#### Step 5: Initialize Database

```bash
docker exec -it mwi python mywi.py db setup
```

#### Step 6: Verify Installation

```bash
docker exec -it mwi python mywi.py land list
```

**Management commands:**
```bash
docker stop mwi          # Stop container
docker start mwi         # Start container
docker restart mwi       # Restart container
docker rm mwi            # Remove container (data persists in ~/mywi_data)
```

---

### Level 2: Docker API (`install docker api`)

**Builds on Basic Docker with API credentials.**

#### Step 1: Check Complete Basic Docker Installation

Check or stop
Follow [Level 1: Basic Docker](#level-1-basic-docker-install-docker) first.

#### Step 2: Configure API Keys in settings.py

Edit `settings.py` in your project directory:

```python
# SerpAPI
serpapi_api_key = "your-serpapi-key-here"

# SEO Rank
seorank_api_key = "your-seorank-key-here"
seorank_api_base_url = "https://seo-rank.my-addr.com/api2/sr+fb"

# OpenRouter
openrouter_enabled = True
openrouter_api_key = "your-openrouter-key-here"
openrouter_model = "deepseek/deepseek-chat-v3.1"
```

#### Step 3: Rebuild Container (if settings.py is copied into image)

```bash
docker stop mwi
docker rm mwi
docker build -t mwi:latest .
docker run -dit \
  --name mwi \
  -v ~/mywi_data:/app/data \
  mwi:latest
```

**Alternative: Pass via environment variables**

```bash
docker run -dit \
  --name mwi \
  -v ~/mywi_data:/app/data \
  -e MWI_SERPAPI_API_KEY="your-serpapi-key" \
  -e MWI_SEORANK_API_KEY="your-seorank-key" \
  -e MWI_OPENROUTER_ENABLED="true" \
  -e MWI_OPENROUTER_API_KEY="your-openrouter-key" \
  -e MWI_OPENROUTER_MODEL="deepseek/deepseek-chat-v3.1" \
  mwi:latest
```

---

### Level 3: Docker LLM (`install docker llm`)

**Complete Docker setup with ML capabilities.**

#### Step 1: Build Image with ML Dependencies

```bash
docker build \
  --build-arg WITH_ML=1 \
  -t mwi:latest .
```

This installs `requirements-ml.txt` during build.

#### Step 2: Configure Embeddings and NLI

Edit `settings.py`:

```python
# Embedding provider
embed_provider = 'mistral'
embed_mistral_api_key = "your-mistral-key"
embed_model_name = "mistral-embed"

# NLI model
nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
nli_backend_preference = "transformers"

# Similarity backend
similarity_backend = "faiss"
similarity_top_k = 50
```

**Or use environment variables:**

```bash
docker run -dit \
  --name mwi \
  -v ~/mywi_data:/app/data \
  -e MWI_EMBED_PROVIDER="mistral" \
  -e MWI_MISTRAL_API_KEY="your-mistral-key" \
  -e MWI_EMBED_MODEL="mistral-embed" \
  -e MWI_NLI_MODEL_NAME="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7" \
  -e MWI_SIMILARITY_BACKEND="faiss" \
  -e MWI_SIMILARITY_TOP_K="50" \
  mwi:latest
```

#### Step 3: Verify ML Installation

```bash
docker exec -it mwi python mywi.py embedding check
```

---

## Docker Compose Installation

**Recommended method - easiest setup with persistent data.**

### Prerequisites

- Docker Desktop (includes Docker Compose)

### Level 1: Basic Docker Compose (`install docker compose`)

#### Step 1: Clone Repository

```bash
git clone https://github.com/MyWebIntelligence/mwi.git
cd mwi
```

#### Step 2: Prepare Configuration Files

```bash
# Copy environment template
cp .env.example .env

# Copy settings template
cp settings-example.py settings.py

# Create data directory (optional, created automatically)
mkdir -p ./data
```

#### Step 3: Configure Data Location

Edit `.env`:

```bash
# Default: store in ./data inside repository
HOST_DATA_DIR=./data

# Alternative: absolute path outside repository
# macOS/Linux:
# HOST_DATA_DIR=/Users/you/mywi_data
# Windows:
# HOST_DATA_DIR=C:/Users/You/mywi_data
```

#### Step 4: Build and Start Services

```bash
# First run (builds image and starts container)
docker compose up -d --build

# Subsequent runs
docker compose up -d
```

#### Step 5: Initialize Database

```bash
docker compose exec mwi python mywi.py db setup
```

#### Step 6: Verify Installation

```bash
docker compose exec mwi python mywi.py land list
```

**Management commands:**
```bash
docker compose up -d          # Start services
docker compose down           # Stop services (data persists)
docker compose down -v        # Stop and DELETE data (destructive!)
docker compose restart        # Restart services
docker compose logs mwi       # View logs
docker compose exec mwi bash # Enter container shell
```

**Where is my data?**
- Host: `HOST_DATA_DIR` from `.env` (default: `./data`)
- Container: `/app/data` (mapped automatically)

---

### Level 2: Docker Compose API (`install docker compose api`)

**Docker Compose with API integrations.**

#### Step 1: Complete Basic Docker Compose Installation

Follow [Level 1: Basic Docker Compose](#level-1-basic-docker-compose-install-docker-compose) first.

#### Step 2: Configure API Keys in .env

Edit `.env` and add your API keys:

```bash
# --- SerpAPI bootstrap ---
MWI_SERPAPI_API_KEY=your-serpapi-key-here

# --- SEO Rank enrichment ---
MWI_SEORANK_API_KEY=your-seorank-key-here
MWI_SEORANK_API_BASE_URL=https://seo-rank.my-addr.com/api2/sr+fb

# --- OpenRouter relevance gate ---
MWI_OPENROUTER_ENABLED=true
MWI_OPENROUTER_API_KEY=your-openrouter-key-here
MWI_OPENROUTER_MODEL=deepseek/deepseek-chat-v3.1
MWI_OPENROUTER_TIMEOUT=15
MWI_OPENROUTER_MIN_CHARS=140
MWI_OPENROUTER_MAX_CHARS=12000
MWI_OPENROUTER_MAX_CALLS=500
```

#### Step 3: Restart Services

```bash
docker compose down
docker compose up -d
```

Environment variables are automatically injected into the container (see `docker-compose.yml`).

#### Step 4: Test API Access

```bash
# Test inside container
docker compose exec mwi bash
python -c "import settings; print('SerpAPI:', settings.serpapi_api_key[:10] + '...')"
python -c "import settings; print('SEO Rank:', settings.seorank_api_key[:10] + '...')"
python -c "import settings; print('OpenRouter:', settings.openrouter_enabled)"
exit
```

---

### Level 3: Docker Compose LLM (`install docker compose llm`)

**Complete Docker Compose setup with ML/LLM capabilities.**

#### Step 1: Enable ML Build Flag

Edit `.env` and enable ML dependencies:

```bash
# Build-time toggles
MYWI_WITH_ML=1                    # Enable ML extras (FAISS + transformers)
MYWI_WITH_PLAYWRIGHT_BROWSERS=0   # Optional: pre-install Playwright browsers (install libs manually)
```

If you set `MYWI_WITH_PLAYWRIGHT_BROWSERS=1`, install the runtime libraries inside the container before running crawls:

```bash
docker compose exec mwi bash -lc "apt-get update && apt-get install -y libnspr4 libnss3 libdbus-1-3 \
    libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libxcomposite1 libxdamage1 \
    libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libasound2"
docker compose exec mwi python install_playwright.py
```

#### Step 2: Configure Embedding Provider

Add to `.env`:

```bash
# --- Embeddings (bi-encoder) ---
MWI_EMBED_PROVIDER=mistral        # fake|http|openai|mistral|gemini|huggingface|ollama
MWI_EMBED_MODEL=mistral-embed
MWI_MISTRAL_API_KEY=your-mistral-key-here

# Alternative providers:
# OpenAI
# MWI_EMBED_PROVIDER=openai
# MWI_OPENAI_API_KEY=sk-...
# MWI_EMBED_MODEL=text-embedding-3-small

# Gemini
# MWI_EMBED_PROVIDER=gemini
# MWI_GEMINI_API_KEY=your-gemini-key
# MWI_EMBED_MODEL=embedding-001

# Ollama (local)
# MWI_EMBED_PROVIDER=ollama
# MWI_OLLAMA_BASE_URL=http://localhost:11434
# MWI_EMBED_MODEL=nomic-embed-text
```

#### Step 3: Configure NLI and Similarity

Add to `.env`:

```bash
# --- Semantic Search & NLI ---
MWI_NLI_MODEL_NAME=MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
MWI_NLI_BACKEND=auto             # auto|transformers|crossencoder|fallback
MWI_NLI_TORCH_THREADS=1
MWI_NLI_FALLBACK_MODEL_NAME=typeform/distilbert-base-uncased-mnli
MWI_SIMILARITY_BACKEND=faiss     # faiss|bruteforce
MWI_SIMILARITY_TOP_K=50
MWI_NLI_ENTAILMENT_THRESHOLD=0.8
MWI_NLI_CONTRADICTION_THRESHOLD=0.8
```

#### Step 4: Rebuild with ML Dependencies

```bash
docker compose down
docker compose up -d --build
```

This will install `requirements-ml.txt` during image build.

#### Step 5: Verify ML Installation

```bash
docker compose exec mwi python mywi.py embedding check
```

Expected output:
```
✓ Embedding provider: mistral
✓ Model: mistral-embed
✓ FAISS: available
✓ Transformers: available
✓ Sentence-Transformers: available
✓ NLI model: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
✓ Database tables: paragraph, paragraph_embedding, paragraph_similarity
```

#### Step 6: Pre-download NLI Models (optional)

```bash
docker compose exec mwi bash
python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
  model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'; \
  AutoTokenizer.from_pretrained(model_name); \
  AutoModelForSequenceClassification.from_pretrained(model_name); \
  print('✓ NLI model downloaded')"
exit
```

---

## Configuration Levels

### Comparison Table

| Feature | Basic | API | LLM |
|---------|-------|-----|-----|
| Web crawling | ✓ | ✓ | ✓ |
| Content extraction | ✓ | ✓ | ✓ |
| Media analysis | ✓ | ✓ | ✓ |
| Data export | ✓ | ✓ | ✓ |
| Google search (SerpAPI) | ✗ | ✓ | ✓ |
| SEO metrics | ✗ | ✓ | ✓ |
| AI relevance filter | ✗ | ✓ | ✓ |
| Embeddings | ✗ | ✗ | ✓ |
| Semantic search | ✗ | ✗ | ✓ |
| NLI classification | ✗ | ✗ | ✓ |
| Pseudolinks | ✗ | ✗ | ✓ |

### Use Cases

**Basic Installation** - For:
- Learning MyWI
- Small research projects
- Manual URL collection
- No external API dependencies

**API Installation** - For:
- Automated URL discovery (Google search)
- SEO/traffic metrics enrichment
- AI-powered content filtering
- Medium to large projects

**LLM Installation** - For:
- Semantic similarity analysis
- Cross-document relationship discovery
- Entailment/contradiction detection
- Advanced research workflows
- Large-scale corpus analysis

---

## Interactive Setup Scripts

### Overview

MyWI provides interactive Python scripts that guide you through configuration with clear prompts, validation, and automatic file generation.

### Available Scripts

| Script | Purpose | Generates |
|--------|---------|-----------|
| `scripts/install-basic.py` | Basic configuration | `settings.py` |
| `scripts/install-api.py` | API credentials | `settings.py` (updates) |
| `scripts/install-llm.py` | Embeddings & NLI | `settings.py` (updates) |
| `scripts/test-apis.py` | Validate API connections | - |

### Running Scripts

**Local installation:**
```bash
source .venv/bin/activate
python scripts/install-basic.py
```

**Docker installation:**
```bash
docker exec -it mwi python scripts/install-basic.py
```

**Docker Compose installation:**
```bash
docker compose exec mwi python scripts/install-basic.py
```

### Script Features

- **Clear prompts** with examples and defaults
- **Input validation** (paths, ranges, formats)
- **Secure input** for API keys (hidden)
- **Configuration preview** before saving
- **Backup** of existing `settings.py`
- **API testing** (optional)

### Example: install-basic.py

```
┌─────────────────────────────────────────────────┐
│  MyWebIntelligence - Basic Installation Setup  │
└─────────────────────────────────────────────────┘

This script will guide you through basic configuration.
Press Ctrl+C at any time to cancel.

[1/6] Data Storage
─────────────────────────────────────────────────
Where should MyWI store the database and exports?

Default: ./data
Examples:
  - ./data (current directory)
  - /Users/you/mywi_data (macOS/Linux)
  - C:/Users/You/mywi_data (Windows)

Data location: /Users/me/research/mywi_data
✓ Directory will be created if it doesn't exist

[2/6] Network Configuration
─────────────────────────────────────────────────
HTTP timeout for web requests (seconds)

Default: 10
Range: 5-60

Timeout [10]: 15
✓ Timeout set to 15 seconds

Concurrent HTTP connections

Default: 10
Range: 1-50
Recommendation: 10-20 for most use cases

Parallel connections [10]: 20
✓ Will use 20 parallel connections

...

[6/6] Configuration Summary
─────────────────────────────────────────────────
✓ Data location: /Users/me/research/mywi_data
✓ Network timeout: 15 seconds
✓ Parallel connections: 20
✓ User agent: (default)
✓ Dynamic media extraction: enabled
✓ Media analysis: enabled

Save this configuration? [Y/n]: y

✓ Backed up existing settings.py to settings.py.backup.20250101_143022
✓ Configuration saved to settings.py

Next steps:
  1. Initialize database: python mywi.py db setup
  2. Create your first land: python mywi.py land create --name="MyResearch"
  3. Add URLs: python mywi.py land addurl --land="MyResearch" --urls="https://example.com"
  4. Crawl: python mywi.py land crawl --name="MyResearch"
```

---

## Post-Installation

### Verification Checklist

After installation, verify everything works:

```bash
# 1. Check database connection
python mywi.py land list
# Expected: Empty list or existing lands

# 2. Check NLTK data
python -c "import nltk; nltk.data.find('tokenizers/punkt'); print('✓ NLTK ready')"

# 3. Check Mercury Parser (if installed)
mercury-parser --version
# Expected: @postlight/mercury-parser 2.x.x

# 4. Check Playwright (if installed)
python -c "from playwright.sync_api import sync_playwright; print('✓ Playwright ready')"

# 5. Check ML dependencies (LLM installation only)
python mywi.py embedding check
```

### Quick Start Workflow

Test your installation with a minimal workflow:

```bash
# 1. Create a test land
python mywi.py land create --name="TestLand" --desc="Testing installation"

# 2. Add test terms
python mywi.py land addterm --land="TestLand" --terms="climate change, global warming"

# 3. Add a test URL
python mywi.py land addurl --land="TestLand" --urls="https://en.wikipedia.org/wiki/Climate_change"

# 4. Crawl the URL
python mywi.py land crawl --name="TestLand"

# 5. Extract readable content (if Mercury installed)
python mywi.py land readable --name="TestLand"

# 6. Export data
python mywi.py land export --name="TestLand" --type=pagecsv

# 7. Check exports
ls data/exports/
# Expected: TestLand_pages_YYYYMMDD_HHMMSS.csv
```

### Configuration Files Location

| Installation Method | settings.py | .env | Database |
|---------------------|-------------|------|----------|
| Local | `./settings.py` | N/A | `$data_location/mwi.db` |
| Docker | `./settings.py` (copied to image) | N/A | `~/mywi_data/mwi.db` |
| Docker Compose | `./settings.py` | `./.env` | `$HOST_DATA_DIR/mwi.db` |

### Data Directory Structure

After initialization, your data directory will contain:

```
data/
├── mwi.db              # SQLite database
├── mwi.db-shm          # Shared memory (temporary)
├── mwi.db-wal          # Write-ahead log (temporary)
├── nltk_data/          # NLTK tokenizers
│   └── tokenizers/
│       ├── punkt/
│       └── punkt_tab/
└── exports/            # CSV, GEXF exports
    ├── TestLand_pages_20250101_120000.csv
    └── TestLand_corpus_20250101_120100.txt
```

---

## Troubleshooting

### Common Issues

#### 1. NLTK Download Errors

**Problem:** `Resource punkt_tab not found`

**Solution:**
```bash
# Manual download
python -m nltk.downloader punkt punkt_tab

# Or run install script
python install.py
```

**macOS SSL Certificate Error:**
```bash
# Install certificates
/Applications/Python\ 3.11/Install\ Certificates.command

# Or set environment variable
export SSL_CERT_FILE=$(python -c "import certifi; print(certifi.where())")
```

#### 2. Mercury Parser Not Found

**Problem:** `mercury-parser: command not found`

**Solution:**
```bash
# Install Node.js first
# macOS:
brew install node

# Ubuntu/Debian:
sudo apt install nodejs npm

# Windows: Download from https://nodejs.org/

# Then install Mercury Parser
sudo npm install -g @postlight/mercury-parser
```

#### 3. Playwright Installation Fails

**Problem:** `playwright install` hangs or fails

**Solution:**
```bash
# Manual installation
python -m playwright install chromium

# Or install specific browsers
python -m playwright install firefox

# Check installation
python -m playwright install --help
```

#### 4. Docker Volume Permissions

**Problem:** Permission denied accessing `/app/data`

**Solution:**
```bash
# Linux: fix ownership
sudo chown -R $(whoami):$(whoami) ~/mywi_data

# Or run container with your UID
docker run -dit \
  --name mwi \
  --user $(id -u):$(id -g) \
  -v ~/mywi_data:/app/data \
  mwi:latest
```

#### 5. ML Dependencies Installation Fails

**Problem:** `pip install -r requirements-ml.txt` fails

**Solution:**
```bash
# Update pip first
python -m pip install -U pip setuptools wheel

# Install PyTorch separately (CPU version)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Then install remaining ML deps
pip install sentence-transformers transformers faiss-cpu sentencepiece
```

#### 6. FAISS Import Error

**Problem:** `ImportError: cannot import name 'faiss'`

**Solution:**
```bash
# Uninstall conflicting packages
pip uninstall faiss faiss-cpu faiss-gpu -y

# Install correct CPU version
pip install faiss-cpu

# Verify
python -c "import faiss; print('FAISS version:', faiss.__version__)"
```

#### 7. API Key Not Recognized

**Problem:** API calls fail with "Invalid API key"

**Docker Compose:**
```bash
# Check environment variables are loaded
docker compose exec mwi env | grep MWI_

# Restart to reload .env
docker compose down
docker compose up -d
```

**Local:**
```bash
# Check settings.py
python -c "import settings; print(settings.serpapi_api_key)"

# Or use environment variables
export MWI_SERPAPI_API_KEY="your-key-here"
python mywi.py land urlist --name="Test" --query="test"
```

#### 8. Database Migration Needed

**Problem:** `no such column: expression.seorank`

**Solution:**
```bash
# Run migrations
python mywi.py db migrate

# Verify schema
python -c "from mwi.model import Expression; print([f.name for f in Expression._meta.fields.values()])"
```

#### 9. Out of Memory (Embeddings)

**Problem:** Process killed during `embedding similarity`

**Solution:**
```bash
# Reduce batch size in settings.py
embed_batch_size = 16  # instead of 32
nli_batch_size = 32    # instead of 64

# Use LSH method instead of exact
python mywi.py embedding similarity \
  --name=MyLand \
  --method=cosine_lsh \
  --lshbits=20 \
  --topk=10

# Filter by relevance
python mywi.py embedding similarity \
  --name=MyLand \
  --method=cosine \
  --minrel=2 \
  --maxpairs=100000
```

#### 10. Slow NLI Scoring

**Problem:** NLI similarity takes hours

**Solution:**
```bash
# Increase threads (settings.py)
nli_torch_num_threads = 4
# And set environment variable
export OMP_NUM_THREADS=4

# Reduce candidate pairs
python mywi.py embedding similarity \
  --name=MyLand \
  --method=nli \
  --topk=10 \        # instead of 50
  --minrel=2 \       # filter low relevance
  --maxpairs=50000   # hard cap

# Use faster model
# In settings.py:
nli_model_name = "typeform/distilbert-base-uncased-mnli"
```

---

### Getting Help

- **Documentation:** [README.md](../README.md) | [CLAUDE.md](../.claude/CLAUDE.md)
- **Issues:** https://github.com/MyWebIntelligence/mwi/issues
- **Logs:** Check container logs with `docker compose logs mwi`

---

### Environment Variables Reference

Quick reference for all `MWI_*` environment variables:

| Variable | Type | Default | Purpose |
|----------|------|---------|---------|
| `MYWI_DATA_DIR` | path | `data` | Data directory location |
| `MWI_OPENROUTER_ENABLED` | bool | `false` | Enable OpenRouter LLM gate |
| `MWI_OPENROUTER_API_KEY` | string | - | OpenRouter API key |
| `MWI_OPENROUTER_MODEL` | string | `deepseek/deepseek-chat-v3.1` | LLM model slug |
| `MWI_OPENROUTER_TIMEOUT` | int | `15` | Request timeout (seconds) |
| `MWI_OPENROUTER_MIN_CHARS` | int | `140` | Min text length for LLM |
| `MWI_OPENROUTER_MAX_CHARS` | int | `12000` | Max text length for LLM |
| `MWI_OPENROUTER_MAX_CALLS` | int | `500` | Max LLM calls per run |
| `MWI_SEORANK_API_BASE_URL` | url | - | SEO Rank API endpoint |
| `MWI_SEORANK_API_KEY` | string | - | SEO Rank API key |
| `MWI_SERPAPI_API_KEY` | string | - | SerpAPI key |
| `MWI_EMBED_PROVIDER` | string | `mistral` | Embedding provider |
| `MWI_EMBED_MODEL` | string | `mistral-embed` | Embedding model name |
| `MWI_EMBED_API_URL` | url | - | Custom HTTP endpoint |
| `MWI_OPENAI_API_KEY` | string | - | OpenAI API key |
| `MWI_MISTRAL_API_KEY` | string | - | Mistral API key |
| `MWI_GEMINI_API_KEY` | string | - | Google Gemini API key |
| `MWI_HF_API_KEY` | string | - | HuggingFace API key |
| `MWI_OLLAMA_BASE_URL` | url | `http://localhost:11434` | Ollama endpoint |
| `MWI_NLI_MODEL_NAME` | string | `MoritzLaurer/...` | NLI model |
| `MWI_NLI_BACKEND` | string | `fallback` | NLI backend preference |
| `MWI_NLI_TORCH_THREADS` | int | `1` | PyTorch threads |
| `MWI_NLI_FALLBACK_MODEL_NAME` | string | `typeform/...` | Fallback NLI model |
| `MWI_SIMILARITY_BACKEND` | string | `faiss` | ANN backend |
| `MWI_SIMILARITY_TOP_K` | int | `50` | Top K neighbors |
| `MWI_NLI_ENTAILMENT_THRESHOLD` | float | `0.8` | Entailment threshold |
| `MWI_NLI_CONTRADICTION_THRESHOLD` | float | `0.8` | Contradiction threshold |

---

**Installation complete!** Choose your installation method above and follow the steps for your desired configuration level.
