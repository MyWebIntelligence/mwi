# My Web Intelligence (MyWI)

This README is also available in French: [README_fr.md](README_fr.md)

MyWebIntelligence (MyWI) is a Python-based tool designed to assist researchers in digital humanities with creating and managing web-based research projects. It facilitates the collection, organization, and analysis of web data, storing information in a SQLite database. For browsing the database, a tool like [SQLiteBrowser](https://sqlitebrowser.org/) can be very helpful.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
  - [Using Docker](#using-docker)
  - [Local Development Setup](#local-development-setup)
- [Usage](#usage)
  - [General Notes](#general-notes)
  - [Land Management](#land-management)
  - [Data Collection](#data-collection)
  - [Domain Management](#domain-management)
  - [Exporting Data](#exporting-data)
  - [Heuristics](#heuristics)
- [Testing](#testing)
- [Helper Scripts](#helper-scripts)
- [SQLite Recovery](#sqlite-recovery)
- [License](#license)

## Features

*   **Land Creation & Management**: Organize your research into "lands," which are thematic collections of terms and URLs.
*   **Web Crawling**: Crawl URLs associated with your lands to gather web page content.
*   **Content Extraction**: Process crawled pages to extract readable content.
*   **SEO Rank Enrichment**: Query the SEO Rank API for each expression and keep the raw JSON payload alongside the URL.
*   **Embeddings & Pseudolinks**: Paragraph-level embeddings, semantic similarity, and CSV export of "pseudolinks" between semantically close paragraphs across pages.
*   **Media Analysis & Filtering**: Automatic extraction and analysis of images, videos, and audio. Extracts metadata (dimensions, size, format, dominant colors, EXIF), supports intelligent filtering and deletion, duplicate detection, and batch asynchronous processing.
*   **Enhanced Media Detection**: Detects media files with both uppercase and lowercase extensions (.JPG, .jpg, .PNG, .png, etc.).
*   **Dynamic Media Extraction**: Optional headless browser-based extraction for JavaScript-generated and lazy-loaded media content.
*   **Domain Analysis**: Gather information about domains encountered during crawling.
*   **Data Export**: Export collected data in various formats (CSV, GEXF, raw corpus) for further analysis.
*   **Tag-based Analysis**: Export tag matrices and content for deeper insights.

---

# Installation

**Three installation options:** Docker Compose (recommended), Docker manual, or Local Python.  
Run every command from the repository root unless stated otherwise. On Windows, use a Bash-capable terminal (Git Bash or WSL) for shell scripts; for Python commands use `python` or `py -3`.

> 📘 **Detailed guide:** See [docs/INSTALL_ZERO_bis.md](docs/INSTALL_ZERO_bis.md) for complete installation instructions with interactive setup scripts.

## Quick Start: Docker Compose (Recommended)

**One-command automated setup:**
```bash
./scripts/docker-compose-setup.sh [basic|api|llm]
```
`basic` is used by default when you omit the argument. Choose `api` to configure SerpAPI/SEO Rank/OpenRouter, or `llm` to also prepare the embeddings/NLI dependencies.

On Windows, run the script from a Bash-capable shell:
- Git Bash: `./scripts/docker-compose-setup.sh`
- PowerShell: `& "C:\Program Files\Git\bin\bash.exe" ./scripts/docker-compose-setup.sh`
- WSL: `wsl bash ./scripts/docker-compose-setup.sh`
Double-clicking the `.sh` file will not execute it.

**Or step-by-step (host terminal):**

1. Clone the project:
   ```bash
   git clone https://github.com/MyWebIntelligence/mwi.git
   cd mwi
   ```
2. Generate `.env` for Docker Compose (interactive wizard):
   ```bash
   python scripts/install-docker-compose.py
   ```
   On Windows you can also use `py -3 scripts/install-docker-compose.py`.
3. Build and start the container:
   ```bash
   docker compose up -d --build
   ```
4. Create `settings.py` **inside** the container (run once per environment):
   ```bash
   docker compose exec mwi bash -lc "cp settings-example.py settings.py"
   ```
   To customize settings interactively instead, run:
   ```bash
   docker compose exec -it mwi python scripts/install-basic.py --output settings.py
   ```
5. Initialize and verify the database:
   ```bash
   docker compose exec mwi python mywi.py db setup
   docker compose exec mwi python mywi.py land list
   ```

> ⚠️ `settings.py` is **not** created automatically inside the container.  
> Create it from within the container (copy `settings-example.py` or run `python scripts/install-basic.py`) before executing MyWI commands; the file stores environment-specific paths and keys and is intentionally excluded from version control and Docker layers.

**Where is my data?**

- Computer: `./data` (default) or path set in `.env`
- Container: `/app/data` (automatic mapping)

**Management:**
```bash
docker compose up -d       # Start
docker compose down        # Stop
docker compose logs mwi    # View logs
docker compose exec mwi bash  # Enter container
```

---

## Manual Docker (Advanced)

For quick tests or when Compose isn't available:
```bash
# Build
docker build -t mwi:latest .

# Run
docker run -dit --name mwi -v ~/mywi_data:/app/data mwi:latest

# Create settings.py in the container (first run)
docker exec mwi bash -lc "cp settings-example.py settings.py"
# Or customize:
# docker exec -it mwi python scripts/install-basic.py --output settings.py

# Initialize
docker exec -it mwi python mywi.py db setup

# Use
docker exec -it mwi python mywi.py land list
```

> ⚠️ Before running commands in the container, make sure `settings.py` exists (copy `settings-example.py` or run `python scripts/install-basic.py`). The project never auto-generates this file.

**Management:** `docker stop mwi` · `docker start mwi` · `docker rm mwi`

---

## Local Installation

**Prerequisites:** Python 3.10+, pip, git

**Quick setup:**
```bash
# 1. Clone and create environment
git clone https://github.com/MyWebIntelligence/mwi.git
cd mwi
python3 -m venv .venv
# Windows (PowerShell): py -3 -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate

# 2. Configure (interactive wizard)
python -m pip install -U pip setuptools wheel
python -m pip install -r requirements.txt
python scripts/install-basic.py

# 3. Initialize database
python mywi.py db setup

# 4. Verify
python mywi.py land list
```

**Optional steps:**

- **API configuration:** `python scripts/install-api.py`
- **LLM/embeddings:** `python -m pip install -r requirements-ml.txt && python scripts/install-llm.py`
- **Dynamic media (Playwright):**
  - Browsers: `python install_playwright.py`
  - Debian/Ubuntu libs: `sudo apt-get install libnspr4 libnss3 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 libatspi2.0-0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libasound2`
  - Docker: `docker compose exec mwi bash -lc "apt-get update && apt-get install -y <libs>"` then `docker compose exec mwi python install_playwright.py`

**Troubleshooting NLTK (Windows/macOS):**
```bash
python -m nltk.downloader punkt punkt_tab
# If SSL errors: pip install certifi
```

---

## Helper Scripts

**Quick starts**
- `scripts/docker-compose-setup.sh` — end-to-end Docker bootstrap (creates/backups `.env`, runs the wizard, builds, starts, initialises DB, and can smoke-test APIs/ML). Run `./scripts/docker-compose-setup.sh [basic|api|llm]`.

**Interactive configuration wizards**
- `scripts/install-docker-compose.py` — writes `.env` for Compose (timezone, host data path ↔ `/app/data`, Playwright/ML build flags, SerpAPI/SEO Rank/OpenRouter keys, embeddings/NLI defaults). Run `python scripts/install-docker-compose.py [--level basic|api|llm] [--output .env]`.
- `scripts/install-basic.py` — generates a minimal `settings.py` (storage path, network timeouts, concurrency, user agent, dynamic media, media analysis, default heuristics). Run `python scripts/install-basic.py [--output settings.py]`.
- `scripts/install-api.py` — records SerpAPI, SEO Rank, and OpenRouter credentials into `settings.py` (with env-var fallbacks). Run `python scripts/install-api.py [--output settings.py]`.
- `scripts/install-llm.py` — configures embeddings provider, NLI models/backends, retry and batching parameters after checking ML dependencies. Run `python scripts/install-llm.py [--output settings.py]`.

**Diagnostics & recovery**
- `scripts/test-apis.py` — validates configured API keys; supports `--serpapi`, `--seorank`, `--openrouter`, or `--all` (add `-v` for verbose). Run `python scripts/test-apis.py ...`.
- `scripts/sqlite_recover.sh` — non-destructive SQLite repair helper (see [SQLite Recovery](#sqlite-recovery)). Run `scripts/sqlite_recover.sh [INPUT_DB] [OUTPUT_DB]`.

**Utilities**
- `scripts/install-nltk.py` — downloads the `punkt` and `punkt_tab` tokenizers required by NLTK. Run `python scripts/install-nltk.py`.
- `scripts/crawl_robuste.sh` — sample retry loop around `land crawl`; edit the land name/limits before running. Execute with `bash scripts/crawl_robuste.sh`.
- `scripts/install_utils.py` — shared helper library for the interactive installers (not executable on its own).

---

# Usage

## General Notes

*   Commands are run using `python mywi.py ...`.
*   If using Docker, first execute `docker exec -it mwi bash` to enter the container. The prompt might be `root@<container_id>:/app#` or similar.

```bash
# Ensure the service is running
docker compose up -d
# Enter the container
docker compose exec mwi bash
# or
docker exec -it mwi bash
#  >>> Prompt typically looks like root@<container_id>:/app#

# Then run any application command
```

*   If using a local development setup, ensure your virtual environment is activated (e.g., `(venv)` prefix in your prompt).
*   Arguments like `LAND_NAME` or `TERMS` are placeholders; replace them with your actual values.

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Windows Command Prompt (cmd.exe)
.\.venv\Scripts\activate.bat

# Then run any application command
python mywi.py land list
```

## Land Management

A "Land" is a central concept in MyWI, representing a specific research area or topic.

---

### 1. Create a New Land

Create a new land (research topic/project).

```bash
python mywi.py land create --name="MyResearchTopic" --desc="A description of this research topic"
```

| Option      | Type   | Required | Default | Description                                 |
|-------------|--------|----------|---------|---------------------------------------------|
| --name      | str    | Yes      |         | Name of the land (unique identifier)        |
| --desc      | str    | No       |         | Description of the land                     |
| --lang      | str    | No       | fr      | Language code for the land (default: fr)    |

**Example:**
```bash
python mywi.py land create --name="AsthmaResearch" --desc="Research on asthma and air quality" --lang="en"
```

---

### 2. List Created Lands

List all lands or show properties of a specific land.

- List all lands:
  ```bash
  python mywi.py land list
  ```
- Show details for a specific land:
  ```bash
  python mywi.py land list --name="MyResearchTopic"
  ```

| Option   | Type | Required | Default | Description                      |
|----------|------|----------|---------|----------------------------------|
| --name   | str  | No       |         | Name of the land to show details |

---

### 3. Add Terms to a Land

Add keywords or phrases to a land.

```bash
python mywi.py land addterm --land="MyResearchTopic" --terms="keyword1, keyword2, related phrase"
```

| Option   | Type | Required | Default | Description                                 |
|----------|------|----------|---------|---------------------------------------------|
| --land   | str  | Yes      |         | Name of the land to add terms to            |
| --terms  | str  | Yes      |         | Comma-separated list of terms/keywords      |

---

### 4. Add URLs to a Land

Add URLs to a land, either directly or from a file.

- Directly:
  ```bash
  python mywi.py land addurl --land="MyResearchTopic" --urls="https://example.com/page1, https://anothersite.org/article"
  ```
- From a file (one URL per line):
  ```bash
  python mywi.py land addurl --land="MyResearchTopic" --path="/path/to/your/url_list.txt"
  ```
  *(If using Docker, ensure this file is accessible within the container, e.g., in your mounted data volume.)*

| Option   | Type | Required | Default | Description                                 |
|----------|------|----------|---------|---------------------------------------------|
| --land   | str  | Yes      |         | Name of the land to add URLs to             |
| --urls   | str  | No       |         | Comma-separated list of URLs to add         |
| --path   | str  | No       |         | Path to a file containing URLs (one per line) |

---

### 5. Gather URLs from SerpAPI (Google)

Bootstrap a land with URLs coming from Google search results via SerpAPI. Only
new URLs are inserted; existing entries keep their data but receive a title if
one was returned by the API.

```bash
python mywi.py land urlist --name="MyResearchTopic" --query="(gilets jaunes) OR (manifestation)" \
  --datestart=2023-01-01 --dateend=2023-03-31 --timestep=week
```

| Option      | Type  | Required | Default | Description |
|-------------|-------|----------|---------|-------------|
| --name      | str   | Yes      |         | Land receiving the URLs |
| --query     | str   | Yes      |         | Google search query (any valid Boolean string) |
| --datestart | str   | No       |         | Start of the date filter (`YYYY-MM-DD`) |
| --dateend   | str   | No       |         | End of the date filter (`YYYY-MM-DD`) |
| --timestep  | str   | No       | week    | Window size when iterating between dates (`day`, `week`, `month`) |
| --sleep     | float | No       | 1.0     | Base delay (seconds) between pages to respect rate limits |
| --lang      | str   | No       | fr      | Comma-separated language list; first value is used for SerpAPI |

> **API key** — populate `settings.serpapi_api_key` or export
> `MWI_SERPAPI_API_KEY` before running the command.

---

### 6. Delete a Land or Expressions

Delete an entire land or only expressions below a relevance threshold.

- Delete an entire land:
  ```bash
  python mywi.py land delete --name="MyResearchTopic"
  ```
- Delete expressions with relevance lower than a specific value:
  ```bash
  python mywi.py land delete --name="MyResearchTopic" --maxrel=MAXIMUM_RELEVANCE
  # e.g., --maxrel=0.5
  ```

| Option   | Type   | Required | Default | Description                                         |
|----------|--------|----------|---------|-----------------------------------------------------|
| --name   | str    | Yes      |         | Name of the land to delete                          |
| --maxrel | float  | No       |         | Only delete expressions with relevance < maxrel      |


## Data Collection

### 1. Crawl Land URLs

Crawl the URLs added to a land to fetch their content.

```bash
python mywi.py land crawl --name="MyResearchTopic" [--limit=NUMBER] [--http=HTTP_STATUS_CODE]
```

| Option   | Type   | Required | Default | Description                                                                 |
|----------|--------|----------|---------|-----------------------------------------------------------------------------|
| --name   | str    | Yes      |         | Name of the land whose URLs to crawl                                        |
| --limit  | int    | No       |         | Maximum number of URLs to crawl in this run                                 |
| --http   | str    | No       |         | Re-crawl only pages that previously resulted in this HTTP error (e.g., 503) |
| --depth  | int    | No       |         | Only crawl URLs that remain to be crawled at the specified depth            |

**Examples:**
```bash
python mywi.py land crawl --name="AsthmaResearch"
python mywi.py land crawl --name="AsthmaResearch" --limit=10
python mywi.py land crawl --name="AsthmaResearch" --http=503
python mywi.py land crawl --name="AsthmaResearch" --depth=2
python mywi.py land crawl --name="AsthmaResearch" --depth=1 --limit=5
```

> **Tip (Bash)** — Running multiple small batches can be faster than a single huge crawl. On macOS/Linux you can loop the crawler in one line:
> ```bash
> for i in {1..100}; do python mywi.py land crawl --name="melenchon" --depth=0 --limit=100; done
> ```

---

### 2. Fetch Readable Content (Mercury Parser Pipeline)

Extract high-quality, readable content using the **Mercury Parser autonomous pipeline**. This modern system provides intelligent content extraction with configurable merge strategies and automatic media/link enrichment.

**Prerequisites:** Requires `mercury-parser` CLI tool installed:
```bash
sudo npm install -g @postlight/mercury-parser
```

**Command:**
```bash
python mywi.py land readable --name="MyResearchTopic" [--limit=NUMBER] [--depth=NUMBER] [--merge=STRATEGY] [--llm=true|false]
```

| Option   | Type   | Required | Default | Description                                         |
|----------|--------|----------|---------|-----------------------------------------------------|
| --name   | str    | Yes      |         | Name of the land to process                         |
| --limit  | int    | No       |         | Maximum number of pages to process in this run      |
| --depth  | int    | No       |         | Maximum crawl depth to process (e.g., 2 = seeds + 2 levels) |
| --merge  | str    | No       | smart_merge | Merge strategy for content fusion (see below)    |
| --llm    | bool   | No       | false   | Enable OpenRouter relevance check (`true` to activate) |

**Merge Strategies:**

- **`smart_merge`** (default): Intelligent fusion based on field type
  - Titles: prefers longer, more informative titles
  - Content: Mercury Parser takes priority (cleaner extraction)
  - Descriptions: keeps the most detailed version
  
- **`mercury_priority`**: Mercury always overwrites existing data
  - Use for data migration or when Mercury extraction is preferred
  
- **`preserve_existing`**: Only fills empty fields, never overwrites
  - Safe option for enrichment without data loss

**Pipeline Features:**

- **High-Quality Extraction**: Mercury Parser provides excellent content cleaning
- **Bidirectional Logic**: 
  - Empty database + Mercury content → Fills from Mercury
  - Full database + Empty Mercury → Preserves database (abstains)
  - Full database + Full Mercury → Applies merge strategy
- **Automatic Enrichment**: 
  - Extracts and links media files (images, videos)
  - Creates expression links from discovered URLs
  - Updates metadata (author, publication date, language)
  - Recalculates relevance scores

**Examples:**
```bash
# Basic extraction with smart merge (default)
python mywi.py land readable --name="AsthmaResearch"

# Process only first 50 pages with depth limit
python mywi.py land readable --name="AsthmaResearch" --limit=50 --depth=2

# Mercury priority strategy (overwrites existing data)
python mywi.py land readable --name="AsthmaResearch" --merge=mercury_priority

# Conservative strategy (only fills empty fields)
python mywi.py land readable --name="AsthmaResearch" --merge=preserve_existing

# Advanced: Limited processing with specific strategy
python mywi.py land readable --name="AsthmaResearch" --limit=100 --depth=1 --merge=smart_merge

# Trigger OpenRouter validation (requires OpenRouter configuration)
python mywi.py land readable --name="AsthmaResearch" --llm=true
```

**Output:** The pipeline provides detailed statistics including:
- Number of expressions processed
- Success/error rates
- Update counts per field type
- Performance metrics

**Note:** This pipeline replaces the legacy readable functionality, providing better content quality, robustness, and flexible merge strategies for different use cases.

---

### 3. Capture SEO Rank Metrics

Fetch SEO Rank metrics for each expression and store the raw JSON payload in the database.

**Prerequisites:**

- Provide `seorank_api_key` in `settings.py` or export `MWI_SEORANK_API_KEY` before running the command.
- Optionally adjust `seorank_request_delay` to respect the provider's throttling policy (default: one second between calls).

**Command:**
```bash
python mywi.py land seorank --name="MyResearchTopic" [--limit=NUMBER] [--depth=NUMBER] [--force]
```

| Option   | Type    | Required | Default | Description |
|----------|---------|----------|---------|-------------|
| --name   | str     | Yes      |         | Land whose expressions will be enriched |
| --limit  | int     | No       |         | Maximum number of expressions to query in this run |
| --depth  | int     | No       |         | Restrict to expressions at a specific crawl depth |
| --http   | str     | No       | 200     | Filter by HTTP status (`all` to disable the filter) |
| --minrel | int     | No       | 1       | Only process expressions with `relevance` ≥ this value |
| --force  | boolean | No       | False   | Re-fetch even if `expression.seorank` already contains data |

**Behavior:**

- By default only expressions without SEO Rank data are selected. Use `--force` to refresh existing entries.
- `--http` defaults to `200`; pass `--http=all` (or `any`) to include every status code.
- `--minrel` defaults to `1`; set to `0` to include pages with relevance `0`.
- `--limit` applies after filtering; set it to keep the run short during testing.
- Each successful call stores the JSON response as-is in the `expression.seorank` column (text field).
- Errors or non-200 HTTP responses are logged and the command continues with the next URL.

**Example:**
```bash
# Enrich the first 100 seed URLs (depth 0) for the "AsthmaResearch" land
python mywi.py land seorank --name="AsthmaResearch" --depth=0 --limit=100

# Refresh every stored payload, regardless of current values
python mywi.py land seorank --name="AsthmaResearch" --force
```

**Tip:** Once data is stored you can inspect it directly via SQLite (`SELECT seorank FROM expression WHERE id=…`) or load it in Python with `json.loads` for downstream analysis.

**Payload fields (SEO Rank API):**
- `sr_domain` – domain that the metrics refer to.
- `sr_rank` – provider’s global SEO Rank score (lower values indicate stronger authority).
- `sr_kwords` – number of tracked keywords the domain currently ranks for.
- `sr_traffic` – estimated monthly organic visits attributed to the domain.
- `sr_costs` – estimated ad-equivalent cost (in USD) of the organic traffic.
- `sr_ulinks` – count of outgoing links found on the analysed URL.
- `sr_hlinks` – total backlinks pointing to the URL (all HTTP links).
- `sr_dlinks` – number of unique referring domains that link to the URL.
- `fb_comments` – Facebook comments recorded for the URL.
- `fb_shares` – Facebook share events recorded for the URL.
- `fb_reac` – Facebook reactions (likes, etc.) recorded for the URL.


### 4. Media Analysis

Analyze media files (images, videos, audio) associated with expressions in a land. This command will fetch media, analyze its properties, and store the results in the database.

```bash
python mywi.py land medianalyse --name=LAND_NAME [--depth=DEPTH] [--minrel=MIN_RELEVANCE]
```

| Option | Type | Required | Default | Description |
|---|---|---|---|---|
| `--name` | str | Yes | | Name of the land to analyze media for. |
| `--depth` | int | No | 0 | Only analyze media for expressions up to this crawl depth. |
| `--minrel` | float | No | 0.0 | Only analyze media for expressions with relevance greater than or equal to this value. |

**Example:**
```bash
python mywi.py land medianalyse --name="AsthmaResearch" --depth=2 --minrel=0.5
```

**Notes:**
- This process downloads media files to perform detailed analysis.
- Configuration for media analysis (e.g., `media_min_width`, `media_max_file_size`) can be found in `settings.py`.
- The results, including dimensions, file size, format, dominant colors, EXIF data, and perceptual hash, are stored in the database.

---

### 5. Crawl Domains

Get information from domains that were identified from expressions added to lands.

```bash
python mywi.py domain crawl [--limit=NUMBER] [--http=HTTP_STATUS_CODE]
```

| Option   | Type   | Required | Default | Description                                                                 |
|----------|--------|----------|---------|-----------------------------------------------------------------------------|
| --limit  | int    | No       |         | Maximum number of domains to crawl in this run                              |
| --http   | str    | No       |         | Re-crawl only domains that previously resulted in this HTTP error (e.g., 503) |

**Examples:**
```bash
python mywi.py domain crawl
python mywi.py domain crawl --limit=5
python mywi.py domain crawl --http=404
```

---

## Exporting Data

Export data from your lands or tags for analysis in other tools.

### 1. Export Land Data

Export data from a land in various formats.

`pagecsv` and `pagegexf` include any SEO Rank fields stored in `expression.seorank`; missing or `unknown` values are exported as `na`.


```bash
python mywi.py land export --name="MyResearchTopic" --type=EXPORT_TYPE [--minrel=MINIMUM_RELEVANCE]
```

| Option   | Type   | Required | Default | Description                                                                 |
|----------|--------|----------|---------|-----------------------------------------------------------------------------|
| --name   | str    | Yes      |         | Name of the land to export                                                  |
| --type   | str    | Yes      |         | Export type (see below)                                                     |
| --minrel | float  | No       |         | Minimum relevance for expressions to be included in the export              |

**EXPORT_TYPE values:**
- `pagecsv`: CSV of pages
- `pagegexf`: GEXF graph of pages
- `fullpagecsv`: CSV with full page content
- `nodecsv`: CSV of nodes
- `nodegexf`: GEXF graph of nodes
- `mediacsv`: CSV of media links
- `corpus`: Raw text corpus
- `pseudolinks`: CSV of semantic paragraph pairs (source/target expression, domain, paragraph indices, relation score, confidence, snippets)
- `pseudolinkspage`: CSV of page‑level aggregated pseudolinks (expression↔expression). Columns: Source_ExpressionID, Target_ExpressionID, Source_DomainID, Target_DomainID, PairCount, EntailCount, NeutralCount, ContradictCount, AvgRelationScore, AvgConfidence.
- `pseudolinksdomain`: CSV of domain‑level aggregated pseudolinks (domain↔domain). Columns: Source_DomainID, Source_Domain, Target_DomainID, Target_Domain, PairCount, EntailCount, NeutralCount, ContradictCount, AvgRelationScore, AvgConfidence.

**Examples:**
```bash
python mywi.py land export --name="AsthmaResearch" --type=pagecsv
python mywi.py land export --name="AsthmaResearch" --type=corpus --minrel=0.7
python mywi.py land export --name="AsthmaResearch" --type=pseudolinks
python mywi.py land export --name="AsthmaResearch" --type=pseudolinkspage
python mywi.py land export --name="AsthmaResearch" --type=pseudolinksdomain
```

---

### 2. Export Tag Data

Export tag-based data for a land.

```bash
python mywi.py tag export --name="MyResearchTopic" --type=EXPORT_TYPE [--minrel=MINIMUM_RELEVANCE]
```

| Option   | Type   | Required | Default | Description                                                                 |
|----------|--------|----------|---------|-----------------------------------------------------------------------------|
| --name   | str    | Yes      |         | Name of the land whose tags to export                                       |
| --type   | str    | Yes      |         | Export type (see below)                                                     |
| --minrel | float  | No       |         | Minimum relevance for tag content to be included in the export              |

**EXPORT_TYPE values:**
- `matrix`: Tag co-occurrence matrix
- `content`: Content associated with tags

**Examples:**
```bash
python mywi.py tag export --name="AsthmaResearch" --type=matrix
python mywi.py tag export --name="AsthmaResearch" --type=content --minrel=0.5
```

---

## Update Domains from Heuristic Settings

Update domain information based on predefined or learned heuristics.

```bash
python mywi.py heuristic update
```

_No options for this command._

## Land Consolidation Pipeline

The `land consolidate` pipeline is designed to re-compute and repair the internal structure of a land after the database has been modified by third-party applications (such as MyWebClient) or external scripts.

**Purpose:**  
- Recalculates the relevance score for each crawled page (expressions with a non-null `fetched_at`).
- Re-extracts and recreates all outgoing links (ExpressionLink) and media (Media) for these pages.
- Adds any missing documents referenced by links.
- Rebuilds the link graph and media associations from scratch, replacing any outdated or inconsistent data.

**When to use:**  
- After importing or modifying data in the database with external tools (e.g., MyWebClient).
- To restore consistency if links or media are out of sync with the actual page content.

**Command:**
```bash
python mywi.py land consolidate --name=LAND_NAME [--limit=LIMIT] [--depth=NbDEEP]
```
- `--name` (required): Name of the land to consolidate.
- `--limit` (optional): Maximum number of pages to process.
- `--depth` (optional): Only process pages at the specified crawl depth.

**Example:**
```bash
python mywi.py land consolidate --name="AsthmaResearch" --depth=0
```

**Notes:**
- Only pages that have already been crawled (`fetched_at` is set) are affected.
- For each page, the number of extracted links and media is displayed.
- This pipeline is especially useful after bulk imports, migrations, or when using third-party clients that may not maintain all MyWI invariants.

---

## Testing

To run tests for the project:
```bash
pytest tests/
```
To run a specific test file:
```bash
pytest tests/test_cli.py
```
To run a specific test method within a file:
```bash
pytest tests/test_cli.py::test_functional_test
```

#  Embeddings & Pseudolinks (User Guide)

## Purpose
- Build paragraph‑level vectors (embeddings) from pages, then link similar paragraphs across pages (“pseudolinks”).
- Optionally classify each pair with an NLI model (entailment/neutral/contradiction).
- Export paragraph links, plus aggregated links at page and domain levels.

Typical flow
1) Crawl + extract readable text
2) Generate embeddings (paragraph vectors)
3) Compute similarities (cosine or ANN+NLI)
4) Export as CSV (paragraph/page/domain)

## Prerequisites & Install
- Database initialized and pages have readable text.
- Create a clean pip‑only venv and install base deps:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip setuptools wheel
  python -m pip install -r requirements.txt
  ```
- Optional (NLI + FAISS acceleration):
  ```bash
  python -m pip install -r requirements-ml.txt
  ```
- Quick environment check:
  ```bash
  python mywi.py embedding check
  ```


## Models
-Pseudo Links
- Multilingual (recommended):
  - MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
- Lightweight fallback (English):
  - typeform/distilbert-base-uncased-mnli
- Set in `settings.py:nli_model_name` (both examples are documented there).

## Settings (Key Reference)
- Embeddings (bi‑encoder):
  - `embed_provider`: 'fake' | 'http' | 'openai' | 'mistral' | 'gemini' | 'huggingface' | 'ollama'
  - `embed_model_name`, `embed_batch_size`, `embed_min_paragraph_chars`, `embed_max_paragraph_chars`
  - `embed_similarity_method`: 'cosine' | 'cosine_lsh'
  - `embed_similarity_threshold` (for cosine‑based methods)
- ANN recall and NLI:
  - `similarity_backend`: 'faiss' | 'bruteforce'
  - `similarity_top_k`: neighbors per paragraph for ANN recall
  - `nli_model_name`, `nli_fallback_model_name`
  - `nli_backend_preference`: 'auto' | 'transformers' | 'crossencoder' | 'fallback'
  - `nli_batch_size`, `nli_max_tokens`
  - `nli_torch_num_threads`: Torch threads (also set `OMP_NUM_THREADS` at runtime)
  - `nli_progress_every_pairs`, `nli_show_throughput`
- CPU env vars (export in your shell):
  - `OMP_NUM_THREADS=N` (FAISS/Torch/NumPy OpenMP threads)
  - Optional: `MKL_NUM_THREADS=N`, `OPENBLAS_NUM_THREADS=N`, `TOKENIZERS_PARALLELISM=false`

## Commands & Parameters
- Generate embeddings:
  ```bash
  python mywi.py embedding generate --name=LAND [--limit N]
  ```
- Compute similarities (pick one):
  - Cosine (exact):
    ```bash
    python mywi.py embedding similarity --name=LAND --method=cosine \
      --threshold=0.85 [--minrel R]
    ```
  - Cosine LSH (approximate):
    ```bash
    python mywi.py embedding similarity --name=LAND --method=cosine_lsh \
      --lshbits=20 --topk=15 --threshold=0.85 [--minrel R] [--maxpairs M]
    ```
  - ANN + NLI:
    ```bash
    python mywi.py embedding similarity --name=LAND --method=nli \
      --backend=faiss|bruteforce --topk=10 [--minrel R] [--maxpairs M]
    ```
- Export CSVs:
  - Paragraph pairs:
    ```bash
    python mywi.py land export --name=LAND --type=pseudolinks
    ```
  - Page‑level aggregation:
    ```bash
    python mywi.py land export --name=LAND --type=pseudolinkspage
    ```
  - Domain‑level aggregation:
    ```bash
    python mywi.py land export --name=LAND --type=pseudolinksdomain
    ```
- Utilities:
  - Check env: `python mywi.py embedding check`
  - Reset embeddings for a land: `python mywi.py embedding reset --name=LAND`

## Troubleshooting & Caution
- “All `score_raw=0.5` and `score=0`” → neutral fallback; install ML extras or switch to the safe EN model.
- “No `score_raw` column” → run `python mywi.py db migrate` once.
- macOS segfaults (OpenMP/Torch): pip‑only venv; try `OMP_NUM_THREADS=1`, then raise; optional `KMP_DUPLICATE_LIB_OK=TRUE`.
- Slow scoring: lower `nli_batch_size`, raise threads moderately, filter with `--minrel`, cap with `--maxpairs`.
- Too many pairs: raise `threshold`, increase `lshbits`, lower `topk`, or use `--minrel`.

## Best Practices — Performance

Quick guidelines for speed vs. quality:

- Small/medium size (≤ ~50k paragraphs)
  - Simple and fast method: `cosine` with `--threshold=0.85` and `--minrel=1`.
  - Example:
    ```bash
    python mywi.py embedding similarity --name=LAND --method=cosine \
      --threshold=0.85 --minrel=1
    ```

- Large size (≥ ~100k paragraphs)
  - Prefer `cosine_lsh` (approx) and bound the fan-out and output:
    - `--lshbits=18–22` (20 default)
    - `--topk=10–20`
    - `--threshold=0.85–0.90`
    - `--minrel=1–2`
    - `--maxpairs` to cap the total number of pairs (e.g., 5–10M)
  - Example:
    ```bash
    python mywi.py embedding similarity --name=LAND --method=cosine_lsh \
      --lshbits=20 --topk=15 --threshold=0.88 --minrel=1 --maxpairs=8000000
    ```

- NLI (ANN + Cross‑Encoder)
  - Use FAISS for recall if available: `--backend=faiss`.
  - Start small: `--topk=6–10`, `--minrel=1–2`, `--maxpairs=20k–200k`.
  - Choose the model:
    - Smoke test/quick CPU: DistilBERT MNLI (EN) → `typeform/distilbert-base-uncased-mnli`.
    - Multilingual quality: DeBERTa XNLI → `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (requires `sentencepiece`).
  - Tune:
    - `nli_batch_size=32–96` depending on RAM.
    - `nli_max_tokens=384–512` if you want to truncate a bit more for speed.
  - Example:
    ```bash
    python mywi.py embedding similarity --name=LAND --method=nli \
      --backend=faiss --topk=8 --minrel=2 --maxpairs=20000
    ```

- CPU & threads (in your venv)
  - Set threads: `export OMP_NUM_THREADS=N` (FAISS/Torch/NumPy),
    and `settings.py:nli_torch_num_threads = N` (Torch intra‑op).
  - Rule of thumb: N = (available cores − 1) to keep system headroom.
  - Keep `TOKENIZERS_PARALLELISM=false` to avoid unnecessary overhead.

- Throughput & logs
  - Track progress every `nli_progress_every_pairs` pairs, with throughput (pairs/s) and ETA.
  - If throughput is low, lower `nli_max_tokens` or `nli_batch_size`, and/or raise `--minrel`.

## Model Choice and Fallbacks

- Default NLI model can be multilingual (DeBERTa‑based) and may require `sentencepiece`.
- Safe alternative (English): `typeform/distilbert-base-uncased-mnli`.
- Configure in `settings.py:nli_model_name`.
- If dependencies are missing, the code can fall back to a neutral predictor (`score=0`, `score_raw=0.5`).

## Progress & Logs

- Recall logs every few hundred paragraphs (candidate pairs count).
- NLI scoring logs progress every `settings.nli_progress_every_pairs` pairs with throughput and ETA.
- Final summary prints total pairs, elapsed time, and pairs/s.

## Similarity Methods

Pick a method with `--method` when running `embedding similarity`:

- `cosine`: exact pairwise cosine (O(n²)) on embeddings.
  - Good for small/medium sets. Uses `--threshold` and optional `--minrel`.
  - Does not use FAISS.
- `cosine_lsh`: approximate, LSH hyperplane bucketing + local brute-force.
  - Scales well without external libs. Uses `--lshbits`, `--topk`, `--threshold`, `--minrel`, `--maxpairs`.
  - Does not use FAISS.
- `nli` (aliases: `ann+nli`, `semantic`): two-step ANN + Cross‑Encoder NLI.
  - Step 1 (Recall): ANN top‑k per paragraph using FAISS if available, otherwise brute‑force.
  - Step 2 (Precision): Cross‑Encoder NLI returns RelationScore ∈ {-1, 0, 1} and ConfidenceScore.
  - Uses `--backend`, `--topk`, `--minrel`, `--maxpairs`. See below for FAISS.

## ANN Backend Selection (FAISS)

- Install FAISS (optional): `pip install faiss-cpu`.
- CLI override: `--backend=faiss` to force FAISS recall for `--method=nli`.
- Settings default: `similarity_backend = 'faiss'` to prefer FAISS when no `--backend` is specified.
- Fallback: if FAISS is not installed or import fails, recall uses `bruteforce` automatically.
- Verify: `python mywi.py embedding check` prints `FAISS: available` when detected.

## Scalable Similarity (Large Lands)

For large collections (hundreds of thousands to millions of paragraphs), prefer the LSH-based method and constrain search/output:

```bash
# LSH buckets + per-paragraph top-k + hard cap of total pairs
python mywi.py embedding similarity \
  --name=MyResearchTopic \
  --method=cosine_lsh \
  --threshold=0.85 \
  --lshbits=20 \
  --topk=15 \
  --minrel=1 \
  --maxpairs=5000000
```

- `--method=cosine_lsh`: Approximate search using random hyperplanes; reduces candidate pairs drastically.
- `--lshbits`: Number of hyperplanes/bits (higher → finer buckets, e.g., 18–22).
- `--topk`: Keep only the top-K neighbors per paragraph (limits per-source fanout).
- `--threshold`: Cosine threshold; raising it reduces pair count.
- `--minrel`: Filter paragraphs by expression relevance (skip low-value content).
- `--maxpairs`: Hard cap on pairs written to DB.

Tuning suggestions:
- Start with `--lshbits=20`, `--topk=10–20`, `--threshold=0.85`, `--minrel=1`.
- If too many pairs, increase `lshbits`, raise `threshold`, or lower `topk`.

## NLI Relations (ANN + Cross‑Encoder)

Classify logical relations between paragraphs (entailment/paraphrase = 1, neutral = 0, contradiction = -1) using a two‑step pipeline: ANN recall then Cross‑Encoder NLI.

Prerequisites (optional, installed only if you need NLI or faster ANN):
```bash
pip install sentence-transformers transformers  # Cross-Encoder NLI
# For faster ANN recall (optional):
pip install faiss-cpu
```

Command example:
```bash
python mywi.py embedding similarity \
  --name=MyResearchTopic \
  --method=nli \
  --backend=bruteforce    # or faiss if installed \
  --topk=50               # candidates per paragraph from ANN \
  --minrel=1              # optional relevance filter \
  --maxpairs=2000000      # optional safety cap
```

Settings touch-points:
- `nli_model_name` (default: MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
- `nli_batch_size` (default: 64)
- `similarity_backend` ('bruteforce' | 'faiss')
- `similarity_top_k` (default: 50)

Quick recipes:
- Exact cosine (small set):
  ```bash
  python mywi.py embedding similarity --name=MyResearchTopic --method=cosine --threshold=0.85 --minrel=1
  ```
- Approx cosine (large set, no deps):
  ```bash
  python mywi.py embedding similarity --name=MyResearchTopic --method=cosine_lsh --lshbits=20 --topk=15 --threshold=0.85 --minrel=1 --maxpairs=5000000
  ```
- ANN + NLI with FAISS:
  ```bash
  pip install sentence-transformers transformers faiss-cpu
  python mywi.py embedding similarity --name=MyResearchTopic --method=nli --backend=faiss --topk=50 --minrel=1 --maxpairs=2000000
  ```

CSV export (pseudolinks):
- `python mywi.py land export --name=MyResearchTopic --type=pseudolinks`
- Columns: `Source_ParagraphID, Target_ParagraphID, RelationScore, ConfidenceScore, Source_Text, Target_Text, Source_ExpressionID, Target_ExpressionID`

Quick environment check:
```bash
python mywi.py embedding check
```

Shows provider config, optional libs (faiss/sentence-transformers/transformers), and DB tables availability.

# Troubleshooting & repairing

## Keep the database schema current

When pulling a newer version of MyWI, make sure your existing database has the latest columns and indexes.

```bash
python mywi.py db migrate
```

This command is idempotent: it inspects `data/mwi.db` (or the location specified via `MYWI_DATA_DIR`) and adds any missing fields. Run it after every upgrade or before sharing a database. For safety, back up the file first:

```bash
cp data/mwi.db data/mwi.db.bak_$(date +%Y%m%d_%H%M%S)
```

## SQLite Recovery

If your SQLite database becomes corrupted (e.g., "database disk image is malformed"), you can attempt a non-destructive recovery with the included helper script. It backs up the original DB, tries `sqlite3 .recover` (then `.dump` as a fallback), rebuilds a new DB, and verifies integrity.

Prerequisites:
- `sqlite3` available in your shell.

Steps:
```bash
chmod +x scripts/sqlite_recover.sh
# Usage: scripts/sqlite_recover.sh [INPUT_DB] [OUTPUT_DB]
scripts/sqlite_recover.sh data/mwi.db data/mwi_repaired.db
```

What it does:
- Backs up `data/mwi.db` (+ `-wal` / `-shm` if present) to `data/sqlite_repair_<timestamp>/backup/`
- Attempts `.recover` first, falls back to `.dump` into `data/sqlite_repair_<timestamp>/dump/`
- Rebuilds `data/mwi_repaired.db`, runs `PRAGMA integrity_check;` and lists tables under `data/sqlite_repair_<timestamp>/logs/`

Validate the repaired DB with MyWI without replacing the original:
```bash
mkdir -p data/test-repaired
cp data/mwi_repaired.db data/test-repaired/mwi.db
MYWI_DATA_DIR="$PWD/data/test-repaired" python mywi.py land list
```

If everything looks good, adopt the repaired DB (after a manual backup):
```bash
cp data/mwi.db data/mwi.db.bak_$(date +%Y%m%d_%H%M%S)
mv data/mwi_repaired.db data/mwi.db
```

Note: You can temporarily point the app to a different data directory using the `MYWI_DATA_DIR` environment variable; it overrides `settings.py:data_location` for that session.

# For Developpers

## Architecture & Internals

### File Structure & Flow

```
mywi.py  →  mwi/cli.py  →  mwi/controller.py  →  mwi/core.py & mwi/export.py
                                     ↘︎ mwi/model.py (Peewee ORM)
                                     ↘︎ mwi/embedding_pipeline.py (paragraph embeddings)
```

- **mywi.py**: Console entry-point, runs CLI.
- **mwi/cli.py**: Parses CLI args, dispatches commands to controllers.
- **mwi/controller.py**: Maps verbs to business logic in core/export/model.
- **mwi/core.py**: Main algorithms (crawling, parsing, pipelines, scoring, etc.).
- **mwi/export.py**: Exporters (CSV, GEXF, corpus).
- **mwi/model.py**: Database schema (Peewee ORM).

### Database Schema (SQLite, via Peewee)

- **Land**: Research project/topic.
- **Word**: Normalized vocabulary.
- **LandDictionary**: Many-to-many Land/Word.
- **Domain**: Unique website/domain.
- **Expression**: Individual URL/page.
- **ExpressionLink**: Directed link between Expressions.
- **Media**: Images, videos, audio in Expressions.
- **Paragraph / ParagraphEmbedding / ParagraphSimilarity**: Paragraph store, embeddings, and semantic links (pseudolinks).
- **Tag**: Hierarchical tags.
- **TaggedContent**: Snippets tagged in Expressions.
  - Expression extra fields: `validllm` (yes/no), `validmodel` (OpenRouter model used), and `seorank` (raw SEO Rank API JSON)

### Main Workflows

- **Project Bootstrap**: `python mywi.py db setup`
- **Media Analysis**: `python mywi.py land medianalyse --name=LAND_NAME [--depth=DEPTH] [--minrel=MIN_RELEVANCE]`
- **Land Life-Cycle**: Create, add terms, add URLs, crawl, extract readable, export, clean/delete.
- **SEO Rank Enrichment**: `python mywi.py land seorank --name=LAND [--limit N] [--depth D] [--force]`
- **Domain Processing**: `python mywi.py domain crawl`
- **Tag Export**: `python mywi.py tag export`
- **Heuristics Update**: `python mywi.py heuristic update`
- **Embeddings & Similarity**:
  - Generate: `python mywi.py embedding generate --name=LAND [--limit N]`
  - Similarity: `python mywi.py embedding similarity --name=LAND [--threshold 0.85] [--method cosine]`

### Implementation Notes

- **Relevance Score**: Weighted sum of lemma hits in title/content.
- **Async Batching**: Polite concurrency for crawling.
- **Media Extraction**: Only `.jpg` images kept, media saved for later download.
- **Export**: Multiple formats, dynamic SQL, GEXF with attributes.

### Settings

Key variables in `settings.py`:
- `data_location`, `user_agent`, `parallel_connections`, `default_timeout`, `archive`, `heuristics`.

#### Embeddings configuration
- `embed_provider`: 'fake' (local deterministic) or 'http'
- Providers supported: `fake`, `http`, `openai`, `mistral`, `gemini`, `huggingface`, `ollama`
- `embed_api_url`: URL for generic HTTP provider (POST {"model": name, "input": [texts...]})
- `embed_model_name`: model label stored alongside vectors
- `embed_batch_size`: batch size when calling the provider
- `embed_min_paragraph_chars` / `embed_max_paragraph_chars`: paragraph length bounds
- `embed_similarity_threshold` / `embed_similarity_method`: similarity gate and method
  
Provider-specific keys:
- OpenAI: `embed_openai_base_url` (default `https://api.openai.com/v1`), `embed_openai_api_key`
- Mistral: `embed_mistral_base_url` (default `https://api.mistral.ai/v1`), `embed_mistral_api_key`
- Gemini: `embed_gemini_base_url` (default `https://generativelanguage.googleapis.com/v1beta`), `embed_gemini_api_key` (query param)
- Hugging Face: `embed_hf_base_url` (default `https://api-inference.huggingface.co/models`), `embed_hf_api_key`
- Ollama: `embed_ollama_base_url` (default `http://localhost:11434`)
  
Notes:
- OpenAI/Mistral expect payload `{ "model": name, "input": [texts...] }` and return `{ "data": [{"embedding": [...]}, ...] }`.
- Gemini uses `:batchEmbedContents` and returns `{ "embeddings": [{"values": [...]}, ...] }`.
- Hugging Face accepts `{ "inputs": [texts...] }` and typically returns a list of vectors.
- Ollama (local) does not batch: sequential calls to `/api/embeddings` with `{ "model": name, "prompt": text }`.

#### Optional: OpenRouter Relevance Gate (AI yes/no filter)

If enabled, pages are first judged by an LLM (via OpenRouter) as relevant (yes) or not (no). A "no" sets `relevance=0` and skips further processing; otherwise, the classic weighted relevance is computed. This applies during crawl/readable/consolidation, but not during bulk recomputation (`land addterm`).

Environment-configurable variables:
- `MWI_OPENROUTER_ENABLED` (default `false`)
- `MWI_OPENROUTER_API_KEY`
- `MWI_OPENROUTER_MODEL` (e.g. `openai/gpt-4o-mini`, `anthropic/claude-3-haiku`)
- `MWI_OPENROUTER_TIMEOUT` (default `15` seconds)
- `MWI_OPENROUTER_READABLE_MAX_CHARS` (default `6000`)
- `MWI_OPENROUTER_MAX_CALLS_PER_RUN` (default `500`)

Note: When disabled or not configured, the system behaves exactly as before.

#### Bulk LLM Validation (yes/no)
Validate relevance in bulk via OpenRouter and record the verdict in DB (`expression.validllm`, `expression.validmodel`).

Command:
```bash
python mywi.py land llm validate --name=LAND [--limit N] [--force]
```

Requirements:
- In `settings.py`: set `openrouter_enabled=True`, and provide `openrouter_api_key` and `openrouter_model`.
- If your DB is old: `python mywi.py db migrate` (adds columns if missing).

Behavior:
- For each expression without a verdict, call the LLM to answer yes/no.
- Saves `validllm` = `"oui"|"non"` (French) and `validmodel` = model slug.
  - Filtering: only processes expressions with no "oui/non" verdict where `readable` is NOT NULL and has length ≥ `openrouter_readable_min_chars`.
  - Respects `openrouter_readable_min_chars`, `openrouter_readable_max_chars` and `openrouter_max_calls_per_run`.
  - If the verdict is `"non"`, the expression's `relevance` is set to `0`.

`--force` option:
- Also includes expressions with an existing `"non"` verdict in the selection (does not include `"oui"`).

#### SEO Rank enrichment

The `land seorank` command enriches each expression with the raw SEO Rank API payload. Configure these keys in `settings.py` (or via environment variables):

- `seorank_api_base_url`: Base endpoint (defaults to `https://seo-rank.my-addr.com/api2/moz+sr+fb`).
- `seorank_api_key`: Required API key (`MWI_SEORANK_API_KEY` overrides the default).
- `seorank_timeout`: Request timeout in seconds.
- `seorank_request_delay`: Pause between calls to stay polite with the provider.

By default the command only targets expressions with `http_status = 200` and `relevance ≥ 1`; pass `--http=all` or `--minrel=0` to broaden the selection.

Without a valid API key the command exits early. Use `--force` to refresh entries that already contain data in `expression.seorank`.

#### SerpAPI bootstrap (`land urlist`)

The `land urlist` command queries a SerpAPI search engine (Google by default;
override with `--engine=bing|duckduckgo`) and pushes new URLs to a land.
Configure the following values in `settings.py` or via environment
variables:

- `serpapi_api_key`: Required API key (`MWI_SERPAPI_API_KEY` overrides the default).
- `serpapi_base_url`: Base endpoint (defaults to `https://serpapi.com/search`).
- `serpapi_timeout`: HTTP timeout in seconds.

Date filters (`--datestart`, `--dateend`, `--timestep`) are optional but must be
provided as valid `YYYY-MM-DD` strings when used, and they require
`--engine=google` or `--engine=duckduckgo`. The command sleeps between
pages (`--sleep`) to avoid rate limits; set it to `0` for tests/mocks only.
When a date range is provided (or when you add `--progress`), the CLI prints one
line per window indicating the covered dates and how many URLs SerpAPI returned.

### Testing

- `tests/test_cli.py`: CLI smoke tests.
- `tests/test_core.py`, etc.: Unit tests for extraction, parsing, export.

### Extending

- Add export: implement `Export.write_<type>`, update controller.
- Change language: pass `--lang` at land creation.
- Add headers/proxy: edit `settings` or patch session logic.
- Custom tags: use tag hierarchy, export flattens to paths.

---

# License

This project is licensed under the terms of the LICENSE file. (Assuming a LICENSE file exists in the repository, e.g., MIT, Apache 2.0).
If `LICENSE` is the actual name of the file, you can link to it: [LICENSE](LICENSE).
