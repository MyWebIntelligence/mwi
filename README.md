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
- [SQLite Recovery](#sqlite-recovery)
- [License](#license)

## Features

*   **Land Creation & Management**: Organize your research into "lands," which are thematic collections of terms and URLs.
*   **Web Crawling**: Crawl URLs associated with your lands to gather web page content.
*   **Content Extraction**: Process crawled pages to extract readable content.
*   **Embeddings & Pseudolinks**: Paragraph-level embeddings, semantic similarity, and CSV export of "pseudolinks" between semantically close paragraphs across pages.
*   **Media Analysis & Filtering**: Automatic extraction and analysis of images, videos, and audio. Extracts metadata (dimensions, size, format, dominant colors, EXIF), supports intelligent filtering and deletion, duplicate detection, and batch asynchronous processing.
*   **Enhanced Media Detection**: Detects media files with both uppercase and lowercase extensions (.JPG, .jpg, .PNG, .png, etc.).
*   **Dynamic Media Extraction**: Optional headless browser-based extraction for JavaScript-generated and lazy-loaded media content.
*   **Domain Analysis**: Gather information about domains encountered during crawling.
*   **Data Export**: Export collected data in various formats (CSV, GEXF, raw corpus) for further analysis.
*   **Tag-based Analysis**: Export tag matrices and content for deeper insights.

---

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

### Main Workflows

- **Project Bootstrap**: `python mywi.py db setup`
- **Media Analysis**: `python mywi.py land medianalyse --name=LAND_NAME [--depth=DEPTH] [--minrel=MIN_RELEVANCE]`
- **Land Life-Cycle**: Create, add terms, add URLs, crawl, extract readable, export, clean/delete.
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

### Testing

- `tests/test_cli.py`: CLI smoke tests.
- `tests/test_core.py`, etc.: Unit tests for extraction, parsing, export.

### Extending

- Add export: implement `Export.write_<type>`, update controller.
- Change language: pass `--lang` at land creation.
- Add headers/proxy: edit `settings` or patch session logic.
- Custom tags: use tag hierarchy, export flattens to paths.

---

## Installation

You can install MyWI using Docker (recommended for ease of use) or by setting up a local development environment.

### Using Docker Compose (recommended)

The simplest way to run MyWI is with Docker Compose. It keeps your database and exports in a folder you control, and gives you an interactive container for the CLI. This section is written for beginners.

Prerequisites:
* Docker Desktop (or Docker Engine) with Compose

0) Optional — choose where to store data/exports
- Copy `.env.example` to `.env`:
  ```bash
  cp .env.example .env
  ```
- Keep the default to store everything in `./data` inside the repo:
  ```env
  HOST_DATA_DIR=./data
  ```
- Or set an absolute path outside the repo:
  - macOS/Linux: `HOST_DATA_DIR=/Users/you/mywi_data`
  - Windows: `HOST_DATA_DIR=C:/Users/you/mywi_data`

1) Build and start the container
```bash
docker compose up -d --build
```

2) Initialize the database (first time only)
```bash
docker compose exec mwi python mywi.py db setup
```

3) Run the CLI (examples)
```bash
docker compose exec mwi python mywi.py land create --name="MyResearchTopic" --desc="…" --lang=fr
docker compose exec mwi python mywi.py land addurl --land="MyResearchTopic" --urls="https://example.org"
docker compose exec mwi python mywi.py land crawl --name="MyResearchTopic" --limit=10
docker compose exec mwi python mywi.py land readable --name="MyResearchTopic" --merge=smart_merge
docker compose exec mwi python mywi.py land export --name="MyResearchTopic" --type=pagecsv
```

Where are my files?
- On your machine: `${HOST_DATA_DIR}` (default `./data` in the repo)
- In the container: `/app/data`
- `settings.py` already uses `data` (resolved to `/app/data`), so you don’t need to edit it.

Optional:
- Install Playwright browsers (for dynamic media extraction):
  ```bash
  docker compose exec mwi python install_playwright.py
  ```
- Build with ML extras (FAISS + transformers for embeddings/NLI):
  ```bash
  MYWI_WITH_ML=1 docker compose build
  # Then run as usual: docker compose up -d
  ```

Stop/remove:
```bash
docker compose down        # stop
docker compose down -v     # stop and remove volumes (destructive)
```

Choosing the data location — common cases
- Case A (default, simplest): leave `HOST_DATA_DIR=./data` in `.env`. Nothing else to do.
- Case B (outside the repo): set an absolute path in `.env`, e.g.:
  - macOS/Linux: `HOST_DATA_DIR=/Users/alice/mywi_data`
  - Windows: `HOST_DATA_DIR=C:/Users/Alice/mywi_data`
  Then run `docker compose up -d --build` and use the CLI as shown above.

### Using Docker (manual)

**Prerequisites:**
*   Python 3.10+ (for understanding the project, not strictly for running Docker if image is pre-built)
*   [Docker Desktop](https://www.docker.com/products/docker-desktop)

**Steps:**

1.  **Create a Data Directory:**
    On your host machine, create a directory to store the SQLite database file and other persistent data. This directory will be mounted into the Docker container.
    ```bash
    mkdir ~/mywi_data 
    # Example: creates a directory named 'mywi_data' in your home folder
    ```

2.  **Clone the Project:**
    ```bash
    git clone https://github.com/MyWebIntelligence/MyWebIntelligencePython.git
    cd MyWebIntelligencePython
    ```

3.  **Configure the data location (optional):**
    For beginners, the simplest approach is to mount your host folder at `/app/data`. The app’s default path is `data` → it resolves to `/app/data`. No code changes are required.
    - Case 1 (recommended): mount to `/app/data` (no changes)
      - macOS/Linux :
        ```bash
        mkdir -p ~/mywi_data
        docker run -dit --name mwi \
          -v ~/mywi_data:/app/data \
          mwi:latest
        ```
      - Windows:
        ```powershell
        docker run -dit --name mwi `
          -v C:/Users/you/mywi_data:/app/data `
          mwi:latest
        ```
    - Case 2 (advanced): use a different internal path and tell the app
      ```bash
      docker run -dit --name mwi \
        -e MYWI_DATA_DIR=/data \
        -v /absolute/host/path:/data \
        mwi:latest
      ```

4.  **Build the Docker Image:**
    ```bash
    docker build -t mwi:latest .
    # Using mwi:1.2 as per original, but latest is also common
    # docker build -t mwi:1.2 . 
    ```

5.  **Run the Docker Container:**
    Replace `/path/to/your/host/data` with your actual folder. Recommended mapping is to `/app/data`:
    ```bash
    docker run -dit --name mwi -v /path/to/your/host/data:/app/data mwi:latest
    # macOS/Linux example:
    # docker run -dit --name mwi -v ~/mywi_data:/app/data mwi:latest
    # Windows example:
    # docker run -dit --name mwi -v C:/Users/you/mywi_data:/app/data mwi:latest
    ```
    *   `-d`: Run in detached mode
    *   `-i`: Keep STDIN open even if not attached
    *   `-t`: Allocate a pseudo-TTY
    *   `--name mwi`: Assign a name to the container
    *   `-v /path/to/your/host/data:/app/data`: Mount your host data directory to `/app/data` inside the container (matches app default).

6.  **Access the Container Shell:**
    ```bash
    docker exec -it mwi bash
    ```

7.  **Setup Database (inside the container):**
    If this is the first time, or if the database doesn't exist in your mounted volume:
    ```bash
    # Inside the Docker container
    python mywi.py db setup
    ```
    You are now ready to use MyWI commands as described in the [Usage](#usage) section.

**Note on Dynamic Media Extraction:**
The Docker image includes Playwright and Chromium browser for enhanced media detection. This enables:
- Detection of JavaScript-generated media content
- Extraction of lazy-loaded images
- Support for dynamic content that requires browser rendering

To test the dynamic media extraction functionality:
```bash
# Inside the Docker container
python test_dynamic_media.py
```

### Local Development Setup

**Prerequisites:**
*   Python 3.10+
*   `pip` (Python package installer)
*   `virtualenv` (Python environment isolation tool)
*   `git`

**Steps:**

1.  **Install `virtualenv` (if not already installed):**
    ```bash
    pip install virtualenv
    ```

2.  **Clone the Project:**
    ```bash
    git clone https://github.com/MyWebIntelligence/MyWebIntelligencePython.git
    cd MyWebIntelligencePython
    ```

3.  **Create and Activate Virtual Environment:**

    *   **macOS / Linux:**
        ```bash
        virtualenv venv
        source venv/bin/activate
        ```
    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    Your command prompt should now be prefixed with `(venv)`.

4.  **Configure Data Location:**
    Create a data directory anywhere on your system. Then, edit the `settings.py` file in the project directory and update `data_location` to the absolute path of this directory.
    ```python
    # settings.py
    data_location = "/path/to/your/local/data" 
    # e.g., "C:/Users/YourUser/mywi_data" on Windows
    # or "/Users/youruser/mywi_data" on macOS/Linux
    ```

5.  **Install Dependencies:**
    ```bash
    (venv) pip install -r requirements.txt
    ```

6.  **Install Playwright Browsers (Optional - for Dynamic Media Extraction):**
    ```bash
    (venv) python install_playwright.py
    ```
    This step is optional but recommended if you want to use the dynamic media extraction feature for JavaScript-generated content.

7.  **Setup Database:**
    ```bash
    (venv) python mywi.py db setup
    ```
    This command creates the database file in the `data_location` you specified. Warning: it will destroy any previous data if the database file already exists from a prior setup.

8.  **Test Installation (Optional):**
    ```bash
    (venv) python test_dynamic_media.py
    ```
    This tests both basic URL resolution and dynamic media extraction (if Playwright is installed).

You are now ready to use MyWI commands as described in the [Usage](#usage) section using `(venv) python mywi.py ...`.

## Usage

### General Notes

*   Commands are run using `python mywi.py ...`.
*   If using Docker, first execute `docker exec -it mwi bash` to enter the container. The prompt might be `root@<container_id>:/app#` or similar.
*   If using a local development setup, ensure your virtual environment is activated (e.g., `(venv)` prefix in your prompt).
*   Arguments like `LAND_NAME` or `TERMS` are placeholders; replace them with your actual values.

### Land Management

A "Land" is a central concept in MyWI, representing a specific research area or topic.

---

### Land Consolidation Pipeline

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

#### 1. Create a New Land

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

#### 2. List Created Lands

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

#### 3. Add Terms to a Land

Add keywords or phrases to a land.

```bash
python mywi.py land addterm --land="MyResearchTopic" --terms="keyword1, keyword2, related phrase"
```

| Option   | Type | Required | Default | Description                                 |
|----------|------|----------|---------|---------------------------------------------|
| --land   | str  | Yes      |         | Name of the land to add terms to            |
| --terms  | str  | Yes      |         | Comma-separated list of terms/keywords      |

---

#### 4. Add URLs to a Land

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

#### 5. Delete a Land or Expressions

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


### Data Collection

#### 1. Crawl Land URLs

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

---

#### 2. Fetch Readable Content (Mercury Parser Pipeline)

Extract high-quality, readable content using the **Mercury Parser autonomous pipeline**. This modern system provides intelligent content extraction with configurable merge strategies and automatic media/link enrichment.

**Prerequisites:** Requires `mercury-parser` CLI tool installed:
```bash
sudo npm install -g @postlight/mercury-parser
```

**Command:**
```bash
python mywi.py land readable --name="MyResearchTopic" [--limit=NUMBER] [--depth=NUMBER] [--merge=STRATEGY]
```

| Option   | Type   | Required | Default | Description                                         |
|----------|--------|----------|---------|-----------------------------------------------------|
| --name   | str    | Yes      |         | Name of the land to process                         |
| --limit  | int    | No       |         | Maximum number of pages to process in this run      |
| --depth  | int    | No       |         | Maximum crawl depth to process (e.g., 2 = seeds + 2 levels) |
| --merge  | str    | No       | smart_merge | Merge strategy for content fusion (see below)    |

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
```

**Output:** The pipeline provides detailed statistics including:
- Number of expressions processed
- Success/error rates
- Update counts per field type
- Performance metrics

**Note:** This pipeline replaces the legacy readable functionality, providing better content quality, robustness, and flexible merge strategies for different use cases.

### Domain Management

#### 1. Crawl Domains

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

### Exporting Data

Export data from your lands or tags for analysis in other tools.

#### 1. Export Land Data

Export data from a land in various formats.

#### 2. Media Analysis

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

#### 2. Export Tag Data

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

### Heuristics

#### 1. Update Domains from Heuristic Settings

Update domain information based on predefined or learned heuristics.

```bash
python mywi.py heuristic update
```

_No options for this command._


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

## Embeddings & Pseudolinks (User Guide)

### Purpose
- Build paragraph‑level vectors (embeddings) from pages, then link similar paragraphs across pages (“pseudolinks”).
- Optionally classify each pair with an NLI model (entailment/neutral/contradiction).
- Export paragraph links, plus aggregated links at page and domain levels.

Typical flow
1) Crawl + extract readable text
2) Generate embeddings (paragraph vectors)
3) Compute similarities (cosine or ANN+NLI)
4) Export as CSV (paragraph/page/domain)

### Prerequisites & Install
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

### Language Models (NLI)
- Multilingual (recommended):
  - MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
- Lightweight fallback (English):
  - typeform/distilbert-base-uncased-mnli
- Set in `settings.py:nli_model_name` (both examples are documented there).

### Settings (Key Reference)
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

### Commands & Parameters
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

### Troubleshooting & Caution
- “All `score_raw=0.5` and `score=0`” → neutral fallback; install ML extras or switch to the safe EN model.
- “No `score_raw` column” → run `python mywi.py db migrate` once.
- macOS segfaults (OpenMP/Torch): pip‑only venv; try `OMP_NUM_THREADS=1`, then raise; optional `KMP_DUPLICATE_LIB_OK=TRUE`.
- Slow scoring: lower `nli_batch_size`, raise threads moderately, filter with `--minrel`, cap with `--maxpairs`.
- Too many pairs: raise `threshold`, increase `lshbits`, lower `topk`, or use `--minrel`.

### Best Practices — Performance

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

### Model Choice and Fallbacks

- Default NLI model can be multilingual (DeBERTa‑based) and may require `sentencepiece`.
- Safe alternative (English): `typeform/distilbert-base-uncased-mnli`.
- Configure in `settings.py:nli_model_name`.
- If dependencies are missing, the code can fall back to a neutral predictor (`score=0`, `score_raw=0.5`).

### Progress & Logs

- Recall logs every few hundred paragraphs (candidate pairs count).
- NLI scoring logs progress every `settings.nli_progress_every_pairs` pairs with throughput and ETA.
- Final summary prints total pairs, elapsed time, and pairs/s.

### Similarity Methods

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

### ANN Backend Selection (FAISS)

- Install FAISS (optional): `pip install faiss-cpu`.
- CLI override: `--backend=faiss` to force FAISS recall for `--method=nli`.
- Settings default: `similarity_backend = 'faiss'` to prefer FAISS when no `--backend` is specified.
- Fallback: if FAISS is not installed or import fails, recall uses `bruteforce` automatically.
- Verify: `python mywi.py embedding check` prints `FAISS: available` when detected.

### Scalable Similarity (Large Lands)

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

### NLI Relations (ANN + Cross‑Encoder)

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
MWI_DATA_LOCATION="$PWD/data/test-repaired" venv/bin/python mywi.py land list
```

If everything looks good, adopt the repaired DB (after a manual backup):
```bash
cp data/mwi.db data/mwi.db.bak_$(date +%Y%m%d_%H%M%S)
mv data/mwi_repaired.db data/mwi.db
```

Note: You can temporarily point the app to a different data directory using the `MWI_DATA_LOCATION` environment variable; it overrides `settings.py:data_location` for that session.

## License

This project is licensed under the terms of the LICENSE file. (Assuming a LICENSE file exists in the repository, e.g., MIT, Apache 2.0).
If `LICENSE` is the actual name of the file, you can link to it: [LICENSE](LICENSE).
