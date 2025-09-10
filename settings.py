"""
Settings — pragmatic defaults with short, actionable comments.

Tips
- Keep values simple here; override per-run from the CLI when needed.
- Strings are empty by default except `data_location` (defaults to "data").
- Timeouts are in seconds. Paths can be absolute.
- For CPU tuning (NLI/FAISS), see the "Semantic Search & NLI" block below.
"""

import os

# Paths & storage
# Allow override via env var for containerized runs
data_location = os.getenv("MYWI_DATA_DIR", "data")

archive = False

# Enable dynamic media extraction using headless browser (requires Playwright)
dynamic_media_extraction = True

default_timeout = 10  # Network HTTP timeout (crawl/fetch)

parallel_connections = 10  # Async HTTP concurrency for crawling

user_agent = ""  # Optionally set a custom UA

heuristics = {
    "facebook.com": r"([a-z0-9\-_]+\.facebook\.com/(?!(?:permalink.php)|(?:notes))[a-zA-Z0-9\.\-_]+)/?\??",
    "twitter.com": r"([a-z0-9\-_]*\.?twitter\.com/(?!(?:hashtag)|(?:search)|(?:home)|(?:share))[a-zA-Z0-9\.\-_]+)",
    "linkedin.com": r"([a-z0-9\-_]+\.linkedin\.com/[a-zA-Z0-9\.\-_]+)/?\??",
    "slideshare.net": r"([a-z0-9\-_]+\.slideshare\.com/[a-zA-Z0-9\.\-_]+)/?\??",
    "instagram.com": r"([a-z0-9\-_]+\.instagram\.com/[a-zA-Z0-9\.\-_]+)/?\??",
    "youtube.com": r"([a-z0-9\-_]+\.youtube\.com/(?!watch)[a-zA-Z0-9\.\-_]+)/?\??",
    "vimeo.com": r"([a-z0-9\-_]+\.vimeo\.com/[a-zA-Z0-9\.\-_]+)/?\??",
    "dailymotion.com": r"([a-z0-9\-_]+\.dailymotion\.com/(?!video)[a-zA-Z0-9\.\-_]+)/?\??",
    "pinterest.com": r"([a-z0-9\-_]+\.pinterest\.com/(?!pin)[a-zA-Z0-9\.\-_]+)/?\??",
    "pinterest.fr": r"([a-z0-9\-_]+\.pinterest\.fr/[a-zA-Z0-9\.\-_]+)/?\??",
}

# Media Analysis Settings
media_analysis = True
media_min_width = 200
media_min_height = 200
media_max_file_size = 10 * 1024 * 1024  # 10MB
media_download_timeout = 30
media_max_retries = 2
media_analyze_content = False
media_extract_colors = True
media_extract_exif = True
media_n_dominant_colors = 5

# OpenRouter relevance gate (disabled by default)
openrouter_enabled = False
openrouter_api_key = ""
openrouter_model = ""
 # Exemples de modèles compatibles OpenRouter (mini/éco)
 # Renseignez `openrouter_model` avec l'un de ces slugs si vous activez la passerelle
openrouter_model_examples = [
    # OpenAI
    "openai/gpt-4o-mini",
    # Anthropic
    "anthropic/claude-3-haiku",
    # Google
    "google/gemini-1.5-flash",
    # Meta (Llama 3.x Instruct – 8B)
    "meta-llama/llama-3.1-8b-instruct",
    # Mistral
    "mistralai/mistral-small-latest",
    # Qwen (Alibaba)
    "qwen/qwen2.5-7b-instruct",
    # Cohere
    "cohere/command-r-mini",
]
openrouter_timeout = 15  # seconds
# Bounds to control costs/latency
openrouter_readable_max_chars = 12000
openrouter_max_calls_per_run = 500

# Embedding settings (bi-encoder)
# Provider: one of 'fake', 'http', 'openai', 'mistral', 'gemini', 'huggingface', 'ollama'
embed_provider = 'mistral'

# Common
embed_model_name = "mistral-embed"  # stored with vectors for provenance
embed_batch_size = 32               # provider batch size
embed_min_paragraph_chars = 150     # skip tiny fragments
embed_max_paragraph_chars = 6000    # safety cap per paragraph
# Similarity between embeddings (bi-encoder phase)
embed_similarity_threshold = 0.75
embed_similarity_method = 'cosine'  # 'cosine' | 'cosine_lsh'
# Robust HTTP policy for embedding calls
embed_max_retries = 5
embed_backoff_initial = 1.0         # seconds
embed_backoff_multiplier = 2.0      # exponential backoff
embed_backoff_max = 30.0            # max sleep between retries
embed_sleep_between_batches = 0.0   # optional throttle

# Generic HTTP provider
# POST JSON: {"model": embed_model_name, "input": ["text1", ...]}
# Expect JSON: {"data": [{"embedding": [..]}, ...]}
embed_api_url = ""
# Optional additional headers for generic HTTP provider (dict as JSON string or mapping)
embed_http_headers = {}

# OpenAI API
embed_openai_base_url = "https://api.openai.com/v1"
embed_openai_api_key = ""

# Mistral API
embed_mistral_base_url = "https://api.mistral.ai/v1"
embed_mistral_api_key = ""

# Google Gemini (Generative Language API)
# Uses v1beta models endpoints; choose model like 'models/text-embedding-004'
embed_gemini_base_url = "https://generativelanguage.googleapis.com/v1beta"
embed_gemini_api_key = ""

# Hugging Face Inference API (Serverless or Endpoint)
# Base URL for serverless: https://api-inference.huggingface.co/models
# For dedicated endpoints, set this to your endpoint root and keep model name empty or include in base.
embed_hf_base_url = "https://api-inference.huggingface.co/models"
embed_hf_api_key = ""

# Ollama local API
embed_ollama_base_url = "http://localhost:11434"

# --- Semantic Search & NLI Settings ---
# Recall uses embeddings (bi-encoder). Precision uses Cross‑Encoder NLI.
# DB note: paragraph_similarity has columns: source_paragraph, target_paragraph,
# score ∈ {-1,0,1}, score_raw ∈ [0..1], method ∈ {'nli','cosine','cosine_lsh'}.
# Run `python mywi.py db migrate` to add score_raw if your DB is old.
#
# For large lands, prefer FAISS recall + moderate top_k + maxpairs caps.
# For CPU speed, tune nli_torch_num_threads and OMP_NUM_THREADS env var.
embedding_model_name = embed_model_name

## Cross-Encoder (NLI) model
# Default multilingual model (DeBERTa):
#   nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# Safe lightweight EN alternative:
#   nli_model_name = "typeform/distilbert-base-uncased-mnli"
nli_model_name = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
nli_batch_size = 64  # increase if RAM allows for more throughput

## Backend preference
# 'auto' tries sentence-transformers CrossEncoder first, then transformers.
# 'transformers' loads the model via HF transformers (safer on macOS/CPU).
# 'crossencoder' forces sentence-transformers CrossEncoder.
# 'fallback' produces all-neutral scores when models are unavailable.
nli_backend_preference = 'fallback'

## CPU threading
# Torch intra-op threads for NLI inference. Example to use 7 of 8 cores:
#   nli_torch_num_threads = 7  (and export OMP_NUM_THREADS=7 when running)
nli_torch_num_threads = 1

# Fallback NLI model if the primary requires sentencepiece and it's missing.
# Keep this small/safe (WordPiece/BERT) to avoid native deps issues.
nli_fallback_model_name = "typeform/distilbert-base-uncased-mnli"

# Max tokens fed to the NLI tokenizer/model (pairs are truncated to fit)
nli_max_tokens = 512

# Progress reporting during NLI scoring
nli_progress_every_pairs = 1000  # print progress every N scored pairs
nli_show_throughput = True       # include pairs/s + ETA in logs

# ANN Backend Configuration (recall)
# Options: 'bruteforce', 'faiss'
similarity_backend = 'faiss'
similarity_top_k = 50

# NLI Classification Thresholds (utilisées en fallback/filtrage)
nli_entailment_threshold = 0.8
nli_contradiction_threshold = 0.8
