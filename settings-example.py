"""
Example settings for My Web Intelligence.

Copy this file to `settings.py` and fill in the blanks or replace values
with environment variables for your environment. All API keys default to
reading the `MWI_*` environment variables to simplify container setups.
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

# Cut Domains

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



# SEO Rank enrichment (land seorank)
seorank_api_base_url = os.getenv(
    "MWI_SEORANK_API_BASE_URL", "https://seo-rank.my-addr.com/api2/sr+fb"
)
seorank_api_key = os.getenv("MWI_SEORANK_API_KEY", "")
seorank_timeout = 15  # seconds
seorank_request_delay = 1.0  # polite sleep between API calls

# SerpAPI enrichment (land urlist)
serpapi_api_key = os.getenv("MWI_SERPAPI_API_KEY", "")
serpapi_base_url = "https://serpapi.com/search"
serpapi_timeout = 15  # seconds


# --- Embedding Settings ---

# Embedding settings (bi-encoder)
# Provider: one of 'fake', 'http', 'openai', 'mistral', 'gemini', 'huggingface', 'ollama'
embed_provider = os.getenv('MWI_EMBED_PROVIDER', 'mistral')

# Common
embed_model_name = os.getenv("MWI_EMBED_MODEL", "mistral-embed")
embed_batch_size = 32
embed_min_paragraph_chars = 150
embed_max_paragraph_chars = 6000
embed_similarity_threshold = 0.75
embed_similarity_method = 'cosine'  # 'cosine' | 'cosine_lsh'
embed_max_retries = 5
embed_backoff_initial = 1.0
embed_backoff_multiplier = 2.0
embed_backoff_max = 30.0
embed_sleep_between_batches = 0.0

# Generic HTTP provider
embed_api_url = os.getenv("MWI_EMBED_API_URL", "")
embed_http_headers = {}

# OpenAI API
embed_openai_base_url = "https://api.openai.com/v1"
embed_openai_api_key = os.getenv("MWI_OPENAI_API_KEY", "")

# Mistral API
embed_mistral_base_url = "https://api.mistral.ai/v1"
embed_mistral_api_key = os.getenv("MWI_MISTRAL_API_KEY", "")

# Google Gemini (Generative Language API)
embed_gemini_base_url = "https://generativelanguage.googleapis.com/v1beta"
embed_gemini_api_key = os.getenv("MWI_GEMINI_API_KEY", "")

# Hugging Face Inference API
embed_hf_base_url = "https://api-inference.huggingface.co/models"
embed_hf_api_key = os.getenv("MWI_HF_API_KEY", "")

# Ollama local API
embed_ollama_base_url = os.getenv("MWI_OLLAMA_BASE_URL", "http://localhost:11434")

# --- Semantic Search & NLI Settings ---
embedding_model_name = embed_model_name

# Cross-Encoder (NLI) model
nli_model_name = os.getenv(
    "MWI_NLI_MODEL_NAME",
    "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
)
nli_batch_size = 64

# Backend preference: 'auto', 'transformers', 'crossencoder', 'fallback'
nli_backend_preference = os.getenv("MWI_NLI_BACKEND", 'fallback')

# CPU threading
nli_torch_num_threads = int(os.getenv("MWI_NLI_TORCH_THREADS", "1"))

# Fallback NLI model
nli_fallback_model_name = os.getenv(
    "MWI_NLI_FALLBACK_MODEL_NAME",
    "typeform/distilbert-base-uncased-mnli",
)

# Max tokens fed to the NLI tokenizer/model
nli_max_tokens = 512

# Progress reporting during NLI scoring
nli_progress_every_pairs = 1000
nli_show_throughput = True

# ANN Backend Configuration (recall)
similarity_backend = os.getenv('MWI_SIMILARITY_BACKEND', 'faiss')
similarity_top_k = int(os.getenv('MWI_SIMILARITY_TOP_K', '50'))

# NLI Classification Thresholds
nli_entailment_threshold = float(os.getenv('MWI_NLI_ENTAILMENT_THRESHOLD', '0.8'))
nli_contradiction_threshold = float(os.getenv('MWI_NLI_CONTRADICTION_THRESHOLD', '0.8'))
