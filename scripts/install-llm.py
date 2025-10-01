#!/usr/bin/env python3
"""
MyWebIntelligence - LLM and Embeddings Configuration

Interactive configuration wizard for embeddings and NLI (Natural Language Inference).
Updates settings.py with embedding provider, NLI models, and semantic search configuration.

Features:
- Embedding providers (OpenAI, Mistral, Gemini, HuggingFace, Ollama, HTTP)
- NLI models for semantic classification
- FAISS/bruteforce similarity backends
- Semantic search parameters

Usage:
    python scripts/install-llm.py
    python scripts/install-llm.py --help
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import install_utils
sys.path.insert(0, str(Path(__file__).parent))

from install_utils import (
    print_banner,
    print_section,
    print_help,
    success,
    error,
    info,
    warning,
    ask_string,
    ask_int,
    ask_float,
    ask_bool,
    ask_secret,
    ask_choice,
    backup_file,
    read_settings,
    write_settings,
    confirm_config,
    truncate_secret,
)


def main():
    """Main LLM configuration wizard."""
    parser = argparse.ArgumentParser(
        description="MyWebIntelligence LLM and Embeddings Configuration"
    )
    parser.add_argument(
        '--output',
        default='settings.py',
        help='Output file path (default: settings.py)'
    )
    args = parser.parse_args()

    print_banner("MyWebIntelligence - LLM Configuration", width=65)
    print_help("This script will configure embeddings and semantic search.")
    print_help("Press Ctrl+C at any time to cancel.")
    print()

    # Check ML dependencies
    check_ml_dependencies()

    # Check if settings.py exists
    settings_path = Path(args.output)
    if not settings_path.exists():
        print(warning(f"{args.output} not found"))
        print_help("Run 'python scripts/install-basic.py' first to create basic configuration")
        print()
        create_new = ask_bool("Create new settings file anyway?", default=False)
        if not create_new:
            return 1
        config = {}
    else:
        config = read_settings()
        print(info(f"Loaded existing configuration from {args.output}"))

    try:
        # Configure components
        configure_embeddings(config)
        configure_nli(config)
        configure_similarity(config)

        # Collect all LLM-related keys
        llm_keys = [
            k for k in config.keys()
            if k.startswith('embed_') or k.startswith('nli_') or k.startswith('similarity_') or k.startswith('embedding_')
        ]

        if not confirm_config(config, llm_keys):
            print()
            print(warning("Configuration cancelled"))
            return 1

        # Backup existing settings
        backup_path = backup_file(args.output)
        if backup_path:
            print()
            print(info(f"Backed up existing settings to: {backup_path}"))

        # Write configuration
        write_settings(config, args.output)
        print(success(f"Configuration saved to {args.output}"))

        # Print next steps
        print_next_steps(config)

        return 0

    except KeyboardInterrupt:
        print()
        print()
        print(warning("Configuration cancelled by user"))
        return 1
    except Exception as e:
        print()
        print(error(f"Configuration failed: {e}"))
        import traceback
        traceback.print_exc()
        return 1


def check_ml_dependencies():
    """Check if ML dependencies are installed."""
    print_section("Checking ML Dependencies")

    missing = []

    try:
        import torch
        print(success("PyTorch installed"))
    except ImportError:
        missing.append("torch")
        print(error("PyTorch not installed"))

    try:
        import transformers
        print(success("Transformers installed"))
    except ImportError:
        missing.append("transformers")
        print(error("Transformers not installed"))

    try:
        import sentence_transformers
        print(success("Sentence-Transformers installed"))
    except ImportError:
        missing.append("sentence-transformers")
        print(error("Sentence-Transformers not installed"))

    try:
        import faiss
        print(success("FAISS installed"))
    except ImportError:
        print(warning("FAISS not installed (optional, but recommended)"))

    if missing:
        print()
        print(warning("Missing ML dependencies detected"))
        print_help("Install with: pip install -r requirements-ml.txt")
        print()
        if not ask_bool("Continue anyway?", default=True):
            sys.exit(1)


def configure_embeddings(config: dict):
    """Configure embedding provider and settings."""
    print_section("Embedding Configuration", step=(1, 3))
    print_help("Choose a provider for generating paragraph embeddings")
    print()

    provider_choices = [
        ("fake", "Fake (testing/development - deterministic random vectors)"),
        ("openai", "OpenAI (text-embedding-3-small, text-embedding-ada-002)"),
        ("mistral", "Mistral AI (mistral-embed) [RECOMMENDED]"),
        ("gemini", "Google Gemini (embedding-001)"),
        ("huggingface", "HuggingFace Inference API"),
        ("ollama", "Ollama (local, nomic-embed-text)"),
        ("http", "Custom HTTP endpoint"),
    ]

    provider = ask_choice(
        "Select embedding provider:",
        provider_choices,
        default="mistral"
    )

    config['embed_provider'] = f'os.getenv("MWI_EMBED_PROVIDER", "{provider}")'

    # Provider-specific configuration
    print()

    if provider == "fake":
        print(info("Using fake embeddings (for testing only)"))
        config['embed_model_name'] = 'os.getenv("MWI_EMBED_MODEL", "fake-embedding-model")'

    elif provider == "openai":
        configure_openai_embeddings(config)

    elif provider == "mistral":
        configure_mistral_embeddings(config)

    elif provider == "gemini":
        configure_gemini_embeddings(config)

    elif provider == "huggingface":
        configure_huggingface_embeddings(config)

    elif provider == "ollama":
        configure_ollama_embeddings(config)

    elif provider == "http":
        configure_http_embeddings(config)

    # Common embedding settings
    print()
    print_section("Embedding Parameters")

    config['embed_batch_size'] = ask_int(
        "Batch size (number of paragraphs per API call)",
        default=32,
        min_val=1,
        max_val=128
    )

    config['embed_min_paragraph_chars'] = ask_int(
        "Minimum paragraph length (characters)",
        default=150,
        min_val=50,
        max_val=500
    )

    config['embed_max_paragraph_chars'] = ask_int(
        "Maximum paragraph length (characters)",
        default=6000,
        min_val=1000,
        max_val=20000
    )

    similarity_method_choices = [
        ("cosine", "Cosine similarity (exact, O(n²))"),
        ("cosine_lsh", "Cosine LSH (approximate, scalable)"),
    ]

    config['embed_similarity_method'] = ask_choice(
        "Similarity method:",
        similarity_method_choices,
        default="cosine"
    )

    config['embed_similarity_threshold'] = ask_float(
        "Similarity threshold (0-1)",
        default=0.75,
        min_val=0.0,
        max_val=1.0
    )

    # Retry settings
    config['embed_max_retries'] = 5
    config['embed_backoff_initial'] = 1.0
    config['embed_backoff_multiplier'] = 2.0
    config['embed_backoff_max'] = 30.0
    config['embed_sleep_between_batches'] = 0.0

    print(success(f"Embedding provider configured: {provider}"))


def configure_openai_embeddings(config: dict):
    """Configure OpenAI embedding settings."""
    print_help("OpenAI embedding models: text-embedding-3-small, text-embedding-ada-002")
    print_help("Get your API key at: https://platform.openai.com/api-keys")
    print()

    api_key = ask_secret("OpenAI API Key", required=True)

    model = ask_string(
        "Model name",
        default="text-embedding-3-small",
        examples=["text-embedding-3-small", "text-embedding-ada-002"]
    )

    config['embed_openai_api_key'] = f'os.getenv("MWI_OPENAI_API_KEY", "{api_key}")'
    config['embed_openai_base_url'] = "https://api.openai.com/v1"
    config['embed_model_name'] = f'os.getenv("MWI_EMBED_MODEL", "{model}")'

    print(success(f"OpenAI configured (key: {truncate_secret(api_key)}, model: {model})"))


def configure_mistral_embeddings(config: dict):
    """Configure Mistral embedding settings."""
    print_help("Mistral embedding model: mistral-embed")
    print_help("Get your API key at: https://console.mistral.ai/")
    print()

    api_key = ask_secret("Mistral API Key", required=True)

    config['embed_mistral_api_key'] = f'os.getenv("MWI_MISTRAL_API_KEY", "{api_key}")'
    config['embed_mistral_base_url'] = "https://api.mistral.ai/v1"
    config['embed_model_name'] = 'os.getenv("MWI_EMBED_MODEL", "mistral-embed")'

    print(success(f"Mistral configured (key: {truncate_secret(api_key)})"))


def configure_gemini_embeddings(config: dict):
    """Configure Google Gemini embedding settings."""
    print_help("Google Gemini embedding model: embedding-001")
    print_help("Get your API key at: https://makersuite.google.com/app/apikey")
    print()

    api_key = ask_secret("Gemini API Key", required=True)

    config['embed_gemini_api_key'] = f'os.getenv("MWI_GEMINI_API_KEY", "{api_key}")'
    config['embed_gemini_base_url'] = "https://generativelanguage.googleapis.com/v1beta"
    config['embed_model_name'] = 'os.getenv("MWI_EMBED_MODEL", "embedding-001")'

    print(success(f"Gemini configured (key: {truncate_secret(api_key)})"))


def configure_huggingface_embeddings(config: dict):
    """Configure HuggingFace embedding settings."""
    print_help("HuggingFace Inference API")
    print_help("Get your API key at: https://huggingface.co/settings/tokens")
    print()

    api_key = ask_secret("HuggingFace API Key", required=True)

    model = ask_string(
        "Model name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        examples=[
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        ]
    )

    config['embed_hf_api_key'] = f'os.getenv("MWI_HF_API_KEY", "{api_key}")'
    config['embed_hf_base_url'] = "https://api-inference.huggingface.co/models"
    config['embed_model_name'] = f'os.getenv("MWI_EMBED_MODEL", "{model}")'

    print(success(f"HuggingFace configured (key: {truncate_secret(api_key)}, model: {model})"))


def configure_ollama_embeddings(config: dict):
    """Configure Ollama (local) embedding settings."""
    print_help("Ollama runs models locally on your machine")
    print_help("Install from: https://ollama.ai/")
    print()

    base_url = ask_string(
        "Ollama API URL",
        default="http://localhost:11434"
    )

    model = ask_string(
        "Model name",
        default="nomic-embed-text",
        examples=["nomic-embed-text", "all-minilm"]
    )

    config['embed_ollama_base_url'] = f'os.getenv("MWI_OLLAMA_BASE_URL", "{base_url}")'
    config['embed_model_name'] = f'os.getenv("MWI_EMBED_MODEL", "{model}")'

    print(success(f"Ollama configured (URL: {base_url}, model: {model})"))


def configure_http_embeddings(config: dict):
    """Configure custom HTTP endpoint for embeddings."""
    print_help("Custom HTTP endpoint for embeddings")
    print_help("Endpoint should accept POST with JSON: {\"model\": \"...\", \"input\": [\"text1\", ...]}")
    print()

    api_url = ask_string(
        "Embedding API URL",
        required=True,
        examples=["https://your-api.com/embeddings"]
    )

    api_key = ask_secret("API Key (optional, leave empty if not needed)", required=False)

    model = ask_string(
        "Model name",
        default="custom-model"
    )

    config['embed_api_url'] = f'os.getenv("MWI_EMBED_API_URL", "{api_url}")'
    config['embed_model_name'] = f'os.getenv("MWI_EMBED_MODEL", "{model}")'

    if api_key:
        config['embed_http_headers'] = {"Authorization": f"Bearer {api_key}"}
        print(success(f"HTTP endpoint configured (URL: {api_url}, key: {truncate_secret(api_key)})"))
    else:
        config['embed_http_headers'] = {}
        print(success(f"HTTP endpoint configured (URL: {api_url}, no auth)"))


def configure_nli(config: dict):
    """Configure NLI (Natural Language Inference) settings."""
    print_section("NLI Configuration", step=(2, 3))
    print_help("NLI models classify semantic relations between text pairs")
    print_help("Used for entailment/neutral/contradiction detection")
    print()

    nli_choices = [
        ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "mDeBERTa XNLI (multilingual, recommended)"),
        ("typeform/distilbert-base-uncased-mnli", "DistilBERT MNLI (English, fast)"),
        ("custom", "Custom model (enter HuggingFace model name)"),
    ]

    nli_model = ask_choice(
        "Select NLI model:",
        nli_choices,
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )

    if nli_model == "custom":
        print()
        nli_model = ask_string(
            "Enter HuggingFace model name",
            required=True,
            examples=["cross-encoder/nli-deberta-v3-base"]
        )

    config['nli_model_name'] = f'os.getenv("MWI_NLI_MODEL_NAME", "{nli_model}")'
    config['embedding_model_name'] = config.get('embed_model_name', 'os.getenv("MWI_EMBED_MODEL", "mistral-embed")')

    # Fallback model
    config['nli_fallback_model_name'] = 'os.getenv("MWI_NLI_FALLBACK_MODEL_NAME", "typeform/distilbert-base-uncased-mnli")'

    # NLI backend preference
    backend_choices = [
        ("auto", "Auto-detect best backend"),
        ("transformers", "Transformers library"),
        ("crossencoder", "CrossEncoder (sentence-transformers)"),
        ("fallback", "Safe fallback (always works)"),
    ]

    backend = ask_choice(
        "NLI backend preference:",
        backend_choices,
        default="auto"
    )

    config['nli_backend_preference'] = f'os.getenv("MWI_NLI_BACKEND", "{backend}")'

    print()
    config['nli_batch_size'] = ask_int(
        "NLI batch size (pairs per batch)",
        default=64,
        min_val=1,
        max_val=256
    )

    config['nli_max_tokens'] = ask_int(
        "Max tokens per text",
        default=512,
        min_val=128,
        max_val=2048
    )

    config['nli_torch_num_threads'] = f'int(os.getenv("MWI_NLI_TORCH_THREADS", "1"))'

    # Progress reporting
    config['nli_progress_every_pairs'] = 1000
    config['nli_show_throughput'] = True

    print(success(f"NLI configured: {nli_model}"))


def configure_similarity(config: dict):
    """Configure similarity backend and search parameters."""
    print_section("Similarity Backend Configuration", step=(3, 3))
    print_help("Choose backend for approximate nearest neighbor (ANN) search")
    print()

    backend_choices = [
        ("faiss", "FAISS (fast, requires faiss-cpu)"),
        ("bruteforce", "Brute-force (slower, no extra deps)"),
    ]

    backend = ask_choice(
        "Similarity backend:",
        backend_choices,
        default="faiss"
    )

    config['similarity_backend'] = f'os.getenv("MWI_SIMILARITY_BACKEND", "{backend}")'

    print()
    config['similarity_top_k'] = f'int(os.getenv("MWI_SIMILARITY_TOP_K", "50"))'

    top_k = ask_int(
        "Top K neighbors per paragraph",
        default=50,
        min_val=1,
        max_val=200
    )
    config['similarity_top_k'] = f'int(os.getenv("MWI_SIMILARITY_TOP_K", "{top_k}"))'

    print()
    print_help("NLI classification thresholds (0-1)")

    entailment_threshold = ask_float(
        "Entailment threshold",
        default=0.8,
        min_val=0.0,
        max_val=1.0
    )

    contradiction_threshold = ask_float(
        "Contradiction threshold",
        default=0.8,
        min_val=0.0,
        max_val=1.0
    )

    config['nli_entailment_threshold'] = f'float(os.getenv("MWI_NLI_ENTAILMENT_THRESHOLD", "{entailment_threshold}"))'
    config['nli_contradiction_threshold'] = f'float(os.getenv("MWI_NLI_CONTRADICTION_THRESHOLD", "{contradiction_threshold}"))'

    print(success(f"Similarity backend configured: {backend} (top-{top_k})"))


def print_next_steps(config: dict):
    """Print next steps after configuration."""
    print()
    print_section("Next Steps")
    print()

    # Check if FAISS was selected
    backend = config.get('similarity_backend', '')
    if 'faiss' in backend.lower():
        print("Verify FAISS installation:")
        print(info("   python -c \"import faiss; print('FAISS version:', faiss.__version__)\""))
        print()

    # NLI model download
    nli_model = config.get('nli_model_name', '')
    if nli_model:
        # Extract model name from the config value
        import re
        match = re.search(r'"([^"]+)"', nli_model)
        if match:
            model_name = match.group(1)
            print("Pre-download NLI model (optional):")
            print(info(f'   python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \\'))
            print(info(f'     AutoTokenizer.from_pretrained(\'{model_name}\'); \\'))
            print(info(f'     AutoModelForSequenceClassification.from_pretrained(\'{model_name}\')"'))
            print()

    print("Verify installation:")
    print(info("   python mywi.py embedding check"))
    print()

    print("Generate embeddings for a land:")
    print(info('   python mywi.py embedding generate --name="MyLand" --limit=100'))
    print()

    print("Compute similarities:")
    print(info('   python mywi.py embedding similarity --name="MyLand" --method=cosine --threshold=0.85'))
    print()

    print("Export semantic links (pseudolinks):")
    print(info('   python mywi.py land export --name="MyLand" --type=pseudolinks'))
    print()


if __name__ == '__main__':
    sys.exit(main())
