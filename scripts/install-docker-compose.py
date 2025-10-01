#!/usr/bin/env python3
"""
MyWebIntelligence - Docker Compose Setup

Interactive wizard to configure Docker Compose installation.
Generates .env file with all necessary variables.

Features:
- Choose installation level (basic, api, llm)
- Configure data directory location
- Set API keys and credentials
- Configure ML/LLM settings
- Build flags (WITH_ML, WITH_PLAYWRIGHT_BROWSERS)

Usage:
    python scripts/install-docker-compose.py
    python scripts/install-docker-compose.py --level=llm
    python scripts/install-docker-compose.py --help
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
    ask_bool,
    ask_secret,
    ask_choice,
    ask_path,
    backup_file,
    truncate_secret,
)


def main():
    """Main Docker Compose setup wizard."""
    parser = argparse.ArgumentParser(
        description="MyWebIntelligence Docker Compose Setup"
    )
    parser.add_argument(
        '--level',
        choices=['basic', 'api', 'llm'],
        help='Installation level (basic, api, llm)'
    )
    parser.add_argument(
        '--output',
        default='.env',
        help='Output file path (default: .env)'
    )
    args = parser.parse_args()

    print_banner("MyWebIntelligence - Docker Compose Setup", width=65)
    print_help("This script will configure Docker Compose installation.")
    print_help("It generates a .env file with all necessary variables.")
    print()

    try:
        # Step 1: Choose installation level
        if args.level:
            level = args.level
            print(info(f"Installation level: {level}"))
        else:
            level = choose_installation_level()

        # Step 2: Configure based on level
        env_config = {}

        configure_basic_docker(env_config)

        if level in ['api', 'llm']:
            configure_api_docker(env_config)

        if level == 'llm':
            configure_llm_docker(env_config)

        # Step 3: Confirm and save
        print_section("Configuration Summary")
        display_env_summary(env_config)
        print()

        if not ask_bool("Save this configuration?", default=True):
            print()
            print(warning("Configuration cancelled"))
            return 1

        # Backup existing .env
        backup_path = backup_file(args.output)
        if backup_path:
            print()
            print(info(f"Backed up existing .env to: {backup_path}"))

        # Write .env file
        write_env_file(env_config, args.output)
        print(success(f"Configuration saved to {args.output}"))

        # Print next steps
        print_next_steps(level)

        return 0

    except KeyboardInterrupt:
        print()
        print()
        print(warning("Setup cancelled by user"))
        return 1
    except Exception as e:
        print()
        print(error(f"Setup failed: {e}"))
        import traceback
        traceback.print_exc()
        return 1


def choose_installation_level() -> str:
    """
    Ask user to choose installation level.

    Returns:
        'basic', 'api', or 'llm'
    """
    print_section("Choose Installation Level")
    print()

    levels = [
        ("basic", "Basic - Core functionality only"),
        ("api", "API - Basic + external APIs (SerpAPI, SEO Rank, OpenRouter)"),
        ("llm", "LLM - Complete with embeddings and AI features"),
    ]

    return ask_choice(
        "Select installation level:",
        levels,
        default="basic"
    )


def configure_basic_docker(config: dict):
    """Configure basic Docker Compose settings."""
    print_section("Basic Docker Compose Configuration", step=(1, 4))
    print_help("Essential settings for running MyWI in Docker")
    print()

    # Build toggles
    config['MYWI_WITH_ML'] = '0'  # Will be set to 1 for LLM level
    config['MYWI_WITH_PLAYWRIGHT_BROWSERS'] = '0'  # Optional

    # Timezone
    config['TZ'] = ask_string(
        "Timezone",
        default="UTC",
        examples=["UTC", "Europe/Paris", "America/New_York", "Asia/Tokyo"]
    )

    # Data directory on host
    print()
    print_help("Where should data be stored on your computer?")
    print_help("Default: ./data (inside repository)")
    print_help("Alternative: absolute path outside repository")
    print()

    use_default_data_dir = ask_bool(
        "Use default data directory (./data)?",
        default=True
    )

    if use_default_data_dir:
        config['HOST_DATA_DIR'] = './data'
    else:
        data_dir = ask_path(
            "Host data directory",
            default=str(Path.home() / "mywi_data"),
            must_exist=False,
            create_if_missing=False
        )
        config['HOST_DATA_DIR'] = data_dir

    # Internal data dir (container-side, should not be changed)
    config['MYWI_DATA_DIR'] = '/app/data'

    print(success(f"Data will be stored at: {config['HOST_DATA_DIR']}"))
    print(info("Inside container: /app/data"))

    # Playwright browsers
    print()
    install_playwright = ask_bool(
        "Pre-install Playwright browsers in Docker image?",
        default=False
    )

    if install_playwright:
        config['MYWI_WITH_PLAYWRIGHT_BROWSERS'] = '1'
        print(info("Playwright browsers will be installed during build"))
    else:
        print(info("Playwright browsers not installed (can install later)"))


def configure_api_docker(config: dict):
    """Configure API settings for Docker Compose."""
    print_section("API Configuration", step=(2, 4))
    print_help("Configure external API integrations")
    print()

    # SerpAPI
    enable_serpapi = ask_bool("Enable SerpAPI?", default=False)
    if enable_serpapi:
        print()
        print_help("Get your API key at: https://serpapi.com/")
        api_key = ask_secret("SerpAPI Key", required=True)
        config['MWI_SERPAPI_API_KEY'] = api_key
        print(success(f"SerpAPI configured (key: {truncate_secret(api_key)})"))
    else:
        config['MWI_SERPAPI_API_KEY'] = ''

    # SEO Rank
    print()
    enable_seorank = ask_bool("Enable SEO Rank?", default=False)
    if enable_seorank:
        print()
        api_base_url = ask_string(
            "SEO Rank API Base URL",
            default="https://seo-rank.my-addr.com/api2/sr+fb"
        )
        api_key = ask_secret("SEO Rank API Key", required=True)
        config['MWI_SEORANK_API_BASE_URL'] = api_base_url
        config['MWI_SEORANK_API_KEY'] = api_key
        print(success(f"SEO Rank configured (key: {truncate_secret(api_key)})"))
    else:
        config['MWI_SEORANK_API_BASE_URL'] = ''
        config['MWI_SEORANK_API_KEY'] = ''

    # OpenRouter
    print()
    enable_openrouter = ask_bool("Enable OpenRouter LLM gate?", default=False)
    if enable_openrouter:
        print()
        print_help("Get your API key at: https://openrouter.ai/")
        api_key = ask_secret("OpenRouter API Key", required=True)

        # Model selection
        print()
        model_choices = [
            ("deepseek/deepseek-chat-v3.1", "DeepSeek v3.1 (economical)"),
            ("openai/gpt-4o-mini", "GPT-4o Mini (fast)"),
            ("anthropic/claude-3-haiku", "Claude 3 Haiku (quality)"),
            ("google/gemini-1.5-flash", "Gemini 1.5 Flash (balanced)"),
            ("custom", "Custom model"),
        ]

        model = ask_choice(
            "Select OpenRouter model:",
            model_choices,
            default="deepseek/deepseek-chat-v3.1"
        )

        if model == "custom":
            model = ask_string("Enter custom model slug", required=True)

        config['MWI_OPENROUTER_ENABLED'] = 'true'
        config['MWI_OPENROUTER_API_KEY'] = api_key
        config['MWI_OPENROUTER_MODEL'] = model
        config['MWI_OPENROUTER_TIMEOUT'] = '15'
        config['MWI_OPENROUTER_MIN_CHARS'] = '140'
        config['MWI_OPENROUTER_MAX_CHARS'] = '12000'
        config['MWI_OPENROUTER_MAX_CALLS'] = '500'

        print(success(f"OpenRouter configured (model: {model})"))
    else:
        config['MWI_OPENROUTER_ENABLED'] = 'false'
        config['MWI_OPENROUTER_API_KEY'] = ''
        config['MWI_OPENROUTER_MODEL'] = 'deepseek/deepseek-chat-v3.1'
        config['MWI_OPENROUTER_TIMEOUT'] = '15'
        config['MWI_OPENROUTER_MIN_CHARS'] = '140'
        config['MWI_OPENROUTER_MAX_CHARS'] = '12000'
        config['MWI_OPENROUTER_MAX_CALLS'] = '500'


def configure_llm_docker(config: dict):
    """Configure LLM/embeddings settings for Docker Compose."""
    print_section("LLM Configuration", step=(3, 4))
    print_help("Configure embeddings and semantic search")
    print()

    # Enable ML dependencies in build
    config['MYWI_WITH_ML'] = '1'
    print(info("ML dependencies will be installed during Docker build"))

    # Embedding provider
    print()
    provider_choices = [
        ("mistral", "Mistral AI (mistral-embed) [RECOMMENDED]"),
        ("openai", "OpenAI (text-embedding-3-small)"),
        ("gemini", "Google Gemini (embedding-001)"),
        ("huggingface", "HuggingFace Inference API"),
        ("ollama", "Ollama (local)"),
        ("fake", "Fake (testing only)"),
    ]

    provider = ask_choice(
        "Select embedding provider:",
        provider_choices,
        default="mistral"
    )

    config['MWI_EMBED_PROVIDER'] = provider

    # Provider-specific configuration
    print()

    if provider == "mistral":
        print_help("Get your API key at: https://console.mistral.ai/")
        api_key = ask_secret("Mistral API Key", required=True)
        config['MWI_MISTRAL_API_KEY'] = api_key
        config['MWI_EMBED_MODEL'] = 'mistral-embed'
        print(success(f"Mistral configured (key: {truncate_secret(api_key)})"))

    elif provider == "openai":
        print_help("Get your API key at: https://platform.openai.com/api-keys")
        api_key = ask_secret("OpenAI API Key", required=True)
        model = ask_string("Model", default="text-embedding-3-small")
        config['MWI_OPENAI_API_KEY'] = api_key
        config['MWI_EMBED_MODEL'] = model
        print(success(f"OpenAI configured (model: {model})"))

    elif provider == "gemini":
        print_help("Get your API key at: https://makersuite.google.com/app/apikey")
        api_key = ask_secret("Gemini API Key", required=True)
        config['MWI_GEMINI_API_KEY'] = api_key
        config['MWI_EMBED_MODEL'] = 'embedding-001'
        print(success(f"Gemini configured (key: {truncate_secret(api_key)})"))

    elif provider == "huggingface":
        print_help("Get your API key at: https://huggingface.co/settings/tokens")
        api_key = ask_secret("HuggingFace API Key", required=True)
        model = ask_string(
            "Model",
            default="sentence-transformers/all-MiniLM-L6-v2"
        )
        config['MWI_HF_API_KEY'] = api_key
        config['MWI_EMBED_MODEL'] = model
        print(success(f"HuggingFace configured (model: {model})"))

    elif provider == "ollama":
        base_url = ask_string(
            "Ollama API URL",
            default="http://localhost:11434"
        )
        model = ask_string("Model", default="nomic-embed-text")
        config['MWI_OLLAMA_BASE_URL'] = base_url
        config['MWI_EMBED_MODEL'] = model
        print(success(f"Ollama configured (URL: {base_url})"))

    elif provider == "fake":
        config['MWI_EMBED_MODEL'] = 'fake-embedding-model'
        print(info("Using fake embeddings (testing only)"))

    # Fill missing API keys
    for key in ['MWI_OPENAI_API_KEY', 'MWI_MISTRAL_API_KEY', 'MWI_GEMINI_API_KEY', 'MWI_HF_API_KEY']:
        if key not in config:
            config[key] = ''

    if 'MWI_OLLAMA_BASE_URL' not in config:
        config['MWI_OLLAMA_BASE_URL'] = 'http://localhost:11434'

    if 'MWI_EMBED_API_URL' not in config:
        config['MWI_EMBED_API_URL'] = ''

    # NLI configuration
    print()
    nli_choices = [
        ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "mDeBERTa XNLI (multilingual)"),
        ("typeform/distilbert-base-uncased-mnli", "DistilBERT MNLI (English, fast)"),
    ]

    nli_model = ask_choice(
        "Select NLI model:",
        nli_choices,
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )

    config['MWI_NLI_MODEL_NAME'] = nli_model
    config['MWI_NLI_BACKEND'] = 'fallback'
    config['MWI_NLI_TORCH_THREADS'] = '1'
    config['MWI_NLI_FALLBACK_MODEL_NAME'] = 'typeform/distilbert-base-uncased-mnli'

    # Similarity backend
    print()
    backend_choices = [
        ("faiss", "FAISS (fast, recommended)"),
        ("bruteforce", "Brute-force (slower)"),
    ]

    backend = ask_choice(
        "Similarity backend:",
        backend_choices,
        default="faiss"
    )

    config['MWI_SIMILARITY_BACKEND'] = backend
    config['MWI_SIMILARITY_TOP_K'] = '50'
    config['MWI_NLI_ENTAILMENT_THRESHOLD'] = '0.8'
    config['MWI_NLI_CONTRADICTION_THRESHOLD'] = '0.8'

    print(success(f"LLM configuration complete (backend: {backend})"))


def display_env_summary(config: dict):
    """Display summary of environment configuration."""
    print(success(f"Build settings:"))
    print(f"  WITH_ML: {config.get('MYWI_WITH_ML', '0')}")
    print(f"  WITH_PLAYWRIGHT_BROWSERS: {config.get('MYWI_WITH_PLAYWRIGHT_BROWSERS', '0')}")
    print()

    print(success(f"Runtime settings:"))
    print(f"  TZ: {config.get('TZ', 'UTC')}")
    print(f"  HOST_DATA_DIR: {config.get('HOST_DATA_DIR', './data')}")
    print()

    # API keys (masked)
    api_keys = {
        'SerpAPI': config.get('MWI_SERPAPI_API_KEY', ''),
        'SEO Rank': config.get('MWI_SEORANK_API_KEY', ''),
        'OpenRouter': config.get('MWI_OPENROUTER_API_KEY', ''),
        'OpenAI': config.get('MWI_OPENAI_API_KEY', ''),
        'Mistral': config.get('MWI_MISTRAL_API_KEY', ''),
        'Gemini': config.get('MWI_GEMINI_API_KEY', ''),
        'HuggingFace': config.get('MWI_HF_API_KEY', ''),
    }

    configured_apis = [name for name, key in api_keys.items() if key]
    if configured_apis:
        print(success(f"Configured APIs:"))
        for name in configured_apis:
            key = api_keys[name]
            print(f"  {name}: {truncate_secret(key)}")
    else:
        print(info("No APIs configured"))

    print()

    # LLM settings
    if config.get('MYWI_WITH_ML') == '1':
        print(success(f"LLM settings:"))
        print(f"  Embedding provider: {config.get('MWI_EMBED_PROVIDER', 'N/A')}")
        print(f"  Embedding model: {config.get('MWI_EMBED_MODEL', 'N/A')}")
        print(f"  NLI model: {config.get('MWI_NLI_MODEL_NAME', 'N/A')[:50]}...")
        print(f"  Similarity backend: {config.get('MWI_SIMILARITY_BACKEND', 'N/A')}")


def write_env_file(config: dict, file_path: str):
    """
    Write configuration to .env file.

    Args:
        config: Dictionary of environment variables
        file_path: Path to .env file
    """
    from datetime import datetime

    with open(file_path, 'w') as f:
        f.write("# MyWebIntelligence - Docker Compose Environment Configuration\n")
        f.write(f"# Generated by install-docker-compose.py on {datetime.now()}\n")
        f.write("#\n")
        f.write("# WARNING: This file contains sensitive information (API keys).\n")
        f.write("# Do NOT commit this file to version control.\n")
        f.write("#\n\n")

        # Build-time settings
        f.write("# ---------------------------------------------------------------------------\n")
        f.write("# Build-time toggles (Docker build args)\n")
        f.write("# ---------------------------------------------------------------------------\n")
        f.write(f"MYWI_WITH_ML={config.get('MYWI_WITH_ML', '0')}\n")
        f.write(f"MYWI_WITH_PLAYWRIGHT_BROWSERS={config.get('MYWI_WITH_PLAYWRIGHT_BROWSERS', '0')}\n\n")

        # Runtime settings
        f.write("# ---------------------------------------------------------------------------\n")
        f.write("# Runtime settings (host + container)\n")
        f.write("# ---------------------------------------------------------------------------\n")
        f.write(f"TZ={config.get('TZ', 'UTC')}\n")
        f.write(f"HOST_DATA_DIR={config.get('HOST_DATA_DIR', './data')}\n")
        f.write(f"MYWI_DATA_DIR={config.get('MYWI_DATA_DIR', '/app/data')}\n\n")

        # OpenRouter
        f.write("# ---------------------------------------------------------------------------\n")
        f.write("# OpenRouter relevance gate (optional)\n")
        f.write("# ---------------------------------------------------------------------------\n")
        f.write(f"MWI_OPENROUTER_ENABLED={config.get('MWI_OPENROUTER_ENABLED', 'false')}\n")
        f.write(f"MWI_OPENROUTER_API_KEY={config.get('MWI_OPENROUTER_API_KEY', '')}\n")
        f.write(f"MWI_OPENROUTER_MODEL={config.get('MWI_OPENROUTER_MODEL', 'deepseek/deepseek-chat-v3.1')}\n")
        f.write(f"MWI_OPENROUTER_TIMEOUT={config.get('MWI_OPENROUTER_TIMEOUT', '15')}\n")
        f.write(f"MWI_OPENROUTER_MIN_CHARS={config.get('MWI_OPENROUTER_MIN_CHARS', '140')}\n")
        f.write(f"MWI_OPENROUTER_MAX_CHARS={config.get('MWI_OPENROUTER_MAX_CHARS', '12000')}\n")
        f.write(f"MWI_OPENROUTER_MAX_CALLS={config.get('MWI_OPENROUTER_MAX_CALLS', '500')}\n\n")

        # SEO Rank
        f.write("# ---------------------------------------------------------------------------\n")
        f.write("# SEO Rank enrichment\n")
        f.write("# ---------------------------------------------------------------------------\n")
        f.write(f"MWI_SEORANK_API_BASE_URL={config.get('MWI_SEORANK_API_BASE_URL', '')}\n")
        f.write(f"MWI_SEORANK_API_KEY={config.get('MWI_SEORANK_API_KEY', '')}\n\n")

        # SerpAPI
        f.write("# ---------------------------------------------------------------------------\n")
        f.write("# SerpAPI bootstrap\n")
        f.write("# ---------------------------------------------------------------------------\n")
        f.write(f"MWI_SERPAPI_API_KEY={config.get('MWI_SERPAPI_API_KEY', '')}\n\n")

        # Embeddings
        f.write("# ---------------------------------------------------------------------------\n")
        f.write("# Embeddings (bi-encoder)\n")
        f.write("# ---------------------------------------------------------------------------\n")
        f.write(f"MWI_EMBED_PROVIDER={config.get('MWI_EMBED_PROVIDER', 'mistral')}\n")
        f.write(f"MWI_EMBED_MODEL={config.get('MWI_EMBED_MODEL', 'mistral-embed')}\n")
        f.write(f"MWI_EMBED_API_URL={config.get('MWI_EMBED_API_URL', '')}\n")
        f.write(f"MWI_OPENAI_API_KEY={config.get('MWI_OPENAI_API_KEY', '')}\n")
        f.write(f"MWI_MISTRAL_API_KEY={config.get('MWI_MISTRAL_API_KEY', '')}\n")
        f.write(f"MWI_GEMINI_API_KEY={config.get('MWI_GEMINI_API_KEY', '')}\n")
        f.write(f"MWI_HF_API_KEY={config.get('MWI_HF_API_KEY', '')}\n")
        f.write(f"MWI_OLLAMA_BASE_URL={config.get('MWI_OLLAMA_BASE_URL', 'http://localhost:11434')}\n\n")

        # NLI
        f.write("# ---------------------------------------------------------------------------\n")
        f.write("# Semantic Search & NLI\n")
        f.write("# ---------------------------------------------------------------------------\n")
        f.write(f"MWI_NLI_MODEL_NAME={config.get('MWI_NLI_MODEL_NAME', 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7')}\n")
        f.write(f"MWI_NLI_BACKEND={config.get('MWI_NLI_BACKEND', 'fallback')}\n")
        f.write(f"MWI_NLI_TORCH_THREADS={config.get('MWI_NLI_TORCH_THREADS', '1')}\n")
        f.write(f"MWI_NLI_FALLBACK_MODEL_NAME={config.get('MWI_NLI_FALLBACK_MODEL_NAME', 'typeform/distilbert-base-uncased-mnli')}\n")
        f.write(f"MWI_SIMILARITY_BACKEND={config.get('MWI_SIMILARITY_BACKEND', 'faiss')}\n")
        f.write(f"MWI_SIMILARITY_TOP_K={config.get('MWI_SIMILARITY_TOP_K', '50')}\n")
        f.write(f"MWI_NLI_ENTAILMENT_THRESHOLD={config.get('MWI_NLI_ENTAILMENT_THRESHOLD', '0.8')}\n")
        f.write(f"MWI_NLI_CONTRADICTION_THRESHOLD={config.get('MWI_NLI_CONTRADICTION_THRESHOLD', '0.8')}\n")


def print_next_steps(level: str):
    """Print next steps after setup."""
    print()
    print_section("Next Steps")
    print()

    print("1. Build and start Docker Compose services:")
    print(info("   docker compose up -d --build"))
    print()

    print("2. Initialize database (first run only):")
    print(info("   docker compose exec mwi python mywi.py db setup"))
    print()

    print("3. Verify installation:")
    print(info("   docker compose exec mwi python mywi.py land list"))
    print()

    if level in ['api', 'llm']:
        print("4. Test API connections:")
        print(info("   docker compose exec mwi python scripts/test-apis.py --all"))
        print()

    if level == 'llm':
        print("5. Check ML environment:")
        print(info("   docker compose exec mwi python mywi.py embedding check"))
        print()

    print("Management commands:")
    print(info("   docker compose up -d          # Start services"))
    print(info("   docker compose down           # Stop services"))
    print(info("   docker compose logs mwi       # View logs"))
    print(info("   docker compose exec mwi bash  # Enter container"))
    print()

    print(warning("Important:"))
    print("  - Your data is stored at: " + info(f"HOST_DATA_DIR (from .env)"))
    print("  - Inside container: /app/data")
    print("  - Do NOT commit .env to version control (contains API keys)")
    print()


if __name__ == '__main__':
    sys.exit(main())
