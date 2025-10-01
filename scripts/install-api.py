#!/usr/bin/env python3
"""
MyWebIntelligence - API Configuration

Interactive configuration wizard for external API integrations.
Updates settings.py with API credentials and configuration.

APIs supported:
- SerpAPI (Google search results)
- SEO Rank (traffic and SEO metrics)
- OpenRouter (LLM relevance filtering)

Usage:
    python scripts/install-api.py
    python scripts/install-api.py --help
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
    """Main API configuration wizard."""
    parser = argparse.ArgumentParser(
        description="MyWebIntelligence API Configuration"
    )
    parser.add_argument(
        '--output',
        default='settings.py',
        help='Output file path (default: settings.py)'
    )
    args = parser.parse_args()

    print_banner("MyWebIntelligence - API Configuration", width=65)
    print_help("This script will configure external API integrations.")
    print_help("Press Ctrl+C at any time to cancel.")
    print()

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
        # Configure APIs
        configure_serpapi(config)
        configure_seorank(config)
        configure_openrouter(config)

        # Confirm and save
        api_keys = [
            'serpapi_api_key', 'serpapi_base_url', 'serpapi_timeout',
            'seorank_api_key', 'seorank_api_base_url', 'seorank_timeout', 'seorank_request_delay',
            'openrouter_enabled', 'openrouter_api_key', 'openrouter_model',
            'openrouter_timeout', 'openrouter_readable_min_chars',
            'openrouter_readable_max_chars', 'openrouter_max_calls_per_run',
        ]

        if not confirm_config(config, api_keys):
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
        print_next_steps()

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


def configure_serpapi(config: dict):
    """Configure SerpAPI for Google search integration."""
    print_section("SerpAPI Configuration", step=(1, 3))
    print_help("Bootstrap research lands with URLs from Google search results")
    print_help("Get your API key at: https://serpapi.com/")
    print_help("Free tier: 100 searches/month")
    print()

    enable_serpapi = ask_bool("Enable SerpAPI?", default=False)

    if not enable_serpapi:
        config['serpapi_api_key'] = 'os.getenv("MWI_SERPAPI_API_KEY", "")'
        print(info("SerpAPI disabled"))
        return

    print()
    api_key = ask_secret("SerpAPI Key", required=True)

    # Basic validation
    if len(api_key) < 32:
        print(warning("API key seems short - please verify it's correct"))

    config['serpapi_api_key'] = f'os.getenv("MWI_SERPAPI_API_KEY", "{api_key}")'
    config['serpapi_base_url'] = "https://serpapi.com/search"
    config['serpapi_timeout'] = 15

    print(success(f"SerpAPI configured (key: {truncate_secret(api_key)})"))


def configure_seorank(config: dict):
    """Configure SEO Rank API for traffic metrics."""
    print_section("SEO Rank Configuration", step=(2, 3))
    print_help("Enrich pages with SEO metrics, traffic estimates, and social stats")
    print_help("Requires API access and key from your provider")
    print()

    enable_seorank = ask_bool("Enable SEO Rank enrichment?", default=False)

    if not enable_seorank:
        config['seorank_api_key'] = 'os.getenv("MWI_SEORANK_API_KEY", "")'
        config['seorank_api_base_url'] = 'os.getenv("MWI_SEORANK_API_BASE_URL", "https://seo-rank.my-addr.com/api2/sr+fb")'
        print(info("SEO Rank disabled"))
        return

    print()
    api_base_url = ask_string(
        "API Base URL",
        default="https://seo-rank.my-addr.com/api2/sr+fb",
        required=True
    )

    api_key = ask_secret("SEO Rank API Key", required=True)

    request_delay = ask_float(
        "Request delay between calls (seconds)",
        default=1.0,
        min_val=0.1,
        max_val=10.0
    )

    config['seorank_api_base_url'] = f'os.getenv("MWI_SEORANK_API_BASE_URL", "{api_base_url}")'
    config['seorank_api_key'] = f'os.getenv("MWI_SEORANK_API_KEY", "{api_key}")'
    config['seorank_timeout'] = 15
    config['seorank_request_delay'] = request_delay

    print(success(f"SEO Rank configured (key: {truncate_secret(api_key)})"))


def configure_openrouter(config: dict):
    """Configure OpenRouter for LLM relevance filtering."""
    print_section("OpenRouter Configuration", step=(3, 3))
    print_help("Use AI models to filter irrelevant pages during crawling")
    print_help("Get your API key at: https://openrouter.ai/")
    print_help("Pay-per-use pricing, very affordable for small models")
    print()

    enable_openrouter = ask_bool("Enable OpenRouter LLM gate?", default=False)

    if not enable_openrouter:
        config['openrouter_enabled'] = 'os.getenv("MWI_OPENROUTER_ENABLED", "false").lower() == "true"'
        config['openrouter_api_key'] = 'os.getenv("MWI_OPENROUTER_API_KEY", "")'
        config['openrouter_model'] = 'os.getenv("MWI_OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1")'
        print(info("OpenRouter disabled"))
        return

    print()
    api_key = ask_secret("OpenRouter API Key", required=True)

    # Model selection
    print()
    model_choices = [
        ("deepseek/deepseek-chat-v3.1", "DeepSeek v3.1 (economical, recommended)"),
        ("openai/gpt-4o-mini", "GPT-4o Mini (fast, cheap)"),
        ("anthropic/claude-3-haiku", "Claude 3 Haiku (quality)"),
        ("google/gemini-1.5-flash", "Gemini 1.5 Flash (balanced)"),
        ("meta-llama/llama-3.1-8b-instruct", "Llama 3.1 8B (open source)"),
        ("mistralai/mistral-small-latest", "Mistral Small (efficient)"),
        ("qwen/qwen2.5-7b-instruct", "Qwen 2.5 7B (multilingual)"),
        ("cohere/command-r-mini", "Command R Mini (fast)"),
        ("custom", "Custom model (enter slug manually)"),
    ]

    model = ask_choice(
        "Select OpenRouter model:",
        model_choices,
        default="deepseek/deepseek-chat-v3.1"
    )

    if model == "custom":
        print()
        model = ask_string(
            "Enter custom model slug (e.g., provider/model-name)",
            required=True
        )

    print()
    print_help("Content length limits (characters)")

    min_chars = ask_int(
        "Minimum content length",
        default=140,
        min_val=50,
        max_val=1000
    )

    max_chars = ask_int(
        "Maximum content length",
        default=12000,
        min_val=1000,
        max_val=50000
    )

    print()
    timeout = ask_int(
        "Request timeout (seconds)",
        default=15,
        min_val=5,
        max_val=60
    )

    max_calls = ask_int(
        "Maximum LLM calls per run",
        default=500,
        min_val=1,
        max_val=10000
    )

    # Example models list (for reference in settings)
    config['openrouter_model_examples'] = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku",
        "google/gemini-1.5-flash",
        "meta-llama/llama-3.1-8b-instruct",
        "mistralai/mistral-small-latest",
        "qwen/qwen2.5-7b-instruct",
        "cohere/command-r-mini",
    ]

    config['openrouter_enabled'] = 'os.getenv("MWI_OPENROUTER_ENABLED", "true").lower() == "true"'
    config['openrouter_api_key'] = f'os.getenv("MWI_OPENROUTER_API_KEY", "{api_key}")'
    config['openrouter_model'] = f'os.getenv("MWI_OPENROUTER_MODEL", "{model}")'
    config['openrouter_timeout'] = f'int(os.getenv("MWI_OPENROUTER_TIMEOUT", "{timeout}"))'
    config['openrouter_readable_min_chars'] = f'int(os.getenv("MWI_OPENROUTER_MIN_CHARS", "{min_chars}"))'
    config['openrouter_readable_max_chars'] = f'int(os.getenv("MWI_OPENROUTER_MAX_CHARS", "{max_chars}"))'
    config['openrouter_max_calls_per_run'] = f'int(os.getenv("MWI_OPENROUTER_MAX_CALLS", "{max_calls}"))'

    print(success(f"OpenRouter configured"))
    print(info(f"  Model: {model}"))
    print(info(f"  Key: {truncate_secret(api_key)}"))


def print_next_steps():
    """Print next steps after configuration."""
    print()
    print_section("Next Steps")
    print()
    print("Test your API configuration:")
    print(info("   python scripts/test-apis.py --all"))
    print()
    print("Use SerpAPI to bootstrap a land:")
    print(info('   python mywi.py land urlist --name="MyLand" --query="search terms"'))
    print()
    print("Enrich pages with SEO Rank:")
    print(info('   python mywi.py land seorank --name="MyLand" --limit=100'))
    print()
    print("Use OpenRouter relevance filter:")
    print(info('   python mywi.py land readable --name="MyLand" --llm=true'))
    print()
    print("Optional: Configure embeddings and NLI")
    print(info("   python scripts/install-llm.py"))
    print()


if __name__ == '__main__':
    sys.exit(main())
