#!/usr/bin/env python3
"""
MyWebIntelligence - Basic Installation Setup

Interactive configuration wizard for basic MyWI installation.
Generates settings.py with core configuration.

Usage:
    python scripts/install-basic.py
    python scripts/install-basic.py --help
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
    ask_path,
    backup_file,
    read_settings,
    write_settings,
    confirm_config,
    ensure_directory,
)


def main():
    """Main installation wizard."""
    parser = argparse.ArgumentParser(
        description="MyWebIntelligence Basic Installation Setup"
    )
    parser.add_argument(
        '--output',
        default='settings.py',
        help='Output file path (default: settings.py)'
    )
    args = parser.parse_args()

    print_banner("MyWebIntelligence - Basic Installation Setup", width=65)
    print_help("This script will guide you through basic configuration.")
    print_help("Press Ctrl+C at any time to cancel.")
    print()

    try:
        config = configure_basic()

        if not confirm_config(config, list(config.keys())):
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
        print(warning("Installation cancelled by user"))
        return 1
    except Exception as e:
        print()
        print(error(f"Installation failed: {e}"))
        return 1


def configure_basic() -> dict:
    """
    Run the basic configuration wizard.

    Returns:
        Dictionary of configuration values
    """
    config = {}

    # ========================================================================
    # Step 1: Data Storage
    # ========================================================================
    print_section("Data Storage", step=(1, 6))
    print_help("Where should MyWI store the database and exports?")
    print()

    data_location = ask_path(
        prompt="Data location",
        default="./data",
        must_exist=False,
        create_if_missing=True
    )
    config['data_location'] = data_location

    # Create directory if it doesn't exist
    ensure_directory(data_location)
    print(success(f"Data directory: {data_location}"))

    config['archive'] = ask_bool(
        "Enable automatic archiving of old data?",
        default=False
    )

    # ========================================================================
    # Step 2: Network Configuration
    # ========================================================================
    print_section("Network Configuration", step=(2, 6))
    print_help("HTTP timeout for web requests (seconds)")
    print()

    config['default_timeout'] = ask_int(
        prompt="Timeout",
        default=10,
        min_val=5,
        max_val=60
    )
    print(success(f"Timeout set to {config['default_timeout']} seconds"))
    print()

    print_help("Concurrent HTTP connections")
    print_help("Recommendation: 10-20 for most use cases")
    print()

    config['parallel_connections'] = ask_int(
        prompt="Parallel connections",
        default=10,
        min_val=1,
        max_val=50
    )
    print(success(f"Will use {config['parallel_connections']} parallel connections"))
    print()

    print_help("User agent for HTTP requests (optional)")
    print_help("Leave empty to use default Python user agent")
    print()

    config['user_agent'] = ask_string(
        prompt="User agent",
        default="",
        required=False,
        examples=[
            "Mozilla/5.0 (compatible; MyWI/1.0)",
            "MyResearchBot/1.0 (+https://example.com/bot)"
        ]
    )

    if config['user_agent']:
        print(success(f"User agent: {config['user_agent']}"))
    else:
        print(info("Using default user agent"))

    # ========================================================================
    # Step 3: Dynamic Media Extraction
    # ========================================================================
    print_section("Dynamic Media Extraction", step=(3, 6))
    print_help("Use headless browser to extract media from JavaScript-heavy sites")
    print_help("Requires Playwright (run: python install_playwright.py)")
    print()

    config['dynamic_media_extraction'] = ask_bool(
        "Enable dynamic media extraction?",
        default=True
    )

    if config['dynamic_media_extraction']:
        print(success("Dynamic media extraction enabled"))
        print(info("Remember to install Playwright: python install_playwright.py"))
    else:
        print(info("Dynamic media extraction disabled"))

    # ========================================================================
    # Step 4: Media Analysis Settings
    # ========================================================================
    print_section("Media Analysis Settings", step=(4, 6))
    print_help("Configure media file analysis and filtering")
    print()

    config['media_analysis'] = ask_bool(
        "Enable media analysis?",
        default=True
    )

    if config['media_analysis']:
        print()
        print_help("Minimum dimensions (width × height) in pixels")
        config['media_min_width'] = ask_int(
            "Minimum width",
            default=200,
            min_val=50,
            max_val=5000
        )
        config['media_min_height'] = ask_int(
            "Minimum height",
            default=200,
            min_val=50,
            max_val=5000
        )

        print()
        print_help("Maximum file size in MB")
        max_mb = ask_int(
            "Max file size (MB)",
            default=10,
            min_val=1,
            max_val=100
        )
        config['media_max_file_size'] = max_mb * 1024 * 1024

        print()
        config['media_download_timeout'] = ask_int(
            "Download timeout (seconds)",
            default=30,
            min_val=10,
            max_val=120
        )

        config['media_max_retries'] = ask_int(
            "Max download retries",
            default=2,
            min_val=0,
            max_val=5
        )

        print()
        config['media_extract_colors'] = ask_bool(
            "Extract dominant colors from images?",
            default=True
        )

        if config['media_extract_colors']:
            config['media_n_dominant_colors'] = ask_int(
                "Number of dominant colors to extract",
                default=5,
                min_val=1,
                max_val=10
            )

        config['media_extract_exif'] = ask_bool(
            "Extract EXIF metadata from images?",
            default=True
        )

        config['media_analyze_content'] = ask_bool(
            "Analyze image content (requires ML dependencies)?",
            default=False
        )

        print(success("Media analysis configured"))
    else:
        print(info("Media analysis disabled"))

    # ========================================================================
    # Step 5: Domain Heuristics
    # ========================================================================
    print_section("Domain Heuristics", step=(5, 6))
    print_help("URL patterns for social media and common platforms")
    print_help("These help extract clean profile/page URLs")
    print()

    use_default_heuristics = ask_bool(
        "Use default heuristics (Facebook, Twitter, LinkedIn, etc.)?",
        default=True
    )

    if use_default_heuristics:
        config['heuristics'] = {
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
        print(success("Using default heuristics for 10 platforms"))
    else:
        config['heuristics'] = {}
        print(info("No heuristics configured (can be added manually later)"))

    # ========================================================================
    # Step 6: Language Configuration
    # ========================================================================
    print_section("Language Configuration", step=(6, 6))
    print_help("Default language for text processing and stemming")
    print()

    language_choices = [
        ("fr", "French (Français)"),
        ("en", "English"),
        ("de", "German (Deutsch)"),
        ("es", "Spanish (Español)"),
        ("it", "Italian (Italiano)"),
        ("pt", "Portuguese (Português)"),
    ]

    # Note: This is informational - actual language is set per-land
    print_help("Note: Language is set per research land, not globally.")
    print_help("This is just for your reference.")
    print()

    return config


def print_next_steps():
    """Print next steps after installation."""
    print()
    print_section("Next Steps")
    print()
    print("1. Initialize database:")
    print(info("   python mywi.py db setup"))
    print()
    print("2. Create your first land:")
    print(info('   python mywi.py land create --name="MyResearch" --desc="Description"'))
    print()
    print("3. Add terms to your land:")
    print(info('   python mywi.py land addterm --land="MyResearch" --terms="keyword1, keyword2"'))
    print()
    print("4. Add URLs:")
    print(info('   python mywi.py land addurl --land="MyResearch" --urls="https://example.com"'))
    print()
    print("5. Crawl URLs:")
    print(info('   python mywi.py land crawl --name="MyResearch"'))
    print()
    print("Optional: Configure external APIs")
    print(info("   python scripts/install-api.py"))
    print()
    print("Optional: Configure embeddings and LLM")
    print(info("   python scripts/install-llm.py"))
    print()


if __name__ == '__main__':
    sys.exit(main())
