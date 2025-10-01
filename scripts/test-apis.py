#!/usr/bin/env python3
"""
MyWebIntelligence - API Testing Utility

Test configured API connections to verify credentials and connectivity.

Supports:
- SerpAPI (Google search)
- SEO Rank API
- OpenRouter (LLM)

Usage:
    python scripts/test-apis.py --all
    python scripts/test-apis.py --serpapi
    python scripts/test-apis.py --seorank
    python scripts/test-apis.py --openrouter
    python scripts/test-apis.py --help
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import install_utils and settings
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from install_utils import (
    print_banner,
    print_section,
    success,
    error,
    info,
    warning,
)


def main():
    """Main API testing entry point."""
    parser = argparse.ArgumentParser(
        description="Test MyWI API connections"
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Test all configured APIs'
    )
    parser.add_argument(
        '--serpapi',
        action='store_true',
        help='Test SerpAPI'
    )
    parser.add_argument(
        '--seorank',
        action='store_true',
        help='Test SEO Rank API'
    )
    parser.add_argument(
        '--openrouter',
        action='store_true',
        help='Test OpenRouter API'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output (show full responses)'
    )
    args = parser.parse_args()

    # If no specific test selected, show help
    if not (args.all or args.serpapi or args.seorank or args.openrouter):
        parser.print_help()
        return 1

    print_banner("MyWebIntelligence - API Testing", width=65)

    # Load settings
    try:
        import settings
    except ImportError:
        print(error("settings.py not found"))
        print(info("Run 'python scripts/install-basic.py' first"))
        return 1

    results = {}

    # Test APIs
    if args.all or args.serpapi:
        results['SerpAPI'] = test_serpapi(settings, args.verbose)

    if args.all or args.seorank:
        results['SEO Rank'] = test_seorank(settings, args.verbose)

    if args.all or args.openrouter:
        results['OpenRouter'] = test_openrouter(settings, args.verbose)

    # Summary
    print()
    print_section("Summary")
    print()

    all_passed = True
    for name, passed in results.items():
        if passed:
            print(success(f"{name}: OK"))
        else:
            print(error(f"{name}: FAILED"))
            all_passed = False

    print()

    return 0 if all_passed else 1


def test_serpapi(settings, verbose: bool = False) -> bool:
    """
    Test SerpAPI connection.

    Args:
        settings: Settings module
        verbose: Show detailed response

    Returns:
        True if test passed, False otherwise
    """
    print_section("Testing SerpAPI")

    # Check if API key is configured
    api_key = getattr(settings, 'serpapi_api_key', '')
    if not api_key or api_key == 'os.getenv("MWI_SERPAPI_API_KEY", "")':
        print(warning("SerpAPI key not configured"))
        print(info("Configure in settings.py or run: python scripts/install-api.py"))
        return False

    print(info(f"API key: {api_key[:10]}..."))

    try:
        import requests

        url = getattr(settings, 'serpapi_base_url', 'https://serpapi.com/search')
        timeout = getattr(settings, 'serpapi_timeout', 15)

        params = {
            'api_key': api_key,
            'engine': 'google',
            'q': 'test',
            'num': 1,
        }

        print(info(f"Sending test request to {url}..."))

        response = requests.get(url, params=params, timeout=timeout)

        if response.status_code == 200:
            data = response.json()

            if 'error' in data:
                print(error(f"API error: {data['error']}"))
                return False

            # Check for results
            if 'organic_results' in data or 'answer_box' in data:
                print(success("SerpAPI connection successful"))

                if verbose:
                    print()
                    print("Response preview:")
                    organic = data.get('organic_results', [])
                    if organic:
                        print(f"  Found {len(organic)} results")
                        print(f"  First result: {organic[0].get('title', 'N/A')}")

                return True
            else:
                print(warning("Unexpected response format"))
                if verbose:
                    print(f"Response: {data}")
                return False

        elif response.status_code == 401:
            print(error("Authentication failed - invalid API key"))
            return False

        elif response.status_code == 403:
            print(error("Access forbidden - check your API plan"))
            return False

        else:
            print(error(f"HTTP {response.status_code}: {response.text[:200]}"))
            return False

    except requests.exceptions.Timeout:
        print(error(f"Request timed out after {timeout}s"))
        return False

    except requests.exceptions.ConnectionError:
        print(error("Connection failed - check your internet connection"))
        return False

    except Exception as e:
        print(error(f"Test failed: {e}"))
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def test_seorank(settings, verbose: bool = False) -> bool:
    """
    Test SEO Rank API connection.

    Args:
        settings: Settings module
        verbose: Show detailed response

    Returns:
        True if test passed, False otherwise
    """
    print_section("Testing SEO Rank API")

    # Check if API key is configured
    api_key = getattr(settings, 'seorank_api_key', '')
    if not api_key or api_key == 'os.getenv("MWI_SEORANK_API_KEY", "")':
        print(warning("SEO Rank API key not configured"))
        print(info("Configure in settings.py or run: python scripts/install-api.py"))
        return False

    base_url = getattr(settings, 'seorank_api_base_url', '')
    if not base_url or 'os.getenv' in base_url:
        print(warning("SEO Rank API base URL not configured"))
        return False

    print(info(f"API key: {api_key[:10]}..."))
    print(info(f"Base URL: {base_url}"))

    try:
        import requests

        timeout = getattr(settings, 'seorank_timeout', 15)

        # Test with a known public URL
        test_url = "https://www.wikipedia.org/"

        params = {
            'key': api_key,
            'url': test_url,
        }

        print(info(f"Sending test request for {test_url}..."))

        response = requests.get(base_url, params=params, timeout=timeout)

        if response.status_code == 200:
            try:
                data = response.json()

                # Check for expected fields
                expected_fields = ['sr_domain', 'sr_rank']
                found_fields = [f for f in expected_fields if f in data]

                if found_fields:
                    print(success("SEO Rank API connection successful"))

                    if verbose:
                        print()
                        print("Response preview:")
                        for field in ['sr_domain', 'sr_rank', 'sr_traffic', 'sr_kwords']:
                            if field in data:
                                print(f"  {field}: {data[field]}")

                    return True
                else:
                    print(warning("Unexpected response format"))
                    if verbose:
                        print(f"Response: {data}")
                    return False

            except ValueError:
                print(error("Invalid JSON response"))
                if verbose:
                    print(f"Response text: {response.text[:200]}")
                return False

        elif response.status_code == 401:
            print(error("Authentication failed - invalid API key"))
            return False

        elif response.status_code == 403:
            print(error("Access forbidden - check your API plan"))
            return False

        else:
            print(error(f"HTTP {response.status_code}: {response.text[:200]}"))
            return False

    except requests.exceptions.Timeout:
        print(error(f"Request timed out after {timeout}s"))
        return False

    except requests.exceptions.ConnectionError:
        print(error("Connection failed - check your internet connection"))
        return False

    except Exception as e:
        print(error(f"Test failed: {e}"))
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def test_openrouter(settings, verbose: bool = False) -> bool:
    """
    Test OpenRouter API connection.

    Args:
        settings: Settings module
        verbose: Show detailed response

    Returns:
        True if test passed, False otherwise
    """
    print_section("Testing OpenRouter API")

    # Check if OpenRouter is enabled
    enabled = getattr(settings, 'openrouter_enabled', False)
    if isinstance(enabled, str):
        # Handle os.getenv() format
        enabled = 'true' in enabled.lower()

    if not enabled:
        print(warning("OpenRouter not enabled"))
        print(info("Enable in settings.py or run: python scripts/install-api.py"))
        return False

    # Check if API key is configured
    api_key = getattr(settings, 'openrouter_api_key', '')
    if not api_key or api_key == 'os.getenv("MWI_OPENROUTER_API_KEY", "")':
        print(warning("OpenRouter API key not configured"))
        return False

    model = getattr(settings, 'openrouter_model', 'deepseek/deepseek-chat-v3.1')
    if isinstance(model, str) and 'os.getenv' in model:
        # Extract default from os.getenv
        import re
        match = re.search(r'"([^"]+)"\)', model)
        if match:
            model = match.group(1)

    print(info(f"API key: {api_key[:10]}..."))
    print(info(f"Model: {model}"))

    try:
        import requests

        timeout = getattr(settings, 'openrouter_timeout', 15)
        if isinstance(timeout, str):
            timeout = int(re.search(r'(\d+)', timeout).group(1))

        url = "https://openrouter.ai/api/v1/chat/completions"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": "Hello, this is a test. Reply with 'OK'."}
            ],
            "max_tokens": 10,
        }

        print(info(f"Sending test request to {url}..."))

        response = requests.post(url, json=payload, headers=headers, timeout=timeout)

        if response.status_code == 200:
            data = response.json()

            if 'choices' in data and len(data['choices']) > 0:
                reply = data['choices'][0].get('message', {}).get('content', '')
                print(success("OpenRouter API connection successful"))

                if verbose:
                    print()
                    print(f"Model reply: {reply}")
                    if 'usage' in data:
                        print(f"Tokens used: {data['usage']}")

                return True
            else:
                print(warning("Unexpected response format"))
                if verbose:
                    print(f"Response: {data}")
                return False

        elif response.status_code == 401:
            print(error("Authentication failed - invalid API key"))
            return False

        elif response.status_code == 403:
            print(error("Access forbidden - check your API credits"))
            return False

        elif response.status_code == 429:
            print(error("Rate limit exceeded - too many requests"))
            return False

        else:
            print(error(f"HTTP {response.status_code}: {response.text[:200]}"))
            return False

    except requests.exceptions.Timeout:
        print(error(f"Request timed out after {timeout}s"))
        return False

    except requests.exceptions.ConnectionError:
        print(error("Connection failed - check your internet connection"))
        return False

    except Exception as e:
        print(error(f"Test failed: {e}"))
        if verbose:
            import traceback
            traceback.print_exc()
        return False


if __name__ == '__main__':
    sys.exit(main())
