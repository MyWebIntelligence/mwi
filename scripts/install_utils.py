"""
Utility functions for MyWI installation scripts.

Provides common validation, I/O, and formatting functions
used across all interactive installation scripts.
"""

import os
import sys
import getpass
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple, Any


# ============================================================================
# ANSI Colors and Formatting
# ============================================================================

class Colors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

    # Foreground colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'


def supports_color() -> bool:
    """Check if terminal supports ANSI colors."""
    # Windows cmd.exe doesn't support ANSI by default
    if sys.platform == 'win32':
        try:
            import colorama
            colorama.init()
            return True
        except ImportError:
            return False
    return True


USE_COLOR = supports_color()


def colored(text: str, color: str = '', bold: bool = False) -> str:
    """Apply color to text if terminal supports it."""
    if not USE_COLOR:
        return text
    prefix = (Colors.BOLD if bold else '') + color
    return f"{prefix}{text}{Colors.RESET}"


def success(text: str) -> str:
    """Format success message in green."""
    return colored(f"✓ {text}", Colors.GREEN)


def error(text: str) -> str:
    """Format error message in red."""
    return colored(f"✗ {text}", Colors.RED)


def warning(text: str) -> str:
    """Format warning message in yellow."""
    return colored(f"⚠ {text}", Colors.YELLOW)


def info(text: str) -> str:
    """Format info message in blue."""
    return colored(f"ℹ {text}", Colors.BLUE)


def header(text: str) -> str:
    """Format header text in bold cyan."""
    return colored(text, Colors.CYAN, bold=True)


def dim(text: str) -> str:
    """Format dimmed text."""
    if not USE_COLOR:
        return text
    return f"{Colors.DIM}{text}{Colors.RESET}"


# ============================================================================
# UI Components
# ============================================================================

def print_banner(title: str, width: int = 65):
    """Print a centered banner with title."""
    border = "─" * (width - 4)
    print()
    print(f"┌─{border}─┐")
    print(f"│  {title.center(width - 4)}  │")
    print(f"└─{border}─┘")
    print()


def print_section(title: str, step: Optional[Tuple[int, int]] = None):
    """Print a section header with optional step counter."""
    print()
    if step:
        step_text = f"[{step[0]}/{step[1]}] "
    else:
        step_text = ""
    print(header(f"{step_text}{title}"))
    print("─" * 65)


def print_examples(examples: List[str]):
    """Print a list of examples."""
    print(dim("Examples:"))
    for ex in examples:
        print(dim(f"  - {ex}"))


def print_help(text: str):
    """Print help text in dimmed color."""
    print(dim(text))


# ============================================================================
# Input Functions
# ============================================================================

def ask_string(
    prompt: str,
    default: Optional[str] = None,
    required: bool = False,
    examples: Optional[List[str]] = None
) -> str:
    """
    Ask user for a string input.

    Args:
        prompt: Question to ask
        default: Default value (shown in brackets)
        required: If True, empty input is not allowed
        examples: Optional list of example values

    Returns:
        User input or default value
    """
    if examples:
        print_examples(examples)

    if default is not None:
        prompt_text = f"{prompt} [{default}]: "
    else:
        prompt_text = f"{prompt}: "

    while True:
        value = input(prompt_text).strip()

        if not value and default is not None:
            return default

        if not value and required:
            print(error("This field is required"))
            continue

        return value


def ask_int(
    prompt: str,
    default: Optional[int] = None,
    min_val: Optional[int] = None,
    max_val: Optional[int] = None
) -> int:
    """
    Ask user for an integer input with validation.

    Args:
        prompt: Question to ask
        default: Default value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Integer value
    """
    if min_val is not None and max_val is not None:
        print(dim(f"Range: {min_val}-{max_val}"))

    if default is not None:
        prompt_text = f"{prompt} [{default}]: "
    else:
        prompt_text = f"{prompt}: "

    while True:
        value_str = input(prompt_text).strip()

        if not value_str and default is not None:
            return default

        try:
            value = int(value_str)
        except ValueError:
            print(error("Please enter a valid integer"))
            continue

        if min_val is not None and value < min_val:
            print(error(f"Value must be at least {min_val}"))
            continue

        if max_val is not None and value > max_val:
            print(error(f"Value must be at most {max_val}"))
            continue

        return value


def ask_float(
    prompt: str,
    default: Optional[float] = None,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> float:
    """
    Ask user for a float input with validation.

    Args:
        prompt: Question to ask
        default: Default value
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Float value
    """
    if min_val is not None and max_val is not None:
        print(dim(f"Range: {min_val}-{max_val}"))

    if default is not None:
        prompt_text = f"{prompt} [{default}]: "
    else:
        prompt_text = f"{prompt}: "

    while True:
        value_str = input(prompt_text).strip()

        if not value_str and default is not None:
            return default

        try:
            value = float(value_str)
        except ValueError:
            print(error("Please enter a valid number"))
            continue

        if min_val is not None and value < min_val:
            print(error(f"Value must be at least {min_val}"))
            continue

        if max_val is not None and value > max_val:
            print(error(f"Value must be at most {max_val}"))
            continue

        return value


def ask_bool(prompt: str, default: bool = True) -> bool:
    """
    Ask user for a yes/no question.

    Args:
        prompt: Question to ask
        default: Default value (True = yes, False = no)

    Returns:
        Boolean value
    """
    default_text = "Y/n" if default else "y/N"
    prompt_text = f"{prompt} [{default_text}]: "

    while True:
        value = input(prompt_text).strip().lower()

        if not value:
            return default

        if value in ('y', 'yes', 'oui', 'true', '1'):
            return True
        elif value in ('n', 'no', 'non', 'false', '0'):
            return False
        else:
            print(error("Please enter 'y' or 'n'"))


def ask_choice(
    prompt: str,
    choices: List[Tuple[str, str]],
    default: Optional[str] = None
) -> str:
    """
    Ask user to choose from a list of options.

    Args:
        prompt: Question to ask
        choices: List of (value, description) tuples
        default: Default choice (value)

    Returns:
        Selected value
    """
    print()
    print(prompt)
    print()

    for i, (value, desc) in enumerate(choices, 1):
        default_marker = " [DEFAULT]" if value == default else ""
        print(f"  {i}. {desc}{colored(default_marker, Colors.GREEN)}")

    print()

    while True:
        if default:
            choice = input(f"Choice [1-{len(choices)}] (default: {default}): ").strip()
        else:
            choice = input(f"Choice [1-{len(choices)}]: ").strip()

        if not choice and default:
            return default

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(choices):
                return choices[idx][0]
            else:
                print(error(f"Please enter a number between 1 and {len(choices)}"))
        except ValueError:
            print(error("Please enter a valid number"))


def ask_secret(prompt: str, required: bool = True) -> str:
    """
    Ask user for a secret value (e.g., API key) with hidden input.

    Args:
        prompt: Question to ask
        required: If True, empty input is not allowed

    Returns:
        Secret value
    """
    while True:
        value = getpass.getpass(f"{prompt}: ")

        if not value and required:
            print(error("This field is required"))
            continue

        return value


def ask_path(
    prompt: str,
    default: Optional[str] = None,
    must_exist: bool = False,
    create_if_missing: bool = False
) -> str:
    """
    Ask user for a file or directory path.

    Args:
        prompt: Question to ask
        default: Default path
        must_exist: If True, path must exist
        create_if_missing: If True, create directory if it doesn't exist

    Returns:
        Validated path
    """
    if default is not None:
        prompt_text = f"{prompt} [{default}]: "
    else:
        prompt_text = f"{prompt}: "

    while True:
        value = input(prompt_text).strip()

        if not value and default is not None:
            value = default

        if not value:
            print(error("Path is required"))
            continue

        # Expand ~ and environment variables
        value = os.path.expanduser(value)
        value = os.path.expandvars(value)

        path = Path(value)

        if must_exist and not path.exists():
            print(error(f"Path does not exist: {value}"))
            continue

        if create_if_missing and not path.exists():
            print(info(f"Directory will be created: {value}"))

        return value


# ============================================================================
# Configuration Management
# ============================================================================

def backup_file(file_path: str) -> Optional[str]:
    """
    Create a timestamped backup of a file.

    Args:
        file_path: Path to file to backup

    Returns:
        Path to backup file, or None if original doesn't exist
    """
    path = Path(file_path)

    if not path.exists():
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = path.parent / f"{path.stem}.backup.{timestamp}{path.suffix}"

    import shutil
    shutil.copy2(path, backup_path)

    return str(backup_path)


def read_settings() -> dict:
    """
    Read existing settings.py file.

    Returns:
        Dictionary of current settings
    """
    settings_path = Path("settings.py")

    if not settings_path.exists():
        return {}

    # Import settings module dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location("settings", settings_path)
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)

    # Extract all non-private attributes
    config = {}
    for key in dir(settings):
        if not key.startswith('_'):
            config[key] = getattr(settings, key)

    return config


def write_settings(config: dict, file_path: str = "settings.py"):
    """
    Write configuration to settings.py file.

    Args:
        config: Dictionary of settings
        file_path: Path to settings file
    """
    path = Path(file_path)

    with open(path, 'w') as f:
        f.write('"""\n')
        f.write('MyWebIntelligence configuration.\n')
        f.write(f'Generated by interactive installer on {datetime.now()}\n')
        f.write('"""\n\n')
        f.write('import os\n\n')

        # Group settings by category
        categories = {
            'paths': ['data_location', 'archive'],
            'network': ['default_timeout', 'parallel_connections', 'user_agent'],
            'media': [k for k in config if k.startswith('media_') or k == 'dynamic_media_extraction'],
            'heuristics': ['heuristics'],
            'serpapi': [k for k in config if k.startswith('serpapi_')],
            'seorank': [k for k in config if k.startswith('seorank_')],
            'openrouter': [k for k in config if k.startswith('openrouter_')],
            'embeddings': [k for k in config if k.startswith('embed_')],
            'nli': [k for k in config if k.startswith('nli_') or k.startswith('embedding_')],
            'similarity': [k for k in config if k.startswith('similarity_')],
        }

        # Write settings by category
        for category, keys in categories.items():
            if not any(k in config for k in keys):
                continue

            f.write(f"# {category.upper()}\n")
            f.write(f"# {'-' * 70}\n\n")

            for key in keys:
                if key not in config:
                    continue

                value = config[key]

                # Format value appropriately
                if isinstance(value, str):
                    # Check if it's an environment variable pattern
                    if value.startswith('os.getenv('):
                        f.write(f"{key} = {value}\n")
                    else:
                        f.write(f'{key} = "{value}"\n')
                elif isinstance(value, bool):
                    f.write(f"{key} = {value}\n")
                elif isinstance(value, (int, float)):
                    f.write(f"{key} = {value}\n")
                elif isinstance(value, dict):
                    f.write(f"{key} = {{\n")
                    for k, v in value.items():
                        f.write(f'    "{k}": r"{v}",\n')
                    f.write("}\n")
                elif isinstance(value, list):
                    f.write(f"{key} = [\n")
                    for item in value:
                        f.write(f'    "{item}",\n')
                    f.write("]\n")
                else:
                    f.write(f"{key} = {repr(value)}\n")

            f.write("\n")


def confirm_config(config: dict, keys: List[str]) -> bool:
    """
    Display configuration summary and ask for confirmation.

    Args:
        config: Configuration dictionary
        keys: Keys to display

    Returns:
        True if user confirms, False otherwise
    """
    print_section("Configuration Summary")

    for key in keys:
        if key not in config:
            continue

        value = config[key]

        # Mask secrets
        if any(secret in key.lower() for secret in ['key', 'secret', 'password', 'token']):
            if value and isinstance(value, str):
                display_value = value[:8] + "..." if len(value) > 8 else "***"
            else:
                display_value = "***"
        else:
            display_value = value

        print(success(f"{key}: {display_value}"))

    print()
    return ask_bool("Save this configuration?", default=True)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_url(url: str) -> bool:
    """Validate URL format."""
    import re
    pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return pattern.match(url) is not None


def validate_api_key(key: str, provider: str) -> bool:
    """
    Validate API key format (basic check).

    Args:
        key: API key to validate
        provider: Provider name (e.g., 'openai', 'mistral')

    Returns:
        True if format looks valid
    """
    if not key:
        return False

    # Basic length check
    if len(key) < 10:
        return False

    # Provider-specific patterns
    patterns = {
        'openai': r'^sk-[A-Za-z0-9]{20,}$',
        'mistral': r'^[A-Za-z0-9]{20,}$',
        'serpapi': r'^[a-f0-9]{64}$',
    }

    if provider in patterns:
        import re
        return bool(re.match(patterns[provider], key))

    return True  # Generic validation passed


# ============================================================================
# Helper Functions
# ============================================================================

def get_project_root() -> Path:
    """Get the project root directory."""
    # Assume this script is in scripts/ subdirectory
    return Path(__file__).parent.parent


def ensure_directory(path: str) -> Path:
    """Ensure directory exists, creating it if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def truncate_secret(secret: str, visible: int = 8) -> str:
    """Truncate a secret for display purposes."""
    if not secret:
        return "***"
    if len(secret) <= visible:
        return "***"
    return secret[:visible] + "..."
