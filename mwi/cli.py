"""
Command Line Interface
"""
import argparse
from typing import Any
from .controller import (
    DbController,
    DomainController,
    LandController,
    HeuristicController,
    TagController,
    EmbeddingController
)


def command_run(args: Any):
    """Execute a command from arguments provided as dict or namespace.

    Converts dictionary arguments to argparse.Namespace if needed, then
    dispatches the command to the appropriate controller.

    Args:
        args: Command arguments as either a dictionary or argparse.Namespace
            object. If dict, it will be converted to Namespace.

    Returns:
        None. The function delegates to dispatch() which calls the
        appropriate controller method.

    Notes:
        This is the programmatic entry point for running commands without
        parsing command-line arguments. Useful for testing and embedding
        MyWI in other Python applications.
    """
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    dispatch(args)


def command_input():
    """Parse command-line arguments and execute the corresponding command.

    Creates an ArgumentParser with all MyWebIntelligence command-line options,
    parses sys.argv, processes language arguments, and dispatches to the
    appropriate controller.

    Returns:
        None. The function delegates to dispatch() which calls the
        appropriate controller method.

    Notes:
        This is the main entry point for command-line usage. It supports
        nested commands (object, verb, optional subverb) and a wide range
        of options including land management, crawling, export, LLM
        validation, and embedding operations.

        Language arguments are automatically converted from comma-separated
        strings to lists for multi-language support.
    """
    parser = argparse.ArgumentParser(description='MyWebIntelligence Command Line Project Manager.')
    parser.add_argument('object',
                        metavar='object',
                        type=str,
                        help='Object to interact with [db, land, request]')
    parser.add_argument('verb',
                        metavar='verb',
                        type=str,
                        help='Verb depending on target object')
    # Optional sub-verb (e.g., `land llm validate`)
    parser.add_argument('subverb',
                        metavar='subverb',
                        type=str,
                        nargs='?',
                        help='Optional sub-verb for nested commands')
    parser.add_argument('--land',
                        type=str,
                        help='Name of the land to work with')
    parser.add_argument('--name',
                        type=str,
                        help='Name of the object')
    parser.add_argument('--desc',
                        type=str,
                        help='Description of the object')
    parser.add_argument('--type',
                        type=str,
                        help='Export type, see README for reference')
    parser.add_argument('--terms',
                        type=str,
                        help='Terms to add to request dictionnary, comma separated')
    parser.add_argument('--urls',
                        type=str,
                        help='URL to add to request, comma separated',
                        nargs='?')
    parser.add_argument('--path',
                        type=str,
                        help='Path to local file containing URLs',
                        nargs='?')
    parser.add_argument('--limit',
                        type=int,
                        help='Set limit of URLs to crawl',
                        nargs='?',
                        const=0)
    parser.add_argument('--minrel',
                        type=int,
                        help='Set minimum relevance threshold',
                        nargs='?',
                        const=0)
    parser.add_argument('--maxrel',
                        type=int,
                        help='Set maximum relevance threshold',
                        nargs='?',
                        const=0)
    parser.add_argument('--http',
                        type=str,
                        help='Limit crawling to specific http status (re crawling)',
                        nargs='?')
    parser.add_argument('--depth',
                        type=int,
                        help='Only crawl URLs with the specified depth (for land crawl)',
                        nargs='?')
    parser.add_argument('--lang',
                        type=str,
                        help='Language of the project (default: fr)',
                        default='fr',
                        nargs='?')
    parser.add_argument('--merge',
                        type=str,
                        help='Merge strategy for readable: smart_merge, mercury_priority, preserve_existing',
                        default='smart_merge',
                        nargs='?')
    parser.add_argument('--llm',
                        type=str,
                        help='Toggle OpenRouter validation during readable pipeline (true|false, default=false)',
                        default='false')
    parser.add_argument('--query',
                        type=str,
                        help='Search query to fetch URLs from SerpAPI',
                        nargs='?')
    parser.add_argument('--engine',
                        type=str,
                        help='Search engine for urlist (google|bing|duckduckgo)',
                        default='google',
                        choices=['google', 'bing', 'duckduckgo'],
                        nargs='?')
    parser.add_argument('--datestart',
                        type=str,
                        help='Start date (YYYY-MM-DD) for SerpAPI urlist filtering',
                        nargs='?')
    parser.add_argument('--dateend',
                        type=str,
                        help='End date (YYYY-MM-DD) for SerpAPI urlist filtering',
                        nargs='?')
    parser.add_argument('--timestep',
                        type=str,
                        help='Date window size when iterating between datestart/dateend (day|week|month)',
                        default='week',
                        nargs='?')
    parser.add_argument('--progress',
                        action='store_true',
                        help='Display SerpAPI progress per date window')
    parser.add_argument('--sleep',
                        type=float,
                        help='Base delay (seconds) between SerpAPI calls to avoid rate limits',
                        default=1.0,
                        nargs='?')
    parser.add_argument('--threshold',
                        type=float,
                        help='Similarity threshold for embeddings',
                        nargs='?')
    parser.add_argument('--method',
                        type=str,
                        help='Similarity method (default: cosine)',
                        nargs='?')
    parser.add_argument('--backend',
                        type=str,
                        help='Similarity backend for ANN (bruteforce|faiss)',
                        nargs='?')
    parser.add_argument('--topk',
                        type=int,
                        help='Keep at most top-K neighbors per paragraph',
                        nargs='?')
    parser.add_argument('--lshbits',
                        type=int,
                        help='Number of LSH hyperplanes/bits (for cosine_lsh method)',
                        nargs='?')
    parser.add_argument('--maxpairs',
                        type=int,
                        help='Max number of similarity pairs to insert (cap)',
                        nargs='?')
    parser.add_argument('--force',
                        action='store_true',
                        help='Force include expressions with previous LLM verdict = non (for land llm validate)')
    args = parser.parse_args()
    # Always convert lang to a list
    if hasattr(args, "lang") and isinstance(args.lang, str):
        args.lang = [l.strip() for l in args.lang.split(",") if l.strip()]
    dispatch(args)


def dispatch(args):
    """Dispatch parsed arguments to the appropriate application controller.

    Maps object-verb combinations to controller methods and handles nested
    commands (e.g., 'land llm validate'). Validates that the requested
    object and action exist before calling.

    Args:
        args: argparse.Namespace containing parsed command-line arguments.
            Must include 'object' and 'verb' attributes at minimum.

    Returns:
        The return value from the called controller method, typically None
        for side-effect operations like database updates or exports.

    Raises:
        ValueError: If the specified object is not recognized or if a nested
            command is missing its required subverb.

    Notes:
        The controller mapping supports both flat commands (e.g., 'land list')
        and nested commands (e.g., 'land llm validate'). Nested commands
        require a subverb argument to identify the specific action.
    """
    controllers = {
        'db': {
            'setup': DbController.setup,
            'migrate': DbController.migrate
        },
        'domain': {
            'crawl': DomainController.crawl
        },
        'land': {
            'list':     LandController.list,
            'create':   LandController.create,
            'delete':   LandController.delete,
            'crawl':    LandController.crawl,
            'readable': LandController.readable,
            'export':   LandController.export,
            'addterm':  LandController.addterm,
            'addurl':   LandController.addurl,
            'urlist':   LandController.urlist,
            'consolidate': LandController.consolidate,
            'medianalyse': LandController.medianalyse,
            'seorank':  LandController.seorank,
            # Nested commands for LLM features
            'llm': {
                'validate': LandController.llm_validate,
            },
        },
        'tag': {
            'export': TagController.export,
        },
        'embedding': {
            'generate': EmbeddingController.generate,
            'similarity': EmbeddingController.similarity,
            'check': EmbeddingController.check,
            'reset': EmbeddingController.reset,
        },
        'heuristic': {
            'update': HeuristicController.update
        }
    }
    controller = controllers.get(args.object)
    if controller:
        action = controller.get(args.verb)
        # Support nested verbs: e.g. controllers['land']['llm']['validate']
        if isinstance(action, dict):
            subverb = getattr(args, 'subverb', None)
            if not subverb:
                raise ValueError("Missing sub-verb for nested command (e.g. 'land llm validate')")
            return call(action.get(subverb), args)
        return call(action, args)
    raise ValueError("Invalid object {}".format(args.object))


def call(func, args):
    """Execute a controller function with the provided arguments.

    Validates that the function is callable before execution and provides
    informative error messages if the action is invalid.

    Args:
        func: The controller method to execute. Must be a callable object.
        args: argparse.Namespace containing command arguments to pass to
            the controller method.

    Returns:
        The return value from the executed controller method.

    Raises:
        ValueError: If func is not callable, includes the attempted verb
            and object in the error message.

    Notes:
        This function serves as a safety wrapper around controller method
        calls, ensuring that only valid callable objects are invoked and
        providing clear error messages for debugging.
    """
    if callable(func):
        return func(args)
    raise ValueError("Invalid action call {} on object {}".format(args.verb, args.object))
