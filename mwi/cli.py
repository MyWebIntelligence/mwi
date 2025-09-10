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
    """
    Run command from args dict or namespace
    :param args:
    :return:
    """
    if isinstance(args, dict):
        args = argparse.Namespace(**args)
    dispatch(args)


def command_input():
    """
    Run command from input
    :return:
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
    args = parser.parse_args()
    # Always convert lang to a list
    if hasattr(args, "lang") and isinstance(args.lang, str):
        args.lang = [l.strip() for l in args.lang.split(",") if l.strip()]
    dispatch(args)


def dispatch(args):
    """
    Disptach command to application controller
    :param args:
    :return:
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
            'consolidate': LandController.consolidate,
            'medianalyse': LandController.medianalyse,
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
        return call(controller.get(args.verb), args)
    raise ValueError("Invalid object {}".format(args.object))


def call(func, args):
    """
    Call application controller
    :param func:
    :param args:
    :return:
    """
    if callable(func):
        return func(args)
    raise ValueError("Invalid action call {} on object {}".format(args.verb, args.object))
