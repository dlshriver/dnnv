"""
"""
import argparse
import logging
import sys


def add_arguments(parser: argparse.ArgumentParser):
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument("--debug", action="store_true", help=argparse.SUPPRESS)
    verbosity_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show messages with finer-grained information",
    )
    verbosity_group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="suppress non-essential messages",
    )


def initialize(name: str, args: argparse.Namespace) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False

    if args.debug:
        logger.setLevel(logging.DEBUG)
    elif args.verbose:
        logger.setLevel(logging.INFO)
    elif args.quiet:
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.WARNING)

    formatter = logging.Formatter("%(levelname)-8s %(asctime)s (%(name)s) %(message)s")

    console_handler = logging.StreamHandler(stream=sys.stderr)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


__all__ = ["add_arguments", "initialize"]
