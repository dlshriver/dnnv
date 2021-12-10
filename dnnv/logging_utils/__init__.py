"""
"""
import argparse
import logging
import os
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
        "-q", "--quiet", action="store_true", help="suppress non-essential messages"
    )


def initialize(name: str, args: argparse.Namespace) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.propagate = False

    TF_CPP_MIN_LOG_LEVEL = os.environ.get("TF_CPP_MIN_LOG_LEVEL", None)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = TF_CPP_MIN_LOG_LEVEL or "1"
    elif args.verbose:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = TF_CPP_MIN_LOG_LEVEL or "2"
        logger.setLevel(logging.INFO)
    elif args.quiet:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = TF_CPP_MIN_LOG_LEVEL or "2"
        logger.setLevel(logging.ERROR)
    else:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = TF_CPP_MIN_LOG_LEVEL or "2"
        logger.setLevel(logging.WARNING)

    formatter = logging.Formatter(f"%(levelname)-8s %(asctime)s (%(name)s) %(message)s")

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def getLogger(name: str) -> logging.Logger:
    return logging.getLogger(name)


__all__ = ["add_arguments", "getLogger", "initialize"]
