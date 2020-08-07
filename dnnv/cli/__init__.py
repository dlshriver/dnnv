"""
"""
import argparse
import importlib
import pkgutil

from pathlib import Path

from .. import verifiers
from .. import __version__
from .. import logging


class HelpFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            (metavar,) = self._metavar_formatter(action, action.dest)(1)
            return metavar
        else:
            parts = []
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            if action.nargs == 0:
                parts.extend(action.option_strings)

            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            # change to
            #    -s, --long ARGS
            else:
                default = action.dest.upper()
                args_string = self._format_args(action, default)
                for option_string in action.option_strings:
                    parts.append("%s" % option_string)
                parts[-1] += " %s" % args_string
            return ", ".join(parts)


class AppendVerifier(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        const=None,
        default=None,
        required=False,
        help=None,
        metavar=None,
    ):
        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=0,
            const=const,
            default=default,
            required=required,
            help=help,
            metavar=metavar,
        )
        self.verifiers = {}

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.verifiers:
            return
        verifier = importlib.import_module(f"{verifiers.__name__}.{option_string[2:]}")
        self.verifiers[option_string] = verifier
        items = (getattr(namespace, self.dest) or [])[:]
        items.append(verifier)
        setattr(namespace, self.dest, items)


def parse_args():
    parser = argparse.ArgumentParser(
        description="dnnv - deep neural network verification",
        prog="dnnv",
        formatter_class=HelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=__version__)
    parser.add_argument("--seed", type=int, default=None, help="the random seed to use")
    logging.add_arguments(parser)

    parser.add_argument("network", type=Path)
    parser.add_argument("property", type=Path)

    prop_format_group = parser.add_mutually_exclusive_group()
    prop_format_group.add_argument(
        "--vnnlib",
        action="store_const",
        const="vnnlib",
        dest="prop_format",
        help="use the vnnlib property format",
    )

    verifier_group = parser.add_argument_group("verifiers")
    for verifier in pkgutil.iter_modules(verifiers.__path__):
        if verifier.name == "common":
            continue
        verifier_group.add_argument(
            f"--{verifier.name}", dest="verifiers", action=AppendVerifier
        )
    parser.set_defaults(verifiers=[])

    known_args, extra_args = parser.parse_known_args()
    return known_args, extra_args
