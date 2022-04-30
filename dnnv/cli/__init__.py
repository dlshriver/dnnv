"""
"""
import argparse
import importlib
import pkgutil
from pathlib import Path

from .. import __version__
from .. import logging_utils as logging
from .. import verifiers


class HelpFormatter(argparse.HelpFormatter):
    def _format_action_invocation(self, action):
        if not action.option_strings:
            (metavar,) = self._metavar_formatter(action, action.dest)(1)
            return metavar
        parts = []
        if action.nargs == 0:
            # if the Optional doesn't take a value, format is:
            #    -s, --long
            parts.extend(action.option_strings)
        else:
            # if the Optional takes a value, format is:
            #    -s ARGS, --long ARGS
            # change to
            #    -s, --long ARGS
            default = action.dest.upper()
            args_string = self._format_args(action, default)
            for option_string in action.option_strings:
                parts.append(option_string)
            parts[-1] += f" {args_string}"
        return ", ".join(parts)


class AppendNetwork(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings=option_strings, dest=dest, **kwargs)
        self.networks = {}

    def __call__(self, parser, namespace, values, option_string=None):
        name, network_path_str = values
        if name in self.networks:
            raise parser.error(f"Multiple paths specified for network {name}")
        network_path = Path(network_path_str)
        self.networks[name] = network_path
        items = (getattr(namespace, self.dest) or {}).copy()
        items[name] = network_path
        setattr(namespace, self.dest, items)


class SetVerifierParameter(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        verifier, parameter = option_string.strip("-").split(".", maxsplit=1)
        params = (getattr(namespace, self.dest) or {}).copy()
        if verifier not in params:
            params[verifier] = {}
        params[verifier][parameter] = values
        setattr(namespace, self.dest, params)


def parse_args():
    parser = argparse.ArgumentParser(
        description="dnnv - deep neural network verification",
        prog="dnnv",
        formatter_class=HelpFormatter,
    )
    parser.add_argument("-V", "--version", action="version", version=__version__)
    parser.add_argument("--seed", type=int, default=None, help="the random seed to use")
    logging.add_arguments(parser)

    parser.add_argument("property", type=Path, nargs="?")
    parser.add_argument(
        "-N",
        "--network",
        metavar=("NAME", "PATH"),
        action=AppendNetwork,
        nargs=2,
        dest="networks",
        help="a network to verify",
    )

    parser.add_argument(
        "--save-violation",
        metavar="PATH",
        type=Path,
        default=None,
        help="the path to save a found violation",
    )

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
        importlib.import_module(f"{verifiers.__name__}.{verifier.name}")
    from ..utils import get_subclasses

    for verifier in sorted(
        get_subclasses(verifiers.Verifier),
        key=lambda v: v.__module__.split(".")[-1],
    ):
        if verifier.is_installed():
            vname = verifier.__module__.split(".")[-1]
            if vname == "convert":
                parser.add_argument(
                    f"--{vname}",
                    dest="verifiers",
                    action="append_const",
                    const=verifier,
                )
            else:
                verifier_group.add_argument(
                    f"--{vname}",
                    dest="verifiers",
                    action="append_const",
                    const=verifier,
                )
            verifier_parameters_group = parser.add_argument_group(f"{vname} parameters")
            for pname, pinfo in verifier.parameters.items():
                metavar = None if pinfo.choices is not None else pname.upper()
                verifier_parameters_group.add_argument(
                    f"--{vname}.{pname}",
                    type=pinfo.type,
                    metavar=metavar,
                    action=SetVerifierParameter,
                    choices=pinfo.choices,
                    dest="verifier_parameters",
                    help=pinfo.help,
                )

    parser.set_defaults(networks={})
    parser.set_defaults(verifiers=[])
    parser.set_defaults(verifier_parameters={})

    known_args, extra_args = parser.parse_known_args()
    return known_args, extra_args
