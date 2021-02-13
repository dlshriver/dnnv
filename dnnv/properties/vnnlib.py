import numpy as np

from pathlib import Path
from simpleparse.parser import Parser
from typing import Callable, Dict, List, Optional, Set, Tuple

from .base import *
from .context import Context
from .dsl import LimitQuantifiers, parse_cli


# TODO : implement full smt-lib grammar
# currently enough for vnn-lib spec?
grammar = r"""# partial smt-lib 2.6 grammar
script := command*
command := (ws*,"(",ws*,"assert",ws*,term,ws*,")",ws*)
          /(ws*,"(",ws*,"declare-const",ws*,symbol,ws*,sort,ws*,")",ws*)
term := spec_constant/qual_identifier/(ws*,"(",ws*,qual_identifier,ws*,(term,ws*)+,")",ws*)
qual_identifier := identifier/(ws*,"(",ws*,"as",ws*,identifier,ws*,sort,ws*,")",ws*)
sort := identifier/(ws*,"(",ws*,identifier,ws*,(sort,ws*)+,")",ws*)
identifier := symbol/(ws*,"(",ws*,"_",ws*,symbol,ws*,(index,ws*)+,")",ws*)
index := numeral/symbol
spec_constant := decimal/numeral/hexadecimal/binary/string
symbol := (letter/[~!@$%^&*+=<>.?/_-]),(digit/letter/[~!@$%^&*+=<>.?/_-])*
string := '"',('""'/[\t\n\r !#-~])*,'"'
binary := "#b",[01]+
hexadecimal := "#x",[0-9A-Fa-f]+
decimal := numeral,".","0"*,numeral?
numeral := ([1-9],[0-9]*)/"0"
letter := [A-Za-z]
digit := [0-9]
ws := [ \t\r\n]/comment
comment := ";",[\t -~]*,[\r\n]
"""


class VNNLibParseError(Exception):
    pass


ParseTree = Tuple[str, int, int, List["ParseTree"]]


def subtraction_helper(*args):
    if len(args) == 1:
        return Negation(*args)
    elif len(args) == 2:
        return Subtract(*args)
    print(args)
    raise ValueError("Subtraction not implemented for more than 2 args.")


class ExpressionBuilder:
    def __init__(self):
        self.parser = Parser(grammar, "script")
        self.buffer = None
        self.symbols: Dict[str, Symbol] = {}
        self.operators: Dict[str, Callable] = {
            "+": Add,
            "/": Divide,
            ">": GreaterThan,
            ">=": GreaterThanOrEqual,
            "<": LessThan,
            "<=": LessThanOrEqual,
            "*": Multiply,
            "-": subtraction_helper,
        }
        self.assertions: Set[Expression] = set()

    def build(self, script):
        self.buffer = script
        success, children, index = self.parser.parse(script + "\n")
        if not success or index != (len(script) + 1):
            raise VNNLibParseError(f"Parsing failed at index {index}")
        for c in children:
            self.visit(c)
        return ~Exists(Symbol("X"), And(*self.assertions))

    def declare_const(self, name, sort):
        if name in self.symbols:
            raise VNNLibParseError(f"Name already exists in symbol table: {name}")
        if name.startswith("X_"):
            index = tuple(int(i) for i in name.split("_")[1:])
            if len(index) == 1:
                self.symbols[name] = Symbol("X")[
                    FunctionCall(
                        np.unravel_index, (index[0], Network("N").input_shape[0]), {}
                    )
                ]
            else:
                self.symbols[name] = Symbol("X")[index]
        elif name.startswith("Y_"):
            index = tuple(int(i) for i in name.split("_")[1:])
            if len(index) == 1:
                self.symbols[name] = Network("N")(Symbol("X"))[
                    FunctionCall(
                        np.unravel_index, (index[0], Network("N").output_shape[0]), {}
                    )
                ]
            else:
                self.symbols[name] = Network("N")(Symbol("X"))[index]
        else:
            self.symbols[name] = Parameter(name, type=sort)

    def visit(self, tree: ParseTree):
        production = tree[0]
        visitor = getattr(self, f"visit_{production}", self.generic_visit)
        return visitor(tree)

    def generic_visit(self, tree: ParseTree):
        raise NotImplementedError(f"{tree[0]}")

    def visit_subtrees(self, tree: ParseTree):
        subtrees = []
        end_index = tree[2]
        for t in tree[3]:
            t_ = self.visit(t)
            if t_ is not None:
                subtrees.append(t_)
                end_index = tree[2]
            else:
                if end_index == tree[2]:
                    end_index = t[1]
                if t[1] == tree[1]:
                    tree = (tree[0], t[2], tree[2], tree[3])
        tree = (tree[0], tree[1], end_index, subtrees)
        return tree

    def visit_command(self, tree: ParseTree):
        tree = self.visit_subtrees(tree)

        command = self.buffer[tree[1] : tree[2]].strip("()").split(" ", maxsplit=1)[0]
        assert command in ["assert", "declare-const"]

        if command == "declare-const":
            assert len(tree[3]) == 2
            self.declare_const(*tree[3])
        elif command == "assert":
            assert len(tree[3]) == 1
            self.assertions.add(tree[3][0])
        return tree[3]

    def visit_decimal(self, tree: ParseTree):
        num = float(self.buffer[tree[1] : tree[2]])
        return Constant(num)

    def visit_identifier(self, tree: ParseTree):
        tree = self.visit_subtrees(tree)
        assert len(tree[3]) == 1
        return tree[3][0]

    def visit_numeral(self, tree: ParseTree):
        num = int(self.buffer[tree[1] : tree[2]])
        return Constant(num)

    def visit_qual_identifier(self, tree: ParseTree):
        tree = self.visit_subtrees(tree)
        assert len(tree[3]) == 1  # TODO: not always true
        if tree[3][0] in self.symbols:
            return self.symbols[tree[3][0]]
        elif tree[3][0] in self.operators:
            return self.operators[tree[3][0]]
        else:
            raise VNNLibParseError(f"Unknown identifier: {tree[3][0]}")

    def visit_sort(self, tree: ParseTree):
        tree = self.visit_subtrees(tree)
        assert len(tree[3]) == 1  # TODO: not always true
        if tree[3][0] == "Real":
            return float
        elif tree[3][0] == "Int":
            return int
        elif tree[3][0] == "Bool":
            return bool
        else:
            raise NotImplementedError(
                f"Unimplemented sort: {self.buffer[tree[1]:tree[2]]}"
            )

    def visit_spec_constant(self, tree: ParseTree):
        tree = self.visit_subtrees(tree)
        assert len(tree[3]) == 1
        return tree[3][0]

    def visit_symbol(self, tree: ParseTree):
        name = self.buffer[tree[1] : tree[2]]
        return name

    def visit_term(self, tree: ParseTree):
        tree = self.visit_subtrees(tree)
        if len(tree[3]) == 1:
            return tree[3][0]
        return tree[3][0](*tree[3][1:])

    def visit_ws(self, tree: ParseTree):
        return None


def parse(path: Path, args: Optional[List[str]] = None) -> Expression:
    with open(path, "r") as f:
        script_str = f.read()
    with Context():
        phi = ExpressionBuilder().build(script_str)
        LimitQuantifiers()(phi)

        phi = phi.propagate_constants()
        parse_cli(phi, args)
    return phi
