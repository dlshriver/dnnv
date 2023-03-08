"""
A simple parser for vnnlib files.

OBS:
This parser is work in progress and is makes strong assumptions on the file format
as explained in the class docstring.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import re

import numpy as np

from verinet.verification.objective import Objective
from verinet.neural_networks.verinet_nn import VeriNetNN
from verinet.constraints.clp_constraint import CLPConstraint


class VNNLIBParser:

    """
    A parser for properties in vnnlib format.

    Note that this parser makes strong assumption on the format of the vnnlib file.

    In particular one of the following may be true:

    1)  The file does not contain any 'or' or 'and' statements.
    2)  The input variables are not inside 'or' or 'and' statements and all output
        variables are inside one 'or' statement with nested 'and' statements.
    3)  The input variables are inside one 'or' statement with nested 'and'
        statements and no output variables are inside an 'or' or 'and' statements.
    4)  All input and output variables are inside one 'or' statement with nested 'and'
        statements. Note that each 'and' statement must define lower and upper bounds
        for all input variables.

    Additionally, the following must be true:

        1) All constraints are linear.
        2) Output constraints can not be mixed with input constraints (e.g. X_1 > Y_1).
        3) All input variables are named X_i, all output variables Y_j.
        4) All input variables have defined lower and upper bounds.
    """

    def __init__(self, vnnlib_path: str):

        self.vnnlib_path = vnnlib_path

        self.num_inputs = 0
        self.num_outputs = 0

        self._global_input_constraints = []
        self._global_output_constraints = []

        self._input_or = []
        self._output_or = []
        self._mixed_or = []

        self._input_var = "X"
        self._output_var = "Y"

        self._traverse_file()

    def _traverse_file(self):

        """
        Traverses the vnnlib file and stores number of variables as well as the
        constraints.
        """

        in_var = self._input_var
        out_var = self._output_var

        with open(self.vnnlib_path, "r") as file:

            while True:

                statement = self._get_next_statement(file)

                if statement is None:
                    return

                elif len(statement) == 3 and statement[0] == "(declare-const" and \
                        statement[1][0] == in_var and statement[2] == "Real)":
                    self.num_inputs += 1

                elif len(statement) == 3 and statement[0] == "(declare-const" and \
                        statement[1][0] == out_var and statement[2] == "Real)":
                    self.num_outputs += 1

                elif statement[0] == "(assert":
                    statement = statement[1:]
                    statement[-1] = statement[-1].strip(')')
                    self._process_statement(statement)

    @staticmethod
    def _get_next_statement(file):

        """
        Gets the next statement enclosed in brackets. Newlines are striped and multiple
        spaces are converted to a single one.

        file:
            The vnnlib file.
        """

        for line in file:

            line, _, _ = line.partition(';')
            if len(line) == 0:
                continue

            line = re.sub(' +', ' ', line.strip("\n"))
            num_open, num_close = line.count('('), line.count(')')

            if num_open == 0:
                continue

            while num_close < num_open:

                new_line = re.sub(' +', ' ', file.readline().strip("\n"))
                num_open += new_line.count('(')
                num_close += new_line.count(')')
                line += new_line

            re.sub(' +', ' ', line)
            line = line.replace('\t', ' ')
            line = line.replace(')(', ') (')
            line = line.split(' ')

            return line

        return None

    def _process_statement(self, statement: list):

        """
        Process the given statement.

        Args:
            statement:
                The statement as a list with elements and variables as elements.
        """

        if statement[0] in ['(<=', '(>=']:

            in_constraints, out_constraints = self._process_simple_constraint(statement)
            self._global_input_constraints += in_constraints
            self._global_output_constraints += out_constraints

        elif statement[0] == '(and':

            statement = statement[1:]
            statement[-1] = statement[-1].strip(')')

            in_constraints, out_constraints = self._process_and_statement(statement)
            self._global_input_constraints += in_constraints
            self._global_output_constraints += out_constraints

        elif statement[0] == '(or':

            statement = statement[1:]
            statement[-1] = statement[-1].strip(')')

            self._process_or_statement(statement)

    def _process_simple_constraint(self, constr: list) -> tuple:

        """
        Processes a simple constraint.

        Args:
            constr:
                A list of size 3 on the form [op, E1, E2] with op = [<=, >=], E1 and E2
                are one input variable and one constant, or one output variable and one
                constant, or two output variables.

        Returns:
            A tuple of lists (list1, list2) where list1 = [constr] and list2 = [] if
            the constraint is an input constraint, otherwise list1 = [] and
            list2 = [constr].
        """

        constr[0], constr[-1] = constr[0].strip('('), constr[-1].strip(')')

        if (constr[1].startswith(self._input_var) and self.isfloat(constr[2])) or \
                (constr[2].startswith(self._input_var) and self.isfloat(constr[1])):

            return [constr], []

        elif (constr[1].startswith(self._output_var) and self.isfloat(constr[2])) or \
                (constr[2].startswith(self._output_var) and self.isfloat(constr[1])) or \
                (constr[1].startswith(self._output_var) and constr[2].startswith(self._output_var)):

            return [], [constr]

        else:
            raise ValueError(f"Expression: {constr} not recognised")

    # noinspection PyTypeChecker
    def _process_and_statement(self, statement: list):

        """
        Processes an and-statement.

        Args:
            statement:
                A list of size 3 where every three elements are on the form
                [op, E1, E2] with op = [<=, >=], E1 and E2 are one input variable and
                one constant, or one output variable and one constant, or two output
                variables.

        Returns:
            A tuple of lists (list1, list2) where list1 consists of input constraints
            and list2 consists of output constraints.
        """

        in_constraints = []
        out_constraints = []

        if len(statement) % 3 != 0:
            raise ValueError(f"And-expression: {statement} not recognised")

        for i in range(len(statement) // 3):

            in_constr, out_constr = self._process_simple_constraint(statement[i * 3: (i + 1) * 3])
            in_constraints += in_constr
            out_constraints += out_constr

        return in_constraints, out_constraints

    def _process_or_statement(self, statement: list):

        """
        Processes an or-statement.

        Args:
            statement:
                A list and-statements.

        Returns:
            A list of (list1, list2) where list1 consists of input constraints
            and list2 consists of output constraints for each clause in the
            or-statement.
        """

        and_indices = []
        for i, elem in enumerate(statement):
            if elem == "(and":
                and_indices.append(i)
        and_indices.append(len(statement))

        this_constr_type = None
        for i in range(len(and_indices)-1):

            this_statement = statement[and_indices[i]:and_indices[i+1]]
            this_statement = this_statement[1:]
            this_statement[-1] = this_statement[-1].strip(')')

            in_constr, out_constr = self._process_and_statement(this_statement)

            if len(in_constr) == 0 and len(out_constr) > 0:

                if this_constr_type is not None and this_constr_type != "in":
                    raise ValueError("Mixed several constraint types in one or-statement.")
                this_constr_type = 'in'

                self._output_or.append(out_constr)

            elif len(in_constr) > 0 and len(out_constr) == 0:

                if this_constr_type is not None and this_constr_type != "out":
                    raise ValueError("Mixed several constraint types in one or-statement.")
                this_constr_type = 'out'

                self._input_or.append(in_constr)

            elif len(in_constr) > 0 and len(out_constr) > 0:

                if this_constr_type is not None and this_constr_type != "mixed":
                    raise ValueError("Mixed several constraint types in one or-statement.")
                this_constr_type = 'mixed'

                self._mixed_or.append([in_constr, out_constr])

            else:
                raise ValueError(f"Could not parse or constraint: {this_statement}")

    def get_objectives_from_vnnlib(self, model: VeriNetNN, input_shape: tuple) -> list:

        """
        Parses the vnnlib file into one or more objectives.

        Args:
            model:
                The VeriNetNN model.
            input_shape:
                The shape of the models input.
        Returns:
            A list of Objectives, if any objective is "unsafe/undecided", the verification
            instances is "unsafe/undecided".
        """

        if (len(self._mixed_or) > 0 and len(self._input_or) == 0 and len(self._output_or) == 0 and
                len(self._global_input_constraints) == 0 and len(self._global_output_constraints) == 0):

            objectives = self._get_objectives_from_mixed_or(model, input_shape)

        elif len(self._input_or) == 0 and len(self._global_input_constraints) > 0:

            input_bounds = self._convert_to_array_input_bounds(self._global_input_constraints)

            if len(self._output_or) == 0 and len(self._global_output_constraints) > 0:

                objectives = [Objective(input_bounds.reshape((*input_shape, 2)), self.num_outputs, model)]
                self._add_output_constraints_to_objective(objectives[0], [self._global_output_constraints])

            elif len(self._output_or) > 0 and len(self._global_output_constraints) == 0:

                objectives = [Objective(input_bounds.reshape((*input_shape, 2)), self.num_outputs, model)]
                self._add_output_constraints_to_objective(objectives[0], self._output_or)

            else:
                raise ValueError("Unexpectedly found both global and local output constraints.")

        elif len(self._input_or) > 0 and len(self._global_input_constraints) == 0:

            objectives = []

            for i, input_constraints in enumerate(self._input_or):

                input_bounds = self._convert_to_array_input_bounds(input_constraints)

                if len(self._output_or) == 0 and len(self._global_output_constraints) > 0:

                    objectives += [Objective(input_bounds.reshape((*input_shape, 2)), self.num_outputs, model)]
                    self._add_output_constraints_to_objective(objectives[i], [self._global_output_constraints])

                elif len(self._output_or) > 0 and len(self._global_output_constraints) == 0:

                    objectives += [Objective(input_bounds.reshape((*input_shape, 2)), self.num_outputs, model)]
                    self._add_output_constraints_to_objective(objectives[i], self._output_or)

                else:
                    raise ValueError("Unexpectedly found both global and local output constraints.")

        else:
            raise ValueError("Unexpectedly found both global and local input constraints.")

        return objectives

    def _get_objectives_from_mixed_or(self, model: VeriNetNN, input_shape: tuple):

        """
        Converts the mixed-or-constraints to verification objectives.

        Args:
            model:
                The VeriNetNN model.
            input_shape:
                The shape of the models input.

        Returns:
            A list of constraints corresponding to the objective.
        """

        or_input_constraints = [constr[0] for constr in self._mixed_or if len(constr[0]) > 0]
        or_output_constraints = [constr[1] for constr in self._mixed_or if len(constr[1]) > 0]

        objectives = []

        for i, input_constraints in enumerate(or_input_constraints):

            input_bounds = self._convert_to_array_input_bounds(input_constraints)

            if len(or_output_constraints) == 0 and len(self._global_output_constraints) > 0:

                objectives += [Objective(input_bounds.reshape((*input_shape, 2)), self.num_outputs, model)]
                self._add_output_constraints_to_objective(objectives[i], [self._global_output_constraints])

            elif len(or_output_constraints) > 0 and len(self._global_output_constraints) == 0:

                objectives += [Objective(input_bounds.reshape((*input_shape, 2)), self.num_outputs, model)]
                self._add_output_constraints_to_objective(objectives[i], [or_output_constraints[i]])

        return objectives

    # noinspection PyUnboundLocalVariable
    def _convert_to_array_input_bounds(self, input_constraints: list) -> np.array:

        """
        Formats the input constraints to lower and upper bounds in an array.

        Returns:
            A Nx2 array where N corresponds to the input variables, the first column
            contains the lower bounds and the second the upper bounds.
        """

        input_bounds = np.zeros((self.num_inputs, 2), dtype=np.float32)
        input_bounds_found = np.zeros((self.num_inputs, 2), dtype=float)

        for op, var1, var2 in input_constraints:

            if not(var1.startswith(self._input_var) or var2.startswith(self._input_var)) or (op not in ['>=', '<=']):
                raise ValueError(f"Input constraint: ({op} {var1} {var2}) not recognised.")

            if var1.startswith(self._input_var):
                index, bound = [int(var1[2:]), 1], float(var2)

            elif var2.startswith(self._input_var):
                index, bound = [int(var2[2:]), 0], float(var1)

            if op == ">=":
                index[1] = abs(index[1] - 1)  # Flip bound.

            input_bounds[index[0], index[1]] = bound
            input_bounds_found[index[0], index[1]] = 1

        if np.sum(input_bounds_found == 0) > 0:
            raise ValueError("Not all input bounds defined in vnnlib file.")

        return input_bounds

    # noinspection PyUnboundLocalVariable
    def _add_output_constraints_to_objective(self, objective: Objective, constraints: list):

        """
        Adds the given output constraints to the objective.

        Args:
            objective:
                The verification objective.
            constraints:
                A nested list of constraints. The first dimension is considered to be
                different 'or' statements for a counterexample, the second dimension
                'and' statements.
        """

        out_vars = objective.output_vars

        for constr_list in constraints:

            constr = CLPConstraint()

            for op, var1, var2 in constr_list:

                if var1.startswith(self._output_var) and self.isfloat(var2):
                    z1, z2 = out_vars[int(var1[2:])], float(var2)

                elif self.isfloat(var1) and var2.startswith(self._output_var):
                    z1, z2 = float(var1), out_vars[int(var2[2:])]

                elif var1.startswith(self._output_var) and var2.startswith(self._output_var):
                    z1, z2 = out_vars[int(var1[2:])], out_vars[int(var2[2:])]

                else:
                    raise ValueError(f"Variables in constraint: ({op} {var1} {var2}) not recognised.")

                if op == "<=":
                    constr = constr | (z1 >= z2)

                elif op == ">=":
                    constr = constr | (z1 <= z2)

                else:
                    raise ValueError(f"Operation in constraint: ({op} {var1} {var2}) not recognised.")

            objective.add_constraints(constr)

    @staticmethod
    def isfloat(value: str) -> bool:

        """
        Returns true if the string can be converted to float.

        Args:
            value:
                The value to be tested as float.
        """

        try:
            float(value)
            return True

        except ValueError:
            return False
