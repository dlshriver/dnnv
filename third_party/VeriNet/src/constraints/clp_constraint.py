
"""
Simple class for representing linear A simple class for representing a conjunction
of linear constraints.

The class is designed to be easy to use and maintain, and is not optimised for speed.
Since the constraints are usually defined on the neural networks outputs,
they should be low-dimensional and low performance shouldn't be an issue here.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import numpy as np

from verinet.constraints.lin_eq import LinEq


class CLPConstraint:

    """
    A simple class for representing a conjunction of linear constraints.

    Not designed to be called directly, instead use overloaded operators on Var objects
    (+, -, *, /, <=, >=) to create constraints. Also note that conjunction of two
    constraints, constr1 and constr2 can be created using the overloaded | operator.
    All constraints stored here are taken to be Less-Equal constraints and the LinEq
    is the LHS.

    Example usage:
        y1 = Var()
        y2 = Var()
        constraint = y1 + 2 * y2 <= 0 | y1 <= 0 | y2 <= 0
    """

    _next_id: int = 0

    def __init__(self, lin_eq: LinEq = None, name: str = None):

        """
        Args:
            lin_eq:
                The left-hand side of the constraint (rhs. is taken to be 0).
            name:
                The name of the constraint.
        """

        if lin_eq is None:
            self._lin_eqs = []
        else:
            self._lin_eqs = [lin_eq]
        self._name = name

        self._id = self._next_id
        self._next_id += 1

    @property
    def name(self):

        """
        The name of the variable.

        If the name is None, 'C'+self._id is returned instead.
        """

        if self._name is None:
            return "C"+str(self._id)
        else:
            return self._name

    @property
    def lin_eqs(self):
        return self._lin_eqs

    def as_arrays(self, start_idx, end_idx):

        """
        Provides the constraints as a list of arrays.

        Notice that the constraint has to be a disjunction of linear constraints. Each
        of the linear constraints in the disjunction is represented by one of the
        arrays in the list.

        Each of the arrays is of length end_idx-start_idx + 1, and the id of each
        variable (variable.id) in the constraint should be between start_idx and
        end_idx.

        The coefficient of variable with 'id' will be stored in array[id - start_idx].
        The last element of the array is the constant term.

        Example:

            Assume that the constraint has variables y1, y2 with y1.id=0 and y2.id=1,
            and that the function is called with start_idx - end_idx.

            The constraints:
                y_1 + y_2 <= 0 | 2*y_2 - 1 <= 0 | y1 <= 0

            will be return as:

                [np.array(0, 0, 0), np.array(0, 2, -1), np.array(1, 0, 0)]

        Args:
            start_idx:
                The index of the first variable
            end_idx:
                The index of the last variable
        """

        constr_arrays = []

        for lin_eq in self._lin_eqs:
            constr_array = np.zeros(end_idx - start_idx + 2)
            for var, coeff in lin_eq.eq_dict.items():
                constr_array[var.id - start_idx] = coeff

            constr_array[-1] = lin_eq.constant

            constr_arrays.append(constr_array)

        return constr_arrays

    def __or__(self, other):

        """
        Adds the lin_eqs from other to self.lin_eqs
        """

        for lin_eq in other.lin_eqs:
            self._lin_eqs.append(lin_eq)
        return self

    def __str__(self):

        constr_str = ""

        for i, lin_eq in enumerate(self._lin_eqs):
            constr_str += str(lin_eq) + " <= 0"
            if i < len(self._lin_eqs) - 1:
                constr_str += " | \n"

        return constr_str
