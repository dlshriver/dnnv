
"""
Simple class for representing linear equations.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import numbers


class LinEq:

    """
    Represents a linear equation.

    Arithmetic operators are overloaded to return a new linear equation, while
    comparison operators return a constraint.
    """

    def __init__(self, eq_dict: dict = None, constant: float = 0):

        """
        Args:
            eq_dict:
                The equation dict on the form {Var.id: Coefficient(int)}.
        """

        if eq_dict is None:
            self._eq_dict: dict = {}
        else:
            self._eq_dict = eq_dict

        self._constant = constant

    @property
    def constant(self) -> float:

        """
        Returns:
            The constant term of the equation.
        """

        return self._constant

    @property
    def eq_dict(self) -> dict:

        """
        The equation dict is on the form {Var.id: Coefficient(int)}.

        Returns:
            The equation dict.
        """

        return self._eq_dict

    def __add__(self, other) -> 'LinEq':

        """
        A LinEq object can be added with a Var or another LinEq.

        The addition is done by adding the coefficient(s) corresponding to the Var
        (or Vars in LinEq) to the coefficients in this self._eq_dict.

        Args:
            other:
                The Var or LinEq that should be added.
        returns:
            The new LinEq object
        """

        new_eq_dict = self._eq_dict.copy()
        new_constant = self._constant

        if isinstance(other, numbers.Number):

            new_constant += other

        elif isinstance(other, LinEq):

            for var, coeff in other.eq_dict.items():

                if var in new_eq_dict:
                    new_eq_dict[var] += coeff
                else:
                    new_eq_dict[var] = coeff

            new_constant += other.constant

        else:
            return NotImplemented

        return LinEq(new_eq_dict, new_constant)

    def __radd__(self, other) -> 'LinEq':

        """
        RHS. version of __add__.
        """

        return self.__add__(other)

    def __mul__(self, other) -> 'LinEq':

        """
        Scalar multiplication is implemented by returning the resulting LinEq.

        Args:
            other:
                A number.
        Returns:
            A Var object.
        """

        if isinstance(other, numbers.Number):

            new_eq_dict = {}
            # noinspection PyTypeChecker
            new_constant = self._constant * other

            for var, coeff in self.eq_dict.items():

                new_eq_dict[var] = coeff*other

            return LinEq(new_eq_dict, constant=new_constant)
        else:
            return NotImplemented

    def __rmul__(self, other) -> 'LinEq':

        """
        RHS Version of __mul__.
        """

        return self.__mul__(other)

    def __sub__(self, other):

        """
        Subtraction is implemented by returning the resulting LinEq.

        Args:
            other:
                A Var.
        Returns:
            The resulting LinEq.
        """

        return self + (-1 * other)

    def __rsub__(self, other):

        """
        RHS. Version of __subtract__.
        """

        return (-1 * self) + other

    def __truediv__(self, other):

        """
        Division is implemented by returning the resulting LinEq.

        Args:
            other:
                A Var or LinEq.
        Returns:
            The resulting LinEq.
        """

        return self * (1/other)

    def __le__(self, other):

        """
        Returns the corresponding Constraint object.
        """

        from verinet.constraints.clp_constraint import CLPConstraint

        return CLPConstraint(self - other)

    def __ge__(self, other):

        """
        Returns the corresponding Constraint object.
        """

        from verinet.constraints.clp_constraint import CLPConstraint

        return CLPConstraint(other - self)

    def __str__(self):

        variables = [var for var in list(self._eq_dict.keys())]
        coeffs = [coeff for coeff in list(self._eq_dict.values())]
        sorted_idx = sorted(range(len(variables)), key=lambda j: variables[j].id)

        desc = ""

        for i, idx in enumerate(sorted_idx):

            if i != 0 and coeffs[idx] >= 0:
                desc += " + " + str(abs(coeffs[idx])) + "*" + str(variables[idx])
            elif coeffs[idx] < 0:
                desc += " - " + str(abs(coeffs[idx])) + "*" + str(variables[idx])
            else:
                desc += str(abs(coeffs[idx])) + "*" + str(variables[idx])

        if self._constant > 0:
            desc += " + " + str(self._constant)
        else:
            desc += " - " + str(abs(self._constant))

        return desc
