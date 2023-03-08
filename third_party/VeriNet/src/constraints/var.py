
"""
Simple class for representing variables.

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import numbers

from verinet.constraints.lin_eq import LinEq


class Var:

    """
    A simple variable class, used for storing a name and id.

    Most overloaded operators will return a LinEq object.
    """

    _next_id: int = 0

    def __init__(self, name: str = None):

        """
        Args:
            name:
                The name of the variable.
        """

        self._id: int = Var._next_id
        Var._next_id += 1

        self._name = name

    @staticmethod
    def factory(num):

        """
        Used to create several variables at once.

        Args:
            num:
                The number of variables to be created.
        Returns:
            A list with the variables.
        """

        variables = []

        for i in range(num):
            variables.append(Var())

        return variables

    @property
    def id(self):

        """
        Returns the id of the variable.
        """

        return self._id

    @property
    def name(self):

        """
        The name of the variable.

        If the name is None, 'V'+self._id is returned instead.
        """

        if self._name is None:
            return "V"+str(self._id)
        else:
            return self._name

    @name.setter
    def name(self, name):

        """
        The name of the variable.

        If the name is None, 'V'+self._id is returned instead.
        """

        self._name = name

    def __add__(self, other):

        """
        Addition is implemented by returning the resulting LinEq.

        Args:
            other:
                A Var or a number.
        Returns:
            The resulting LinEq.
        """

        if isinstance(other, numbers.Number):
            # noinspection PyTypeChecker
            return LinEq({self: 1}, constant=other)

        elif isinstance(other, Var):
            if other != self:
                return LinEq({self: 1, other: 1})
            else:
                return LinEq({self: 2})

        elif isinstance(other, LinEq):

            new_eq_dict = other.eq_dict.copy()

            if self in new_eq_dict:
                new_eq_dict[self] += 1
            else:
                new_eq_dict[self] = 1

            return LinEq(new_eq_dict, constant=other.constant)

        else:
            return NotImplemented

    def __radd__(self, other):

        """
        RHS. Version of __add__.
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
            return LinEq({self: other})
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

        """
        Returns the name of the variable.
        """

        return self.name
