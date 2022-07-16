from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog


class Variable:
    _count = 0

    def __init__(self, shape: Sequence[int], name: Optional[str] = None):
        self.shape = shape
        self.name = name
        if self.name is None:
            self.name = f"x_{Variable._count}"
        Variable._count += 1

    def size(self) -> int:
        return int(np.product(self.shape))

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Variable({self.shape}, {self.name!r})"

    def __hash__(self):
        return hash(self.shape) * hash(self.name)

    def __eq__(self, other):
        if not isinstance(other, Variable):
            return False
        return (self.name == other.name) and (self.shape == other.shape)


class Constraint(ABC):
    def __init__(self, variable: Optional[Variable] = None):
        self.variables: Dict[Variable, int] = {}
        if variable is not None:
            self.add_variable(variable)

    @property
    def is_consistent(self) -> Optional[bool]:
        return None

    @property
    def num_variables(self) -> int:
        return len(self.variables)

    def size(self) -> int:
        return sum(variable.size() for variable in self.variables)

    def add_variable(self, variable: Variable) -> Constraint:
        if variable not in self.variables:
            self.variables[variable] = self.size()
        return self

    def unravel_index(self, index: int) -> Tuple[Variable, Sequence[np.intp]]:
        c_size = self.size()
        for variable, size in sorted(self.variables.items(), key=lambda kv: -kv[1]):
            if size <= index < c_size:
                return variable, np.unravel_index(index - size, variable.shape)
        raise ValueError(
            f"index {index} is out of bounds for constraint with size {c_size}"
        )

    @abstractmethod
    def as_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def update_constraint(
        self,
        variables: Sequence[Variable],
        indices: Sequence[Sequence[int]],
        coefficients: Sequence[float],
        b: float,
        is_open=False,
    ) -> None:
        pass

    @abstractmethod
    def validate(self, *x: np.ndarray) -> bool:
        pass


@dataclass
class Halfspace:
    indices: Sequence[int]
    coefficients: Sequence[float]
    b: float
    is_open: bool


class HalfspacePolytope(Constraint):
    def __init__(self, variable=None):
        super().__init__(variable)
        self.halfspaces: List[Halfspace] = []
        self._A: List[np.ndarray] = []
        self._b: List[np.ndarray] = []
        self._A_mat: Optional[np.ndarray] = None
        self._b_vec: Optional[np.ndarray] = None
        self._lower_bound = np.full(self.size(), -np.inf)
        self._upper_bound = np.full(self.size(), np.inf)

    @property
    def A(self) -> np.ndarray:
        if self._A_mat is None:
            if self._A:
                self._A_mat = np.vstack(self._A)
            else:
                return np.empty((0, self.size()))
        return self._A_mat

    @property
    def b(self) -> np.ndarray:
        if self._b_vec is None:
            if self._b:
                self._b_vec = np.hstack(self._b)
            else:
                return np.empty((0,))
        return self._b_vec

    def as_matrix_inequality(
        self, tol: Optional[float] = None, include_bounds=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert tol is None
        A = self.A
        b = self.b
        if include_bounds:
            n = self.size()
            A_ = np.vstack([A, -np.eye(n), np.eye(n)])
            b_ = np.hstack([b, -self._lower_bound, self._upper_bound])
            is_finite = ~np.isposinf(b_)
            A_ = A_[is_finite]
            b_ = b_[is_finite]
            return A_, b_
        return A, b

    def as_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._lower_bound, self._upper_bound

    @property
    def is_consistent(self):
        A, b = self.as_matrix_inequality()
        obj = np.zeros(A.shape[1])
        lb, ub = self.as_bounds()
        if A.size == 0:
            return np.all(lb <= ub)
        # linprog breaks if bounds are too big or too small
        bounds = list(
            zip(
                (b if b > -1e6 else None for b in lb),
                (b if b < 1e6 else None for b in ub),
            )
        )
        try:
            result = linprog(
                obj,
                A_ub=A,
                b_ub=b,
                bounds=bounds,
                method="highs",
            )
        except ValueError as e:
            if "The problem is (trivially) infeasible" in e.args[0]:
                return False
            raise e
        if result.status == 4:
            return None
        if result.status == 2:  # infeasible
            return False
        if result.status == 0:  # feasible
            return True
        return None  # unknown

    def add_variable(self, variable: Variable) -> HalfspacePolytope:
        prev_size = self.size()
        super().add_variable(variable)
        new_size = self.size()

        new_lb = np.full(new_size, -np.inf)
        new_ub = np.full(new_size, np.inf)

        if prev_size > 0:
            new_lb[:prev_size] = self._lower_bound
            new_ub[:prev_size] = self._upper_bound

            size_diff = new_size - prev_size
            self._A = [
                np.hstack([a, np.zeros((1, size_diff), dtype=a.dtype)]) for a in self._A
            ]

        self._lower_bound = new_lb
        self._upper_bound = new_ub

        return self

    def _update_bounds(
        self,
        indices: Sequence[int],
        coefficients: Sequence[float],
        b: float,
        is_open=False,
    ) -> None:
        if len(indices) == 1:
            index = indices[0]
            coef_sign = np.sign(coefficients[0])
            value = b / coefficients[0]
            if coef_sign < 0:
                if is_open:
                    value = np.nextafter(value, value + 1)
                self._lower_bound[index] = max(value, self._lower_bound[index])
            elif coef_sign > 0:
                if is_open:
                    value = np.nextafter(value, value - 1)
                self._upper_bound[index] = min(value, self._upper_bound[index])
        else:
            n = self.size()
            for i in indices:
                obj = np.zeros(n)
                bounds = list(
                    zip(
                        (b if np.isfinite(b) else None for b in self._lower_bound),
                        (b if np.isfinite(b) else None for b in self._upper_bound),
                    )
                )
                try:
                    obj[i] = 1
                    result = linprog(
                        obj,
                        A_ub=self.A,
                        b_ub=self.b,
                        bounds=bounds,
                        method="highs",
                    )
                    if result.status == 0:
                        self._lower_bound[i] = max(result.x[i], self._lower_bound[i])
                except ValueError as e:
                    if e.args[0] != (
                        "The algorithm terminated successfully and determined"
                        " that the problem is infeasible."
                    ):
                        raise e
                try:
                    obj[i] = -1
                    result = linprog(
                        obj,
                        A_ub=self.A,
                        b_ub=self.b,
                        bounds=bounds,
                        method="highs",
                    )
                    if result.status == 0:
                        self._upper_bound[i] = min(result.x[i], self._upper_bound[i])
                except ValueError as e:
                    if e.args[0] != (
                        "The algorithm terminated successfully and determined"
                        " that the problem is infeasible."
                    ):
                        raise e

    def update_constraint(
        self,
        variables: Sequence[Variable],
        indices: Sequence[Sequence[int]],
        coefficients: Sequence[float],
        b: float,
        is_open=False,
    ) -> None:
        flat_indices = [
            self.variables[var] + int(np.ravel_multi_index(idx, var.shape))
            for var, idx in zip(variables, indices)
        ]
        halfspace = Halfspace(flat_indices, coefficients, b, is_open)
        self.halfspaces.append(halfspace)

        if len(indices) > 1:
            n = self.size()
            _A = np.zeros((1, n))
            _b = np.zeros((1,))
            for i, a in zip(flat_indices, coefficients):
                _A[0, i] = a
            _b[0] = b
            if is_open:
                _b[0] = np.nextafter(b, b - 1)
            self._A.append(_A)
            self._b.append(_b)
            self._A_mat = None
            self._b_vec = None

        self._update_bounds(flat_indices, coefficients, b, is_open=is_open)

    def validate(self, *x: np.ndarray, threshold: float = 1e-6) -> bool:
        if self.size() == 0:
            return True
        if len(x) != len(self.variables):
            return False
        x_flat_ = []
        for x_, v in zip(x, self.variables):
            if x_.size != v.size():
                return False
            x_flat_.append(x_.flatten())
        x_flat = np.concatenate(x_flat_)
        cast = np.cast[x_flat.dtype]
        for hs in self.halfspaces:
            t = sum(cast(c) * x_flat[i] for c, i in zip(hs.coefficients, hs.indices))
            b = cast(hs.b)
            if hs.is_open:
                b = np.nextafter(b, b - 1)
            if (t - b) > threshold:
                return False
        return True

    def __str__(self):
        strs = []
        for hs in self.halfspaces:
            lhs_strs = []
            for i, c in zip(hs.indices, hs.coefficients):
                variable, index = self.unravel_index(i)
                lhs_strs.append(f"{c} * {variable}[{index}]")
            if hs.is_open:
                strs.append(" + ".join(lhs_strs) + f" < {hs.b}")
            else:
                strs.append(" + ".join(lhs_strs) + f" <= {hs.b}")
        return "\n".join(strs)


class HyperRectangle(HalfspacePolytope):
    @property
    def is_consistent(self) -> bool:
        if (self._lower_bound > self._upper_bound).any():
            return False
        return True

    @property
    def lower_bounds(self) -> Sequence[np.ndarray]:
        lbs = []
        for variable, start_index in self.variables.items():
            size = variable.size()
            lbs.append(
                self._lower_bound[start_index : start_index + size].reshape(
                    variable.shape
                )
            )
        return lbs

    @property
    def upper_bounds(self) -> Sequence[np.ndarray]:
        ubs = []
        for variable, start_index in self.variables.items():
            size = variable.size()
            ubs.append(
                self._upper_bound[start_index : start_index + size].reshape(
                    variable.shape
                )
            )
        return ubs

    def update_constraint(self, variables, indices, coefficients, b, is_open=False):
        if len(indices) > 1:
            raise ValueError(
                "HyperRectangle constraints can only constrain a single dimension"
            )
        super().update_constraint(variables, indices, coefficients, b, is_open)

    def __str__(self):
        strs = []
        for i in range(self.size()):
            lb = self._lower_bound[i]
            ub = self._upper_bound[i]
            variable, index = self.unravel_index(i)
            strs.append(f"{lb:f} <= {variable}[{index}] <= {ub:f}")
        return "\n".join(strs)


__all__ = [
    "Constraint",
    "Halfspace",
    "HalfspacePolytope",
    "HyperRectangle",
    "Variable",
]
