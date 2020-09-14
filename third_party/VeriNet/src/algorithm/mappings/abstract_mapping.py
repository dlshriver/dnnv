
"""
This file contains abstractions for the network mappings. Note that both layers (FC, Conv, ...) and activation
functions (ReLU, Sigmoid, ...) are considered network mappings.

The abstractions are used to calculate linear relaxations, function values, and derivatives

Author: Patrick Henriksen <patrick@henriksen.as>
"""

import numpy as np


class AbstractMapping:

    """
    Abstract class for describing network mappings.

    This class should be subclassed to add support for layers (FC, Conv, ...) and activation functions
    (ReLU, Sigmoid, ...).

    The main purpose of this class is to provide a standardised implementation to calculate function values, derivatives
    and linear relaxations for all supported network mappings.
    """

    def __init__(self):
        self.params = {}

    @property
    def is_linear(self) -> bool:
        raise NotImplementedError(f"is_linear(...) not implemented in {self.__name__}")

    @property
    def is_1d_to_1d(self) -> bool:
        raise NotImplementedError(f"is_linear(...) not implemented in {self.__name__}")

    @property
    def required_params(self) -> list:
        return []

    @classmethod
    def get_subclasses(cls):

        """
        Returns a generator yielding all subclasses of this class.

        This is intended to only be used in get_activation_mapping_dict().
        """

        # Subclasses in not-imported files are not found

        # noinspection PyUnresolvedReferences
        import src.algorithm.mappings.s_shaped
        # noinspection PyUnresolvedReferences
        import src.algorithm.mappings.piecewise_linear
        # noinspection PyUnresolvedReferences
        import src.algorithm.mappings.layers

        for subclass in cls.__subclasses__():
            yield from subclass.get_subclasses()
            yield subclass

    @staticmethod
    def get_activation_mapping_dict():

        """
        Returns a dictionary mapping the torch activation functions and layers to the relevant subclasses.

        This is achieved by looping through all subclasses and calling abstracted_torch_funcs(), which simplifies
        adding new activation functions in the future.
        """

        activation_map = {}

        for subcls in AbstractMapping.get_subclasses():
            for torch_func in subcls.abstracted_torch_funcs():

                activation_map[torch_func] = subcls

        return activation_map

    @classmethod
    def abstracted_torch_funcs(cls) -> list:

        """
        Defines the torch activation functions abstracted by this class. For example, a ReLU abstraction might have:
        [nn.modules.activation.ReLU, nn.ReLU]

        Returns:
            A list with all torch functions that are abstracted by the current subclass.
        """

        raise NotImplementedError(f"abstracted_torch_funcs(...) not implemented in {cls.__name__}")

    def propagate(self, x: np.array, add_bias: bool = True) -> np.array:

        """
        Propagates trough the mapping (by applying the activation function or layer-operation).

        Args:
            x           : The input as a np.array
            add_bias    : Adds bias if relevant, for example for FC and Conv layers.
        Returns:
            The value of the activation function at x
        """

        raise NotImplementedError(f"propagate(...) not implemented in {self.__name__}")

    def split_point(self, xl: float, xu: float) -> float:

        """
        Calculates the preferred split point for branching.

        Args:
            xl  : The lower bound on the input
            xu  : The upper bound on the input

        Returns:
            The preferred split point
        """

        raise NotImplementedError(f"split_point(...) not implemented in {self.__name__}")

    def linear_relaxation(self, lower_bounds_concrete_in: np.array, upper_bounds_concrete_in: np.array,
                          upper: bool) -> np.array:

        """
        Calculates the linear relaxation

        The linear relaxation is a Nx2 array, where each row represents a and b of the linear equation:
        l(x) = ax + b

        Args:
            lower_bounds_concrete_in    : The concrete lower bounds of the input to the nodes
            upper_bounds_concrete_in    : The concrete upper bounds of the input to the nodes
            upper                       : If true, the upper relaxation is calculated, else the lower

        Returns:
            The relaxations as a Nx2 array
        """

        raise NotImplementedError(f"linear_relaxation(...) not implemented in {self.__name__}")

    def _linear_relaxation_equal_bounds(self,
                                        lower_bounds_concrete_in: np.array,
                                        upper_bounds_concrete_in: np.array,
                                        relaxations: np.array):

        """
        Calculates the linear relaxations for the nodes with equal lower and upper bounds.

        Added for convenience; this function is intended to be called from linear_relaxation in subclasses.

        The linear relaxations used when both bounds are equal is l=ax + b with a = 0 and b equal the activation at
        the bounds. The relaxations array is modified in place and is assumed to be initialized to 0.

        Args:
            lower_bounds_concrete_in    : The concrete lower bounds of the input to the nodes
            upper_bounds_concrete_in    : The concrete upper bounds of the input to the nodes
            relaxations                 : A Nx2 array with the a,b parameters of the relaxation lines for each node.
        """

        assert self.is_1d_to_1d, "Equal bounds called on a multi-dimensional function"

        equal_idx = np.argwhere(lower_bounds_concrete_in == upper_bounds_concrete_in)
        relaxations[equal_idx, 1] = self.propagate(lower_bounds_concrete_in[equal_idx])

    def out_shape(self, in_shape: np.array) -> np.array:

        """
        Returns the output-shape of the data as seen in the original network.
        """

        return in_shape


class ActivationFunctionAbstractionException(Exception):
    pass
