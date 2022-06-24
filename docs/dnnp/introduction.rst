Introduction to DNNP
====================

A property specification defines the desired behavior of a
DNN in a formal language. DNNV uses a custom Python-embedded DSL 
for writing property specifications, which we call DNNP. 
In this section we will go over this language in
detail and describe how properties can be specified in DNNP.
To see some examples of common properties specified in this
language, check :doc:`here <../getting_started/examples>`.

Because DNNP extends from Python, it should support
execution of arbitrary Python code. However, DNNV is still
a work-in-progress, so some expressions (such as star expressions)
are not yet supported by our property parser. We are still working to
fully support all Python expressions, but the current version
supports the most common use cases, and can handle all of the DNN
properties that we have tried.

General Structure
-----------------

The general structure of a property specification is as follows:

    1. A set of python module imports
    2. A set of variable definitions
    3. A property formula

Imports
^^^^^^^

Imports have the same syntax as Python import statements, and
they can be used to import arbitrary Python modules and packages.
This allows re-use of datasets or input pre-processing code.
For example, the Python package ``numpy`` can be imported to
load a dataset.
Inputs can then be selected from the dataset, or statistics, such
as the mean data point, can be computed on the fly.

In general, imported functions and modules can only operate on 
concrete values, however we do offer 
:doc:`limited support <../dnnp/function_support>` for the
application of some builtin and numpy functions to symbolic values.


Definitions
^^^^^^^^^^^

After any imports, DNNP allows a sequence of assignments to define
variables that can be used in the final property specification.
For example, ``i = 0``, will define the variable ``i`` to a
value of 0.

These definitions can be used to load data and configuration parameters, 
or to alias expressions that may be used in the property formula.
For example, if the ``torchvision.datasets`` package has been imported,
then ``data = datasets.MNIST("/tmp")`` will define a variable ``data``
referencing the MNIST dataset from this package.
Additionally, the :py:class:`Parameter` class can be used to declare
parameters that can be specified at run time. For example, 
``eps = Parameter("epsilon", type=float)``, will define the variable 
``eps`` to have type float and will expect a value to be specified at 
run time. This value can be specified to DNNV with the option 
``--prop.epsilon``.

Definitions can also assign expressions to variables to be used in the
property specification later.
For example, ``x_in_unit_hyper_cube = 0 <= x <= 1`` can be used to assign
an expression specifying that the variable ``x`` is within the unit hyper cube
to a variable. This could be useful for more complex properties with a lot
of redundant sub expressions.

A network can be defined using the :py:class:`Network` class.
For example, ``N = Network("N")``, specifies a network with the name N
(which is used at run time to concretize the network with a specific DNN model).
All networks with the same name refer to the same model.


Property Formula
^^^^^^^^^^^^^^^^

Finally, the last part of the property specification is the property
formula itself. It must appear at the end of the property specification.
All statements before the property formula must be either import or
assignment statements.

The property formula defines the desired behavior of the DNN in a
subset of first-order-logic. It can make use of arbitrary Python
code, as well as any of the expressions defined before it.

DNNP provides many functions to define expressions.
The function :py:class:`Forall(symbol, expression)` can be used to specify that the
provided expression is valid for all values of the specified symbol.
The function :py:class:`And(*expression)`, specifies that all of the expressions
passed as arguments to the function must be valid. ``And(expr1, expr2)`` can be
equivalently specified as ``expr1 & expr2``.
The function :py:class:`Or(*expression)`, specifies that at least one of 
the expressions passed as arguments to the function must be valid. 
``Or(expr1, expr2)`` can be equivalently specified as ``expr1 | expr2``.
The function :py:class:`Implies(expression1, expression2)`, specifies that
if ``expression1`` is true, then ``expression2`` must also be true.
The :py:func:`argmin` and :py:func:`argmax` functions
can be used to get the argmin or argmax value of a network's output,
respectively.

In property expressions, networks can be called like functions to get
the outputs for the network for a given input. Networks can be applied to
symbolic variables (such as universally quantified variables), as well as
numpy arrays.

*Currently DNNV only supports universally quantified properties over a single
network input variable. Support for more complex properties is planned.*


.. Property Structures
.. -------------------

.. **TODO** This section needs a better title (and content).
.. The plan is to discuss our extensions that make specifying
.. properties easier (e.g., symbols, first order logic
.. implementation, etc.), and how to use them.

.. **TODO** Should mention that network inputs should be one of
.. our builtin types or a numpy array. For instance, if loading
.. data from a PyTorch DataLoader, the resulting Tensor must be
.. converted to a numpy array before being passed into the network.

.. **TODO** Explain symbols. Variables don't need to be declared before
.. use. Any variable that is used without being defined will be considered
.. symbolic. Currently, there is no way to provide a concrete value to
.. symbolic variables from the command line interface. In general, the
.. current version of the tool supports at most 1 symbolic variable per
.. property, and it must be the input to a network, and have a defined
.. lower and upper bound.
