Specifying Custom Properties
============================

A property specification defines the desired behavior of a
DNN in a formal language. DNNV uses a custom DSL for writing
property specifications, based on the Python programming
language. In this section we will go over this language in
detail and describe how properties can be specified in this DSL.
To see some examples of common properties specified in this
language, check :doc:`here <../getting_started/examples>`).

Because the property DSL extends from Python, it should support
execution of arbitrary Python code. However, DNNV is still
of a work-in-progress, so some expressions (such as star expressions)
are not yet supported by our property parser. We are still working to
fully support all Python expressions, but the current version
supports the most common use cases, and can handle all of the DNN
properties that we have tried.

General Structure
-----------------

The general structure of a property specification is as follows:

    1. A set of python module imports
    2. A set of aliasing statements
    3. A property formula

Imports
^^^^^^^

Imports have the same syntax as Python import statements, and
they can be used to import arbitrary Python modules and packages.
This allows re-use of datasets or input pre-processing code.

While not necessary for correctness, we recommend importing
the ``dnnv.properties`` package as ``from dnnv.properties import *``,
which can enable autocompletion and type hints in many code editors.


Aliases
^^^^^^^

After imports, assignment expressions can be used to load data and
configuration parameters, or to alias expressions that may be used
in the property formula.


Property Formula
^^^^^^^^^^^^^^^^

Finally, the last part of the property specification is the property
formula itself. It must appear at the end of the property specification.
All statements before the property formula must be either import or
assignment statements.

The property formula defines the desired behavior of the DNN in a
subset of first-order-logic. It can make use of arbitrary Python
code, as well as any of the expressions aliased before it.


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
