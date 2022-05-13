Function Support
================

DNNP supports the application of any python function to concrete variables
and values. These are values which are known to DNNV at runtime. This includes
things like: 
literal values such as 1 or 127.5, any imported functions or classes, any objects
built from other concrete values, and any Parameter declared in the specification.

Values that are not concrete are symbolic. The most common instance of a symbolic
value is the free variable of a quantifier, such as the first argument to ``Forall``
in a DNNP specification. Any expression that is built from symbolic expressions, or 
a mix of symbolic and concrete expressions is a symbolic expression. Usually this
will include a symbolic variable for the neural network input, and a symbolic 
expression representing the neural network output (or intermediate outputs).
Because symbolic expressions do not have a concrete value, it is not possible to
apply just any function to them since there is no actual value on which to apply
the function and get a result.

Python Support
--------------

abs, len, max, min, sum


Numpy Support
-------------

We partially support the following functions, however not all parameter options are
currently supported.

abs, argmax, argmin, max, mean min, shape, sum

