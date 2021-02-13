.. DNNV documentation master file, created by
   sphinx-quickstart on Fri Dec 13 15:52:14 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DNNV
====

DNNV is a framework for verifying deep neural networks (DNN).
DNN verification takes in a neural network, and a property over
that network, and checks whether the property is true, or false.
One common DNN property is local robustness, which specifies that
inputs near a given input, will be classified similarly to that
given input.

DNNV standardizes the network and property input formats to enable
multiple verification tools to run on a single network and property.
This facilitates both verifier comparison, and artifact re-use.

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation
   getting_started/basic_usage
   getting_started/examples

.. toctree::
   :maxdepth: 1
   :caption: Usage

   usage/specifying_properties

.. toctree::
   :maxdepth: 1
   :caption: Advanced

   advanced/property_reduction
   advanced/dnn_simplification

.. toctree::
   :maxdepth: 1
   :caption: Developers

   developers/new_verifier

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
