Property Reduction
==================

A *verification problem* is a pair, :math:`\psi = \langle \mathcal{N}, \phi \rangle`, of a DNN, :math:`\mathcal{N}`, and a property specification :math:`\phi`, formed to determine
whether :math:`\mathcal{N} \models \phi` is *valid or *invalid*.  

Reduction, :math:`reduce: \Psi \rightarrow P(\Psi)`, aims to transform a verification problem, :math:`\langle \mathcal{N}, \phi \rangle = \psi \in \Psi`, to an equivalid 
form, :math:`reduce(\psi) = \{ \langle \mathcal{N}_1, \phi_1 \rangle, \ldots, \langle \mathcal{N}_k, \phi_k \rangle \}`, in which property specifications are in a common supported form.

Reduction enables the application of a broad array of efficient DNN analysis techniques to compute problem validity and/or invalidity.

As defined, reduction has two key properties.
The first property is that the set of resulting problems is equivalid with the original verification problem.

.. math::

    \mathcal{N} \models \psi \Leftrightarrow \forall \langle \mathcal{N}_i, \phi_i \rangle \in reduce(\psi) . \mathcal{N}_i \models \phi_i

The second property  is that the resulting set of problems all use the same property type.
For example, if the desired property type is robustness; all resulting properties assert that 
:math:`\mathcal{N}(x)_0` is the output class for all inputs.
Applying reduction enables verifiers to support a large set of verification problems by implementing support for this single property type.

Overview
--------

To illustrate, consider a property for 
`DroNet <https://github.com/uzh-rpg/rpg_public_dronet>`_; 
a DNN for controlling an autonomous quadrotor.
Inputs to this network are 200 by 200 pixel grayscale images 
with pixel values between 0 and 1.
For each image, DroNet predicts a steering angle and a probability that the drone is about to collide with an object.
The property states that for all inputs, if the probability of collision is no greater than 0.1, then the steering angle is capped at :math:`\pm5` degrees and is specified as:

.. math::
    
    \forall x . ((x \in [0, 1]^{40000}) \land (\mathcal{N}(x)_{Pcoll} \leq 0.1))  \rightarrow (-5^{\circ} \leq \mathcal{N}(x)_{Steer} \leq 5^{\circ})

To enable the application of many verifiers, we can reduce the property to a set of verification problems with robustness properties.
This particular example is reduced to two verification problems with robustness properties.
Each of the problems produced pair a robustness property (i.e., :math:`\forall x. (x \in [0,1]^40000) \rightarrow (\mathcal{N}_0 > \mathcal{N}_1)`) with a modified version of the original DNN. 
The new DNN is created by incorporating a suffix network that takes in the outputs of the original DNN and classifies whether they constitute a violation of the original property. 
This suffix transforms the network into a classifier for which violations of a robustness property correspond to violations of the original property.

Reduction
---------

We rely on three assumptions to transform a verification problem into a reduced form.
First, the constraints on the network inputs must be represented as a union of convex polytopes.
Second, the constraints on the outputs of the network must be represented as a union of convex polytopes.
Third, we assume that each convex polytope is represented as a conjunction of linear inequalities.
Complying with these assumptions still enables properties to retain a high degree of expressiveness as unions of polytopes are extremely general and subsume other geometric representations, such as intervals and zonotopes.

We present each step of property reduction below and describe their application to the DroNet example described above.
For the purpose of this description we choose to reduce to verification problems with robustness problems.
Reducing to reachability problems differs only in the final property, which specifies that the output value :math:`\mathcal{N}(x)_0` must always be greater than 0.

Reformat the property
^^^^^^^^^^^^^^^^^^^^^

Reduction first negates the original property specification and converts it to disjunctive normal form (DNF).
Negating the specification means that a satisfying model falsifies the original property.
The DNF representation allows us to construct a property for each disjunct, such that if any are violated, the negated specification is satisfied and thus the original specification is falsified.
For each of these disjuncts the approach defines a new robustness problem, as described below.

Transform into halfspace-polytopes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Constraints in each disjunct that correspond to constraints over the output are converted to halfspace-polytope constraints, defined over the concatenation of the input and output domains.
A halfspace-polytope can be represented in the form :math:`Ax \leq b`, where :math:`A` is a matrix of :math:`k` rows, where each row represents 1 constraint, and :math:`m` columns, one for each dimension in the output space.
This representation facilitates the transformation of constraints into network operations.
To build the matrix :math:`A` and vector :math:`b`, we first transform all inequalities in the conjunction to :math:`\leq` inequalities with variables on the left-hand-side and constants on the right-hand-side.
The transformation first converts :math:`\geq` to :math:`\leq` and :math:`>` to :math:`<`.
Then, all variables are moved to the left-hand-side and all constants to the right-hand-side.
Next, :math:`<` constraints are converted to :math:`\leq` constraints by decrementing the constant value on the right-hand-side.
This transformation assumes that there exists a representable number with greatest possible value that is less than the right-hand-side.
Finally, each inequality is converted to a row of :math:`A` and value in :math:`b`.

Suffix construction
^^^^^^^^^^^^^^^^^^^

Using the generated halfspace-polytope, we build a suffix subnetwork that classifies whether outputs satisfy the specification.
The constructed suffix has two layers, a hidden fully-connected layer with ReLU activations, and dimension equal to the number of constraints in the halfspace-polytope defined by the current disjunct, and a final output layer of size 2.

The hidden layer of the suffix has a weight matrix equal to the constraint matrix, :math:`A`, of the halfspace-polytope representation, and a bias equal to :math:`-b`.
With this construction, each neuron will only have a value greater than 0 if the corresponding constraint is not satisfied, otherwise it will have a value less than or equal to 0, which becomes equal to 0 after the ReLU activation is applied.
In the DroNet problem for example, one of the constraints for a disjunct is :math:`(\mathcal{N}(x)_S \leq -5^{\circ})`.
For this conjunct we define the weights for one of the neurons to have a weight of 1 from :math:`\mathcal{N}(x)_S`, a weight of 0 from :math:`\mathcal{N}(x)_P`, and a bias of :math:`5^{\circ}`.

The output layer of the suffix has 2 neurons, each with no activation function.
The first of these neurons is the sum of all neurons in the previous layer, and has a bias value of 0.
Because the neurons in the previous layer each represent a constraint, and each of these neurons is 0 only when the constraint is satisfied, if the sum of all these neurons is 0, then the conjunction of the constraints is satisfied, indicating that a violation has been found.
The second of these neurons has a constant value of 0 -- all incoming weights and bias are 0.
The resulting network will predict class 1 if the input satisfies the corresponding disjunct and class 0 otherwise.

Problem construction
^^^^^^^^^^^^^^^^^^^^

The robustness property specification states that the network should classify all inputs that satisfy the input preconditions as class 0 -- no violations.
If a violation is found to this property, then the original property is violated by the input that violated the robustness property.
In the end, we have generated a set of verification problems such that, if any of the problems is violated, then the original problem is also violated.
This comes from our construction of a property for each disjunct in the DNF of the negation of the original property.
