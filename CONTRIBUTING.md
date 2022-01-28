# Contributing to DNNV

Thank you for choosing to invest your time in contributing to DNNV!
Before participating, please read our [Code of Conduct](./CODE_OF_CONDUCT.md) 
to help keep our community approachable and respectable.


## How to Ask a Question

If you have a question, please don't hesitate to ask!
We only ask that you not file an issue to ask a question. 
We have an official [discussion board](https://github.com/dlshriver/dnnv/discussions) on GitHub where the community and DNNV developers can chime in with helpful advice on any questions.


## How to Contribute

### Report a Bug

Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/dlshriver/dnnv/issues).
If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/dlshriver/dnnv/issues/new). 
Be sure to include a **title** and **clear description** with as much relevant information as possible, as well as a **code sample** or an **executable test case** demonstrating the issue.

### Submit a Patch

Open a new GitHub pull request with the patch.
Ensure the PR description clearly describes the problem and solution. Include the relevant issue number if applicable.
<!-- 
TODO: Describe coding conventions, etc.
-->
Every patch should also include relevant tests to show that the patch behaves as intended.
Before opening a pull request with your path, please run the test suite on your local machine to ensure that all tests pass.
To run the test suite, first build the test artifacts (only needs to be built once), and then run coveragepy, as follows:

```bash
$ python tests/system_tests/artifacts/build_artifacts.py # only needs to be run the first time you run the test suite
$ coverage run
```

> Cosmetic changes that do not add anything substantial to the stability, functionality, or testability of DNNV will generally not be accepted.

<!--
TODO:
## General Overview

Information on repo layout, standards, conventions, etc.
-->
