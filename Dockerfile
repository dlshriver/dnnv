FROM ubuntu:focal

LABEL maintainer="David Shriver"

RUN useradd -ms /bin/bash dnnv

SHELL ["/bin/bash", "-c"]

RUN apt-get -qq update
RUN apt-get -qq install -y software-properties-common
RUN apt-get -qq install -y build-essential
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get -qq update
RUN apt-get -qq install -y python3.7 python3.7-dev python3.7-venv
RUN apt-get -qq install -y cmake
RUN apt-get -qq install -y wget
RUN apt-get -qq install -y git
RUN apt-get -qq install -y valgrind

USER dnnv
WORKDIR /home/dnnv/

# create venv
RUN python3.7 -m venv .venv
RUN . .venv/bin/activate && pip install --upgrade pip flit
# load venv on interactive shell
RUN echo ". .venv/bin/activate" >>.bashrc

# install DNNV base
COPY --chown=dnnv pyproject.toml .
COPY --chown=dnnv README.md .
COPY --chown=dnnv dnnv/__init__.py dnnv/__init__.py
COPY --chown=dnnv dnnv/__version__.py dnnv/__version__.py
RUN . .venv/bin/activate && flit install -s

# build test artifacts
COPY --chown=dnnv tests/artifacts/ tests/artifacts/
RUN . .venv/bin/activate && python tests/artifacts/build_artifacts.py

# copy files to container
COPY --chown=dnnv scripts/ scripts/
COPY --chown=dnnv dnnv/ dnnv/
COPY --chown=dnnv tests/ tests/
COPY --chown=dnnv docs/ docs/
COPY --chown=dnnv tools/ tools/

# install verifiers
# RUN . .venv/bin/activate && dnnv_manage install bab # requires gurobi
# RUN . .venv/bin/activate && dnnv_manage install eran
# RUN . .venv/bin/activate && dnnv_manage install marabou
# RUN . .venv/bin/activate && dnnv_manage install mipverify # requires gurobi
# RUN . .venv/bin/activate && dnnv_manage install neurify
# RUN . .venv/bin/activate && dnnv_manage install nnenum
# RUN . .venv/bin/activate && dnnv_manage install planet
# RUN . .venv/bin/activate && dnnv_manage install reluplex
# RUN . .venv/bin/activate && dnnv_manage install verinet # requires gurobi
