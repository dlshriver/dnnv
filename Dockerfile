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
RUN apt-get -qq install -y liblapack-dev
RUN apt-get -qq install -y openssl libssl-dev
RUN apt-get -qq install -y valgrind
RUN apt-get -qq install -y libtool

USER dnnv
WORKDIR /home/dnnv/

ENV MAKEFLAGS="--silent"

# load env on interactive shell
RUN echo "source .env.d/openenv.sh" >>.bashrc

# create dummy venv for verifier install
RUN python3.7 -m venv .venv

# copy infrequently changed files to container
COPY --chown=dnnv manage.sh .
COPY --chown=dnnv pyproject.toml .
COPY --chown=dnnv .env.d/ .env.d/
COPY --chown=dnnv scripts/ scripts/
COPY --chown=dnnv tools/verifier_runners tools/verifier_runners

# install verifiers
RUN ./manage.sh install neurify
RUN ./manage.sh install eran
RUN ./manage.sh install reluplex
RUN ./manage.sh install planet
RUN ./manage.sh install marabou
RUN ./manage.sh install nnenum
# RUN ./manage.sh install bab # requires gurobi
# RUN ./manage.sh install mipverify # requires gurobi
# RUN ./manage.sh install verinet # requires gurobi

COPY --chown=dnnv dnnv/ dnnv/
COPY --chown=dnnv README.md .
RUN ./manage.sh update

COPY --chown=dnnv tests/ tests/
RUN source .env.d/openenv.sh && python tests/artifacts/build_artifacts.py

COPY --chown=dnnv docs/ docs/
COPY --chown=dnnv tools/ tools/
