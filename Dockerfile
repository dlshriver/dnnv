FROM ubuntu:focal

LABEL maintainer="David Shriver"

RUN useradd -ms /bin/bash dnnv

SHELL ["/bin/bash", "-c"]

ENV TZ=America/New_York

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends -y \
    build-essential \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends -y \
    cmake \
    git \
    python3.8 \
    python3.8-dev \
    python3.8-venv \
    valgrind \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER dnnv
WORKDIR /home/dnnv/

# create venv
RUN python3.8 -m venv .venv \
    && . .venv/bin/activate \
    # upgrade pip and flit
    && pip install --upgrade pip flit \
    # load venv on interactive shell
    && echo ". .venv/bin/activate" >>.bashrc

# install DNNV base
COPY --chown=dnnv pyproject.toml README.md ./
COPY --chown=dnnv dnnv/__init__.py dnnv/__version__.py dnnv/
RUN . .venv/bin/activate && flit install -s

# build test artifacts
COPY --chown=dnnv tests/system_tests/artifacts/build_artifacts.py tests/system_tests/artifacts/build_artifacts.py
RUN . .venv/bin/activate && python tests/system_tests/artifacts/build_artifacts.py

# copy project files to container
COPY --chown=dnnv . .

RUN . .venv/bin/activate \
    # # install verifiers
    # # verifiers that do not require gurobi
    # && dnnv_manage install eran \
    # && dnnv_manage install marabou \
    # && dnnv_manage install neurify \
    # && dnnv_manage install nnenum \
    # && dnnv_manage install planet \
    # && dnnv_manage install reluplex \
    # # verifiers that require gurobi license
    # && dnnv_manage install bab \
    # && dnnv_manage install mipverify \
    # && dnnv_manage install verinet \
    # clean up cache
    && rm -rf .cache
