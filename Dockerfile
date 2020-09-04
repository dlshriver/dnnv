FROM ubuntu:focal

LABEL maintainer="David Shriver"

RUN useradd -ms /bin/bash dnnv

SHELL ["/bin/bash", "-c"]

RUN apt update
RUN apt install -y software-properties-common
RUN apt install -y build-essential
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt update
RUN apt install -y python3.7 python3.7-dev python3.7-venv
RUN apt install -y gcc g++
RUN apt install -y wget
RUN apt install -y git
RUN apt install -y liblapack-dev

USER dnnv
WORKDIR /home/dnnv/
COPY --chown=dnnv . .

RUN chmod u+x .env.d/*
RUN chmod u+x scripts/*

RUN ./manage.sh init
RUN echo "source .env.d/openenv.sh" >>.bashrc
RUN ./manage.sh install neurify
RUN ./manage.sh install eran
RUN ./manage.sh install reluplex
RUN ./manage.sh install planet
# RUN ./manage.sh install bab # requires gurobi
RUN source .env.d/openenv.sh; python tests/artifacts/build_artifacts.py
