#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)}
cd $PROJECT_DIR/bin

git clone https://github.com/dlshriver/ReluplexCav2017.git
cd ReluplexCav2017
make
cp check_properties/generic_prover/generic_prover.elf ../reluplex
