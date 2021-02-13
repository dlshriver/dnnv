#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}
source $PROJECT_DIR/scripts/install_common.sh

ensure_julia
ensure_gurobi
ensure_zlib

julia -e 'using Pkg; Pkg.add("Gurobi")'
julia -e 'using Pkg; Pkg.add("MAT")'
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/vtjeng/MIPVerify.jl", rev="49cd9c7"))'
julia -e 'using Pkg; Pkg.update()'
