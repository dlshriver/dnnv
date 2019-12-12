#!/usr/bin/env bash

PROJECT_DIR=$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)
cd bin

wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.4-linux-x86_64.tar.gz
tar -xvf julia-1.0.4-linux-x86_64.tar.gz
rm julia-1.0.4-linux-x86_64.tar.gz

cd $PROJECT_DIR
./scripts/install_gurobi.sh

julia -e 'using Pkg; Pkg.add("Gurobi")'
julia -e 'using Pkg; Pkg.add("MAT")'
julia -e 'using Pkg; Pkg.add(PackageSpec(url="https://github.com/vtjeng/MIPVerify.jl"))'

# if fails due to libz, must install libz-dev
# sudo apt-get install libz-dev
