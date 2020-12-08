#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}
source $PROJECT_DIR/scripts/install_common.sh

ensure_cmake
ensure_openblas
ensure_lpsolve
ensure_suitesparse
ensure_m4
ensure_gmp

ensure_libtool
ensure_zlib
ensure_glpk
ensure_valgrind

cd $PROJECT_DIR/bin
git clone https://github.com/progirep/planet.git planet-master
cd planet-master/src
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o Options.o minisat2/Options.cc
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o Solver.o minisat2/Solver.cc
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o System.o minisat2/System.cc
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o main.o main.cpp
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o verifierContext.o verifierContext.cpp
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o supersetdatabase.o supersetdatabase.cpp
g++ -m64 -Wl,-O1 -L$PROJECT_DIR/lib/ -o planet Options.o Solver.o System.o main.o verifierContext.o supersetdatabase.o -Bstatic -lglpk -lgmp -lumfpack -lcholmod -lamd -lcolamd -lccolamd -lcamd -lz -lltdl -ldl -lsuitesparseconfig
cp planet $PROJECT_DIR/bin/
