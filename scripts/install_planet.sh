#!/usr/bin/env bash

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}

set -x

## requirements:
# libglpk-dev
# qt5-qmake
# valgrind
# libltdl-dev
# protobuf-compiler

# troubleshooting on Ubuntu:

# if no qmake, or qmake error, try:
# sudo apt-get install qt5-default qt5-qmake

# if make fails due to missing glpk, try:
# sudo apt-get install libglpk-dev

# if make fails due to missing valgrind, try:
# sudo apt-get install valgrind

# if make fails due to missing ltdl, try:
# sudo apt-get install libltdl-dev

cd $PROJECT_DIR
./scripts/install_lpsolve.sh

cd $PROJECT_DIR
./scripts/install_suitesparse.sh

cd $PROJECT_DIR/lib
wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx --prefix=$PROJECT_DIR
make
make install

cd $PROJECT_DIR/lib
wget http://gnu.mirror.constant.com/libtool/libtool-2.4.tar.xz
tar xf libtool-2.4.tar.xz
cd libtool-2.4
./configure --prefix=$PROJECT_DIR
make
make install

cd $PROJECT_DIR/lib
wget https://www.zlib.net/zlib-1.2.11.tar.xz
tar xf zlib-1.2.11.tar.xz
cd zlib-1.2.11
make
make install

cd $PROJECT_DIR/lib
wget http://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz
tar xf glpk-4.65.tar.gz
cd glpk-4.65
./configure --prefix=$PROJECT_DIR
make
make install

cd $PROJECT_DIR/lib
wget https://sourceware.org/pub/valgrind/valgrind-3.15.0.tar.bz2
tar xf valgrind-3.15.0.tar.bz2
cd valgrind-3.15.0
./configure --prefix=$PROJECT_DIR
make
make install

cd $PROJECT_DIR/bin
git clone https://github.com/progirep/planet.git planet-master
cd planet-master/src
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o Options.o minisat2/Options.cc
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o Solver.o minisat2/Solver.cc
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o System.o minisat2/System.cc
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o main.o main.cpp
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o verifierContext.o verifierContext.cpp
g++ -c -m64 -pipe -std=c++14 -g -O2 -Wall -W -fPIC -DUSE_GLPK -DNDEBUG -I. -I$PROJECT_DIR/include -L$PROJECT_DIR/lib/ -o supersetdatabase.o supersetdatabase.cpp
g++ -m64 -Wl,-O1 -L$PROJECT_DIR/lib -o planet Options.o Solver.o System.o main.o verifierContext.o supersetdatabase.o -Bstatic -lglpk -lgmp -lumfpack -lcholmod -lamd -lcolamd -lccolamd -lcamd -lz -lltdl -ldl -lsuitesparseconfig
cp planet $PROJECT_DIR/bin/
