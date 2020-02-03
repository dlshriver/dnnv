#!/usr/bin/env bash

PROJECT_DIR=$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)
cd lib

git clone https://github.com/eth-sri/eran.git
cd eran
git pull
git checkout a1a870f

export LD_LIBRARY_PATH=$PROJECT_DIR/lib:$LD_LIBRARY_PATH

wget ftp://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz
tar -xvzf m4-1.4.1.tar.gz
cd m4-1.4.1
./configure --prefix=$PROJECT_DIR
make
make install
cd ..
rm m4-1.4.1.tar.gz

wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
tar -xvf gmp-6.1.2.tar.xz
cd gmp-6.1.2
./configure --enable-cxx --prefix=$PROJECT_DIR
make
make install
cd ..
rm gmp-6.1.2.tar.xz

wget https://www.mpfr.org/mpfr-current/mpfr-4.0.2.tar.xz
tar -xvf mpfr-4.0.2.tar.xz
cd mpfr-4.0.2
CFLAGS="$CFLAGS -I$PROJECT_DIR/include" CXXFLAGS="$CXXFLAGS -I$PROJECT_DIR/include" LDFLAGS="$LDFLAGS -L$PROJECT_DIR/lib" ./configure --prefix=$PROJECT_DIR
make
make install
cd ..
rm mpfr-4.0.1.tar.xz

git clone https://github.com/eth-sri/ELINA.git
cd ELINA
LDFLAGS="-L$PROJECT_DIR/lib" CXXFLAGS="-I$PROJECT_DIR/include" ./configure -prefix $PROJECT_DIR -gmp-prefix $PROJECT_DIR -mpfr-prefix $PROJECT_DIR
make
make install
cd ..

pip install \
    --global-option=build_ext --global-option="-I$PROJECT_DIR/lib/eran/gmp-6.1.2" \
    --global-option=build_ext --global-option="-L$PROJECT_DIR/lib/eran/gmp-6.1.2/.libs" \
    pycddlib

cd $PROJECT_DIR
./scripts/install_gurobi.sh
