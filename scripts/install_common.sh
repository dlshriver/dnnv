#!/usr/bin/env bash

set -x
mkdir -p bin
mkdir -p include
mkdir -p lib
mkdir -p share

PROJECT_DIR=${PROJECT_DIR:-$(
    cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..
    pwd
)}

ensure_cddlib() {
    cd $PROJECT_DIR/lib
    if [ -e libcdd.a ]; then
        echo "cddlib is already installed"
        return
    fi

    wget -q https://github.com/cddlib/cddlib/releases/download/0.94j/cddlib-0.94j.tar.gz
    tar xf cddlib-0.94j.tar.gz
    cd cddlib-0.94j
    ./configure --prefix=$PROJECT_DIR
    make
    make install
    cd ..
    rm cddlib-0.94j.tar.gz
}

ensure_cmake() {
    cd $PROJECT_DIR/bin
    if [ $(command -v cmake) ]; then
        echo "CMAKE is already installed"
        return
    fi

    wget -q https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2.tar.gz
    tar xf cmake-3.18.2.tar.gz
    cd cmake-3.18.2/
    ./bootstrap --prefix=$PROJECT_DIR
    make
    make install
    cd $PROJECT_DIR/bin
    rm cmake-3.18.2.tar.gz
}

ensure_elina() {
    cd $PROJECT_DIR/lib
    if [ -e ELINA ]; then
        echo "ELINA is already installed"
        return
    fi

    git clone https://github.com/eth-sri/ELINA.git
    cd ELINA
    git pull
    git checkout 23fe9d5
    LDFLAGS="-L$PROJECT_DIR/lib" CXXFLAGS="-I$PROJECT_DIR/include" ./configure -prefix $PROJECT_DIR -gmp-prefix $PROJECT_DIR -mpfr-prefix $PROJECT_DIR -cdd-prefix $PROJECT_DIR -use-deeppoly
    make
    make install
}

ensure_glpk() {
    cd $PROJECT_DIR/lib
    if [ -e libglpk.a ]; then
        echo "glpk is already installed"
        return
    fi

    wget -q https://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz
    tar xf glpk-4.65.tar.gz
    cd glpk-4.65
    ./configure --prefix=$PROJECT_DIR
    make
    make install
    cd ..
    rm glpk-4.65.tar.gz
}

ensure_gmp() {
    cd $PROJECT_DIR/lib
    if [ -e libgmp.so ]; then
        echo "gmp is already installed"
        return
    fi

    wget -q https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
    tar xf gmp-6.1.2.tar.xz
    cd gmp-6.1.2
    ./configure --enable-cxx --prefix=$PROJECT_DIR
    make
    make install
    cd ..
    rm gmp-6.1.2.tar.xz
}

ensure_gurobi() {
    cd $PROJECT_DIR/bin
    if [ -e gurobi902 ]; then
        echo "Gurobi is already installed"
        return
    fi

    wget -q https://packages.gurobi.com/9.0/gurobi9.0.2_linux64.tar.gz
    tar xf gurobi9.0.2_linux64.tar.gz
    cd gurobi902/linux64
    cp bin/* $PROJECT_DIR/bin/
    cp lib/libgurobi90.so $PROJECT_DIR/lib
    python3 setup.py install
    cd ../../
    rm gurobi9.0.2_linux64.tar.gz

    echo "Gurobi requires a license, which can be obtained from gurobi.com"
}

ensure_julia() {
    cd $PROJECT_DIR/bin
    if [ -e julia ]; then
        echo "julia is already installed"
        return
    fi

    wget -q https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.4-linux-x86_64.tar.gz
    tar xf julia-1.0.4-linux-x86_64.tar.gz
    rm julia-1.0.4-linux-x86_64.tar.gz
}

ensure_lapack() {
    cd $PROJECT_DIR/lib
    if [ -e $PROJECT_DIR/lib/liblapack.a ]; then
        echo "lapack is already installed"
        return
    fi

    wget -q https://github.com/Reference-LAPACK/lapack/archive/v3.9.0.tar.gz
    tar xf v3.9.0.tar.gz
    cd lapack-3.9.0
    mkdir build
    cd build
    cmake ..
    cmake --build . -j
    cp lib/*.a $PROJECT_DIR/lib/
    cd ../..
    rm v3.9.0.tar.gz
}

ensure_libtool() {
    cd $PROJECT_DIR/bin
    if [ -e $PROJECT_DIR/lib/libltdl.so ]; then
        echo "libtool is already installed"
        return
    fi

    wget -q http://gnu.mirror.constant.com/libtool/libtool-2.4.tar.xz
    tar xf libtool-2.4.tar.xz
    cd libtool-2.4
    ./configure --prefix=$PROJECT_DIR
    make
    make install
    cd ..
    rm libtool-2.4.tar.xz
}

ensure_lpsolve() {
    cd $PROJECT_DIR/include
    if [ -e lpsolve ]; then
        echo "lpsolve is already installed"
        return
    fi

    cd $PROJECT_DIR/lib
    wget -q https://downloads.sourceforge.net/project/lpsolve/lpsolve/5.5.2.5/lp_solve_5.5.2.5_dev_ux64.tar.gz
    tar xf lp_solve_5.5.2.5_dev_ux64.tar.gz
    rm lp_solve_5.5.2.5_dev_ux64.tar.gz
    mkdir $PROJECT_DIR/include/lpsolve/
    cp lp_*.h $PROJECT_DIR/include/lpsolve/
}

ensure_m4() {
    cd $PROJECT_DIR/bin
    if [ -e m4 ]; then
        echo "m4 is already installed"
        return
    fi

    wget -q https://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz
    tar xf m4-1.4.1.tar.gz
    cd m4-1.4.1
    ./configure --prefix=$PROJECT_DIR
    make
    make install
    cd ..
    rm m4-1.4.1.tar.gz
}

ensure_mpfr() {
    cd $PROJECT_DIR/lib
    if [ -e libmpfr.a ]; then
        echo "mpfr is already installed"
        return
    fi

    wget -q https://www.mpfr.org/mpfr-current/mpfr-4.1.0.tar.xz
    tar xf mpfr-4.1.0.tar.xz
    cd mpfr-4.1.0
    CFLAGS="$CFLAGS -I$PROJECT_DIR/include" CXXFLAGS="$CXXFLAGS -I$PROJECT_DIR/include" LDFLAGS="$LDFLAGS -L$PROJECT_DIR/lib" ./configure --prefix=$PROJECT_DIR
    make
    make install
    cd ..
    rm mpfr-4.1.0.tar.xz
}

ensure_openblas() {
    cd $PROJECT_DIR/lib
    if [ -e libopenblas.so ]; then
        echo "OpenBLAS is already installed"
        return
    fi

    wget -q https://github.com/xianyi/OpenBLAS/archive/v0.3.6.tar.gz
    tar xf v0.3.6.tar.gz
    cd OpenBLAS-0.3.6
    make
    make PREFIX=$PROJECT_DIR install
    cd ..
    rm v0.3.6.tar.gz
}

ensure_suitesparse() {
    cd $PROJECT_DIR/lib
    if [ -e libsuitesparseconfig.so ]; then
        echo "SuiteSparse is already installed"
        return
    fi

    cd $PROJECT_DIR/lib
    wget -q https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v5.6.0.tar.gz
    tar xf v5.6.0.tar.gz
    cd SuiteSparse-5.6.0
    make library BLAS="-L$PROJECT_DIR/lib -lopenblas"
    cp lib/* $PROJECT_DIR/lib
    cp */Lib/*.a $PROJECT_DIR/lib
    cp SuiteSparse_config/libsuitesparseconfig.a $PROJECT_DIR/lib
    cd $PROJECT_DIR/lib
    rm v5.6.0.tar.gz
}

ensure_valgrind() {
    cd $PROJECT_DIR/bin
    if [ -e $PROJECT_DIR/include/valgrind/callgrind.h ]; then
        echo "valgrind is already installed"
        return
    fi

    wget -q https://sourceware.org/pub/valgrind/valgrind-3.15.0.tar.bz2
    tar xf valgrind-3.15.0.tar.bz2
    cd valgrind-3.15.0
    ./configure --prefix=$PROJECT_DIR
    make
    make install
}

ensure_zlib() {
    cd $PROJECT_DIR/lib
    if [ -e libz.a ]; then
        echo "zlib is already installed"
        return
    fi

    wget -q https://www.zlib.net/zlib-1.2.11.tar.xz
    tar xf zlib-1.2.11.tar.xz
    cd zlib-1.2.11
    ./configure --prefix=$PROJECT_DIR
    make
    make install
    cd ..
    rm zlib-1.2.11.tar.xz
}
