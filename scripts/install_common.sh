#!/usr/bin/env bash

set -x

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

    wget https://github.com/cddlib/cddlib/releases/download/0.94j/cddlib-0.94j.tar.gz
    tar -xvf cddlib-0.94j.tar.gz
    cd cddlib-0.94j
    ./configure --prefix=$PROJECT_DIR
    make
    make install
    cd ..
    rm cddlib-0.94j.tar.gz
}

ensure_cmake() {
    cd $PROJECT_DIR/bin
    if [ -e cmake ]; then
        echo "CMAKE is already installed"
        return
    fi

    wget https://github.com/Kitware/CMake/releases/download/v3.17.1/cmake-3.17.1-Linux-x86_64.sh
    chmod u+x cmake-3.17.1-Linux-x86_64.sh
    yes | ./cmake-3.17.1-Linux-x86_64.sh
    cp -r cmake-3.17.1-Linux-x86_64/bin/* .
    cp -r cmake-3.17.1-Linux-x86_64/share/* $PROJECT_DIR/share/
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

    wget http://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz
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

    wget https://gmplib.org/download/gmp/gmp-6.1.2.tar.xz
    tar -xvf gmp-6.1.2.tar.xz
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

    wget https://packages.gurobi.com/9.0/gurobi9.0.2_linux64.tar.gz
    tar -xvf gurobi9.0.2_linux64.tar.gz
    cd gurobi902/linux64
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

    wget https://julialang-s3.julialang.org/bin/linux/x64/1.0/julia-1.0.4-linux-x86_64.tar.gz
    tar -xvf julia-1.0.4-linux-x86_64.tar.gz
    rm julia-1.0.4-linux-x86_64.tar.gz
}

ensure_libtool() {
    cd $PROJECT_DIR/bin
    if [ -e libtool ]; then
        echo "libtool is already installed"
        return
    fi

    wget http://gnu.mirror.constant.com/libtool/libtool-2.4.tar.xz
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
    wget https://downloads.sourceforge.net/project/lpsolve/lpsolve/5.5.2.5/lp_solve_5.5.2.5_dev_ux64.tar.gz
    tar -xzf lp_solve_5.5.2.5_dev_ux64.tar.gz
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

    wget ftp://ftp.gnu.org/gnu/m4/m4-1.4.1.tar.gz
    tar -xvzf m4-1.4.1.tar.gz
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

    wget https://www.mpfr.org/mpfr-current/mpfr-4.1.0.tar.xz
    tar -xvf mpfr-4.1.0.tar.xz
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

    wget http://github.com/xianyi/OpenBLAS/archive/v0.3.6.tar.gz
    tar -xzf v0.3.6.tar.gz
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
    wget https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v5.6.0.tar.gz
    tar -xzf v5.6.0.tar.gz
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
    if [ -e valgrind ]; then
        echo "zlib is already installed"
        return
    fi

    wget https://sourceware.org/pub/valgrind/valgrind-3.15.0.tar.bz2
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

    wget https://www.zlib.net/zlib-1.2.11.tar.xz
    tar xf zlib-1.2.11.tar.xz
    cd zlib-1.2.11
    ./configure --prefix=$PROJECT_DIR
    make
    make install
    cd ..
    rm zlib-1.2.11.tar.xz
}
