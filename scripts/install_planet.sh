#!/usr/bin/env bash

PROJECT_DIR=$(cd $(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)/..; pwd)
cd bin

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

git clone https://github.com/progirep/planet.git planet-master
cd planet-master/src
qmake Tool.pro
make
cp planet ../../
