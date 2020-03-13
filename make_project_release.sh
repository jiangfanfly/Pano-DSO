#!/usr/bin/env bash

BUILD_TYPE1=Release
BUILD_TYPE2=Debug
NUM_PROC=2

BASEDIR="$PWD"

# cd "$BASEDIR/thirdparty/DBoW3"
# mkdir build
# cd build
# cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE1 ..
# make -j$NUM_PROC

# cd "$BASEDIR/thirdparty/g2o"
# mkdir build
# cd build
# cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE1 ..
# make -j$NUM_PROC

cd "$BASEDIR"
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE1 ..
make ##-j$NUM_PROC

