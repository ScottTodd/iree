#!/bin/bash
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Cross-compile IREE's compiler through Emscripten to WebAssembly with CMake.
# Designed for CI, but can be run manually. This uses previously cached build
# results and does not clear build directories.
#
# Host binaries (e.g. compiler tools) should already be built at
# ./build-host/install. Emscripten binaries (e.g. .wasm and .js files) will be
# built in ./build-emscripten-compiler/.

set -x
set -e

if ! command -v emcmake &> /dev/null
then
    echo "'emcmake' not found, setup environment according to https://emscripten.org/docs/getting_started/downloads.html"
    exit
fi

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
"${CMAKE_BIN?}" --version
ninja --version

ROOT_DIR=$(git rev-parse --show-toplevel)
cd ${ROOT_DIR?}

if [ -d "build-emscripten-compiler" ]
then
  echo "build-emscripten-compiler directory already exists. Will use cached results there."
else
  echo "build-emscripten-compiler directory does not already exist. Creating a new one."
  mkdir build-emscripten-compiler
fi
cd build-emscripten-compiler

# Configure using Emscripten's CMake wrapper, then build.
emcmake "${CMAKE_BIN?}" -G Ninja .. \
  -DIREE_HOST_BINARY_ROOT=$PWD/../build-host/install \
  -DLLVM_TABLEGEN=$PWD/../build-host/third_party/llvm-project/llvm/bin/llvm-tblgen.exe \
  -DLLVM_TARGET_ARCH=wasm32 \
  -DLLVM_DEFAULT_TARGET_TRIPLE=wasm32-unknown-unknown \
  -DLLVM_TARGETS_TO_BUILD=WebAssembly \
  -DCMAKE_SYSTEM_NAME=Generic \
  -DIREE_HAL_DRIVERS_TO_BUILD=VMVX \
  -DIREE_BUILD_COMPILER=ON \
  -DIREE_BUILD_TESTS=OFF \
  -DIREE_BUILD_SAMPLES=OFF

# TODO(scotttodd): expand this list of targets
"${CMAKE_BIN?}" --build . --target iree_tools_iree-opt
