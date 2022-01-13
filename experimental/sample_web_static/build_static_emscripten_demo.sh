#!/bin/bash
# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# set -x
set -e

###############################################################################
# Setup and checking for dependencies                                         #
###############################################################################

# D:\dev\projects\emsdk\emsdk_env.bat

if ! command -v emcmake &> /dev/null
then
  echo "'emcmake' not found, setup environment according to https://emscripten.org/docs/getting_started/downloads.html"
  exit
fi

CMAKE_BIN=${CMAKE_BIN:-$(which cmake)}
ROOT_DIR=$(git rev-parse --show-toplevel)

###############################################################################
# Compile from .mlir input to static C source files using host tools          #
###############################################################################

INSTALL_ROOT="D:\dev\projects\iree-build\install\bin"
TRANSLATE_TOOL="${INSTALL_ROOT?}\iree-translate.exe"
EMBED_DATA_TOOL="${INSTALL_ROOT?}\generate_embed_data.exe"
INPUT_NAME="simple_mul"

mkdir -p generated/

echo "=== Translating MLIR to static library output (.vmfb, .h, .o) ==="
${TRANSLATE_TOOL?} ${INPUT_NAME}.mlir \
  --iree-mlir-to-vm-bytecode-module \
  --iree-hal-target-backends=llvm \
  --iree-llvm-target-triple=wasm32-unknown-unknown \
  --iree-llvm-link-embedded=false \
  --iree-llvm-link-static \
  --iree-llvm-static-library-output-path=generated/${INPUT_NAME}_static.o \
  --o generated/${INPUT_NAME}.vmfb

echo "=== Embedding bytecode module (.vmfb) into C source files (.h, .c) ==="
${EMBED_DATA_TOOL?} generated/${INPUT_NAME}.vmfb \
  --output_header=generated/${INPUT_NAME}_bytecode.h \
  --output_impl=generated/${INPUT_NAME}_bytecode.c \
  --identifier=iree_static_${INPUT_NAME} \
  --flatten

###############################################################################
# Build the web artifacts using Emscripten                                    #
###############################################################################

echo "=== Building web artifacts using Emscripten ==="

mkdir -p ${ROOT_DIR?}/build-emscripten
pushd ${ROOT_DIR?}/build-emscripten

# Configure using Emscripten's CMake wrapper, then build.
# Note: The sample creates a task device directly, so no drivers are required,
#       but some targets are gated on specific CMake options.
emcmake "${CMAKE_BIN?}" -G Ninja .. \
  -DIREE_HOST_BINARY_ROOT=$PWD/../build-host/install \
  -DIREE_HAL_DRIVER_DEFAULTS=OFF \
  -DIREE_HAL_DRIVER_DYLIB=ON \
  -DIREE_BUILD_COMPILER=OFF \
  -DIREE_BUILD_TESTS=OFF

"${CMAKE_BIN?}" --build . --target \
  iree_experimental_sample_web_static_sync \
  iree_experimental_sample_web_static_multithreaded
popd

###############################################################################
# Serve the demo using a local webserver                                      #
###############################################################################

echo "=== Copying static files (index.html) to the build directory ==="

cp ${ROOT_DIR?}/experimental/sample_web_static/index.html \
  ${ROOT_DIR?}/build-emscripten/experimental/sample_web_static

echo "=== Running local webserver ==="

# local_server.py is needed when using SharedArrayBuffer, with multithreading
# python3 local_server.py \
#   --directory ${ROOT_DIR?}/build-emscripten/experimental/sample_web_static

# http.server on its own is fine for single threaded use, and this doesn't
# break CORS for external resources like easeljs from a CDN
python3 -m http.server \
  --directory ${ROOT_DIR?}/build-emscripten/experimental/sample_web_static
