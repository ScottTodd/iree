#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Builds test artifacts using a provided "package build".
#
# The package directory to use is passed as the first argument.
#
# Tests considered in-scope for this script:
#   * `runtime/` tests
#   * `tests/`, `tools/`, `samples/`, etc. tests from other directories that
#     use binaries from the CMake `IREE_HOST_BIN_DIR` option
#
# Tests considered out-of-scope for this script:
#   * `compiler/` tests and others using the `IREE_BUILD_COMPILER` CMake option

###############################################################################
# Script setup                                                                #
###############################################################################

set -euo pipefail

PACKAGE_DIR="$1"
SOURCE_DIR_ROOT=$(git rev-parse --show-toplevel)
TEST_BUILD_DIR="${TEST_BUILD_DIR:-build-tests}"
LLVM_EXTERNAL_LIT="${LLVM_EXTERNAL_LIT:-${SOURCE_DIR_ROOT}/third_party/llvm-project/llvm/utils/lit/lit.py}"

# Respect user settings, but default to turning off all GPU tests.
export IREE_VULKAN_DISABLE="${IREE_VULKAN_DISABLE:-1}"
export IREE_METAL_DISABLE="${IREE_METAL_DISABLE:-1}"
export IREE_CUDA_DISABLE="${IREE_CUDA_DISABLE:-1}"

# Set cmake options based on disabled features.
declare -a cmake_config_options=()
if (( IREE_VULKAN_DISABLE == 1 )); then
  cmake_config_options+=("-DIREE_HAL_DRIVER_VULKAN=OFF")
fi
if (( IREE_METAL_DISABLE == 1 )); then
  cmake_config_options+=("-DIREE_HAL_DRIVER_METAL=OFF")
fi
if (( IREE_CUDA_DISABLE == 1 )); then
  cmake_config_options+=("-DIREE_HAL_DRIVER_CUDA=OFF")
fi

###############################################################################
# Build the runtime and compile 'test deps'                                   #
###############################################################################

echo "::group::Configure"
cmake_args=(
  "."
  "-G Ninja"
  "-B ${TEST_BUILD_DIR?}"
  "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
  "-DIREE_BUILD_PYTHON_BINDINGS=OFF"
  "-DIREE_BUILD_COMPILER=OFF"
  "-DIREE_HOST_BIN_DIR=${PACKAGE_DIR?}/bin"
  "-DLLVM_EXTERNAL_LIT=${LLVM_EXTERNAL_LIT?}"
)
cmake_args+=(${cmake_config_options[@]})
cmake ${cmake_args[@]}
echo "::endgroup::"

echo "::group::Build runtime targets"
cmake --build ${TEST_BUILD_DIR?}
echo "::endgroup::"

echo "::group::Build iree-test-deps"
cmake --build ${TEST_BUILD_DIR?} --target iree-test-deps
echo "::endgroup::"
