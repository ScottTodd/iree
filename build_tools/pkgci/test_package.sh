#!/bin/bash

# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Runs tests.
#
# This expects that `build_tests_using_package.sh` has already been run.

# TODO(scotttodd): Test iree-compiler Python packages. This only tests native
#   iree-dist packages right now.

###############################################################################
# Script setup                                                                #
###############################################################################

set -euo pipefail

SOURCE_DIR_ROOT=$(git rev-parse --show-toplevel)
TEST_BUILD_DIR="${TEST_BUILD_DIR:-build-tests}"

# Respect user settings, but default to as many test actions as we have cores.
get_default_parallel_level() {
  if [[ "$(uname)" == "Darwin" ]]; then
    echo "$(sysctl -n hw.logicalcpu)"
  else
    echo "$(nproc)"
  fi
}
export CTEST_PARALLEL_LEVEL="${CTEST_PARALLEL_LEVEL:-$(get_default_parallel_level)}"

# Allow users to provide any additional --tests-regex parameters to ctest.
export IREE_CTEST_TESTS_REGEX="${IREE_CTEST_TESTS_REGEX:-}"
# Allow users to provide any additional --label-regex parameters to ctest.
export IREE_CTEST_LABEL_REGEX="${IREE_CTEST_LABEL_REGEX:-}"

# Respect user settings, but default to turning off all GPU tests.
export IREE_VULKAN_DISABLE="${IREE_VULKAN_DISABLE:-1}"
export IREE_METAL_DISABLE="${IREE_METAL_DISABLE:-1}"
export IREE_CUDA_DISABLE="${IREE_CUDA_DISABLE:-1}"

# Respect user settings, but default to turning off specialized hardware tests.
# The VK_KHR_shader_float16_int8 extension is optional prior to Vulkan 1.2.
export IREE_VULKAN_F16_DISABLE="${IREE_VULKAN_F16_DISABLE:-1}"
# Some tests (typically using CUDA) require an NVIDIA GPU.
export IREE_NVIDIA_GPU_TESTS_DISABLE="${IREE_NVIDIA_GPU_TESTS_DISABLE:-1}"
# Some tests require the specific SM80 architecture of NVIDIA GPU.
export IREE_NVIDIA_SM80_TESTS_DISABLE="${IREE_NVIDIA_SM80_TESTS_DISABLE:-1}"
# Some tests require the specific RDNA architecture of AMD GPU.
export IREE_AMD_RDNA3_TESTS_DISABLE="${IREE_AMD_RDNA3_TESTS_DISABLE:-1}"
# Some tests require multiple devices (GPUs).
export IREE_MULTI_DEVICE_TESTS_DISABLE="${IREE_MULTI_DEVICE_TESTS_DISABLE:-1}"

# Set ctest label exclusions based on disabled features.
declare -a label_exclude_args=()
if (( IREE_VULKAN_DISABLE == 1 )); then
  label_exclude_args+=("^driver=vulkan$")
fi
if (( IREE_METAL_DISABLE == 1 )); then
  label_exclude_args+=("^driver=metal$")
fi
if (( IREE_CUDA_DISABLE == 1 )); then
  label_exclude_args+=("^driver=cuda$")
fi
if (( IREE_VULKAN_F16_DISABLE == 1 )); then
  label_exclude_args+=("^vulkan_uses_vk_khr_shader_float16_int8$")
fi
if (( IREE_NVIDIA_GPU_TESTS_DISABLE == 1 )); then
  label_exclude_args+=("^requires-gpu")
fi
if (( IREE_NVIDIA_SM80_TESTS_DISABLE == 1 )); then
  label_exclude_args+=("^requires-gpu-sm80$")
fi
if (( IREE_AMD_RDNA3_TESTS_DISABLE == 1 )); then
  label_exclude_args+=("^requires-gpu-rdna3$")
fi
if (( IREE_MULTI_DEVICE_TESTS_DISABLE == 1 )); then
  label_exclude_args+=("^requires-multiple-devices$")
fi

# Some tests are just failing on some platforms and this filtering lets us
# exclude any type of test. Ideally each test would be tagged with the
# platforms it doesn't support, but that would require editing through layers
# of CMake functions. Hopefully this list stays very short.
declare -a excluded_tests=()
if [[ "${OSTYPE}" =~ ^msys ]]; then
  # These tests are failing on Windows.
  excluded_tests+=(
    # TODO(#11077): INVALID_ARGUMENT: argument/result signature mismatch
    "iree/tests/e2e/matmul/e2e_matmul_dt_uk_i8_small_vmvx_local-task"
    "iree/tests/e2e/matmul/e2e_matmul_dt_uk_f32_small_vmvx_local-task"
    # TODO: Regressed when `pack` ukernel gained a uint64_t parameter in #13264.
    "iree/tests/e2e/tensor_ops/check_vmvx_ukernel_local-task_pack.mlir"
    "iree/tests/e2e/tensor_ops/check_vmvx_ukernel_local-task_pack_dynamic_inner_tiles.mlir"
    # TODO: Fix equality mismatch
    "iree/tests/e2e/tensor_ops/check_vmvx_ukernel_local-task_unpack.mlir"
    # TODO(#11070): Fix argument/result signature mismatch
    "iree/tests/e2e/tosa_ops/check_vmvx_local-sync_microkernels_fully_connected.mlir"

    # TODO(scotttodd): Fix "No Windows linker tool specified or discovered" with packages
    "iree/tests/e2e/regression/libm_linking.mlir.test"
    # TODO(scotttodd): Debug mystery segfault on Windows when used from packages
    "iree/samples/custom_module/dynamic/test/example.mlir.test"
    # TODO(scotttodd): Dev Windows machine can't find numpy here... why?
    "iree/tools/test/iree-run-module-outputs.mlir.test"
  )
elif [[ "${OSTYPE}" =~ ^darwin ]]; then
  excluded_tests+=(
    #TODO(#13501): Fix failing sample on macOS
    "iree/samples/custom_module/async/test/example.mlir.test"
  )
fi

excluded_tests+=(
  # TODO(#12305): figure out how to run samples with custom binary outputs
  # on the CI. $IREE_BINARY_DIR may not be setup right or the object files may
  # not be getting deployed to the test_all/test_gpu bots.
  "iree/samples/custom_dispatch/cpu/embedded/example_hal.mlir.test"
  "iree/samples/custom_dispatch/cpu/embedded/example_stream.mlir.test"
  "iree/samples/custom_dispatch/cpu/embedded/example_transform.mlir.test"
)

###############################################################################
# Run tests                                                                   #
###############################################################################

ctest_args=(
  "--test-dir ${TEST_BUILD_DIR?}"
  "--timeout 900"
  "--output-on-failure"
  "--no-tests=error"
)

if [[ -n "${IREE_CTEST_TESTS_REGEX}" ]]; then
  ctest_args+=("--tests-regex ${IREE_CTEST_TESTS_REGEX}")
fi
if [[ -n "${IREE_CTEST_LABEL_REGEX}" ]]; then
  ctest_args+=("--label-regex ${IREE_CTEST_LABEL_REGEX}")
fi

if (( ${#label_exclude_args[@]} )); then
  # Join on "|"
  label_exclude_regex="($(IFS="|" ; echo "${label_exclude_args[*]}"))"
  ctest_args+=("--label-exclude ${label_exclude_regex}")
fi

if (( ${#excluded_tests[@]} )); then
  # Prefix with `^` anchor
  excluded_tests=( "${excluded_tests[@]/#/^}" )
  # Suffix with `$` anchor
  excluded_tests=( "${excluded_tests[@]/%/$}" )
  # Join on `|` and wrap in parens
  excluded_tests_regex="($(IFS="|" ; echo "${excluded_tests[*]?}"))"
  ctest_args+=("--exclude-regex ${excluded_tests_regex}")
fi

echo "*************** Running CTest ***************"
set -x
ctest ${ctest_args[@]}
