# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO(scotttodd): override these?
#   or move out of compiler/plugins/ to iree/samples where that doesn't apply
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tools")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

set(_NAME "iree_compiler_plugins_target_ROCM_test_opt_pass_plugin_GPUHello")
add_library(${_NAME} SHARED
  "GPUHello.cpp"
)
target_link_libraries(${_NAME}
  iree_compiler_API_Impl
)

# NOTE: this is only required because we want this sample to run on all
# platforms without needing to change the library name (libfoo.so/foo.dll).
set_target_properties(${_NAME}
  PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    PREFIX ""
    OUTPUT_NAME "GPUHello"
)

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "gpu_hello.mlir"
  TOOLS
    FileCheck
    iree-opt
)
