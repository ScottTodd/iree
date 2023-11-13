# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "input_tosa")
set(IREE_COMPILER_TABLEGEN_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}")

add_library(input_tosa_compiler_defs INTERFACE)
target_include_directories(input_tosa_compiler_defs
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# Configures all iree_cc_* targets to take this implicit dep,
# which provides common includes and copts for the tree.
set(IREE_IMPLICIT_DEFS_CC_DEPS input_tosa_compiler_defs)

iree_add_all_subdirs()

iree_compiler_register_plugin(
  PLUGIN_ID
    input_tosa
  TARGET
    ::registration
)

iree_cc_library(
  NAME
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    MLIRIR
    MLIRPass
    MLIRTosaDialect
    MLIRTosaToArith
    MLIRTosaToLinalg
    MLIRTosaToSCF
    MLIRTosaToTensor
    MLIRTransforms
    iree::compiler::PluginAPI
    input_tosa::InputConversion::InputConversion
  PUBLIC
)
